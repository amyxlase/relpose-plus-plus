"""
If training from the start, first get the output directory name by running:
    $ python -m relpose/trainer_ddp --generate_name
    > output/...

Then, run with --resume flag to use the new output directory:
    $ torchrun --nnodes=1 --nproc_per_node=4 relpose/trainer_diffusion.py --resume output/...
"""

import argparse
import datetime
import json
import os
import os.path as osp
import shutil
import time
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from termcolor import colored
from torch.nn import L1Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from dataset import TRAINING_CATEGORIES, get_dataloader
from models import RelPose
from utils import get_permutations, make_visualization_n

matplotlib.use("Agg")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", nargs="+", type=str, default=["all"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_iterations", type=int, default=800_000)
    parser.add_argument("--interval_checkpoint", type=int, default=1000)
    parser.add_argument("--interval_delete_checkpoint", type=int, default=10000)
    parser.add_argument("--interval_visualize", type=int, default=1000)
    parser.add_argument("--interval_evaluate", type=int, default=25000)
    parser.add_argument("--num_visualize", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--dataset", type=str, default="co3d")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--resume", default="", type=str, help="Path to directory.")
    parser.add_argument("--gpu_ids", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--random_num_images", type=bool, default=False)
    parser.add_argument(
        "--normalize_cameras",
        action="store_true",
        help="Place intersection of optical axes at origin, re-scale so that T1 has unit norm",
    )
    parser.add_argument(
        "--first_camera_transform",
        action="store_true",
        help="Transform cameras so that first camera is [I | 0] (after normalizing)",
    )
    parser.add_argument("--generate_name", action="store_true")
    parser.add_argument(
        "--use_tf32",
        action="store_true",
        help="Switch to TF32 Floats (could have slightly better performance on Ampere)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use AMP for reduced memory (seems to be slightly slower for some reason)",
    )
    return parser


def generate_name(args):
    # Setup output directory.
    name = datetime.datetime.now().strftime("%m%d_%H%M")
    if args.category[0] != "all":
        name += "_" + "_".join(args.category)
    name += f"_LR{args.lr}"
    name += f"_N{args.num_images}"
    name += f"_RandomN{args.random_num_images}"
    name += f"_B{args.batch_size}"
    if args.pretrained:
        # First 9 characters **should** be give the date time of the model.
        name += "_Pretrained" + osp.basename(args.pretrained)[:9]
    if args.use_amp:
        name += "_AMP"
    if args.first_camera_transform:
        name += "_TRFIRST"
    else:
        name += "_TROURS"
    name += "_DDP"
    return osp.join(args.output_dir, name)


class Trainer(object):
    def __init__(self, args) -> None:
        self.args = args
        self.num_iterations = int(args.num_iterations)
        self.lr = args.lr
        self.category = args.category
        self.interval_visualize = args.interval_visualize
        self.interval_checkpoint = args.interval_checkpoint
        self.interval_delete_checkpoint = args.interval_delete_checkpoint
        self.interval_evaluate = args.interval_evaluate
        assert self.interval_delete_checkpoint % self.interval_checkpoint == 0
        self.debug = args.debug
        self.num_images = args.num_images
        self.random_num_images = args.random_num_images
        self.gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        self.num_gpus = len(self.gpu_ids)
        assert args.batch_size % self.num_gpus == 0
        self.batch_size = args.batch_size // self.num_gpus
        self.num_visualize = min(self.batch_size, args.num_visualize)
        self.amp = args.use_amp
        self.normalize_cameras = args.normalize_cameras
        self.first_camera_transform = args.first_camera_transform

        self.iteration = 0
        self.epoch = 0

        num_workers = self.num_gpus * 4
        if self.category[0] == "all":
            self.category = TRAINING_CATEGORIES

        self.dataloader = get_dataloader(
            category=self.category,
            dataset=args.dataset,
            split="train",
            batch_size=self.batch_size,
            num_workers=num_workers,
            debug=self.debug,
            num_images=self.num_images,
            random_num_images=args.random_num_images,
            normalize_cameras=self.normalize_cameras,
            first_camera_transform=self.first_camera_transform,
            img_size=224,
        )

        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = RelPose(
            num_layers=4,
            num_pe_bases=8,
            hidden_size=256,
            num_queries=36864,
            num_images=self.num_images,
        )

        self.translation_regressor = self.net.translation_regressor

        print("ddp initialization")

        # DDP Initialization
        dist.init_process_group("nccl", init_method="env://", world_size=self.num_gpus)
        dist.barrier()

        self.rank = dist.get_rank()
        device_id = self.rank % torch.cuda.device_count()
        self.net.to(device_id)
        torch.cuda.set_device(device_id)

        self.net = DDP(self.net, device_ids=[device_id], find_unused_parameters=False)

        print(f"Process {self.rank} on device {device_id}")

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        if args.pretrained != "" and not osp.exists(args.resume):
            checkpoint_dir = osp.join(args.pretrained, "checkpoints")
            last_checkpoint = sorted(os.listdir(checkpoint_dir))[-1]
            checkpoint = torch.load(
                osp.join(checkpoint_dir, last_checkpoint),
                map_location=self.device,
            )
            if "state_dict" in checkpoint:
                feature_extractor_state_dict = checkpoint["state_dict"]
                missing_keys, unexpected_keys = self.net.load_state_dict(
                    feature_extractor_state_dict, strict=False
                )

                print(f"Missing keys: {missing_keys}")
                print(f"Unexpected keys: {unexpected_keys}")
                keys1 = self.net.state_dict().keys()
                keys2 = feature_extractor_state_dict.keys()
                matched_keys = set(keys1).intersection(keys2)
                print("Matched keys", colored(str(matched_keys), "green"))

            del checkpoint, feature_extractor_state_dict
            torch.cuda.empty_cache()

        if osp.exists(args.resume):
            self.output_dir = args.resume
            self.checkpoint_dir = osp.join(self.output_dir, "checkpoints")
            last_checkpoint = sorted(os.listdir(self.checkpoint_dir))[-1]
            self.load_model(osp.join(self.checkpoint_dir, last_checkpoint))
        else:
            self.output_dir = generate_name(args) if args.resume == "" else args.resume
            self.start_time = time.time()
            self.checkpoint_dir = osp.join(self.output_dir, "checkpoints")

        dist.barrier()
        # Only rank 0 writes
        if self.rank == 0:
            if not osp.exists(args.resume):
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                with open(osp.join(self.output_dir, "args.json"), "w") as f:
                    json.dump(vars(args), f)
                # Make a copy of the code.
                shutil.copytree("relpose", osp.join(self.output_dir, "relpose"))
                print("Output Directory:", self.output_dir)

            # Setup tensorboard.
            self.writer = SummaryWriter(log_dir=self.output_dir, flush_secs=10)

    def debug_print(self, *s):
        torch.set_printoptions(precision=10)
        if self.rank == 0:
            st = " ".join(map(str, s))
            print(f"{self.iteration}: {st}")

    def train(self):
        print("Begin training")

        while self.iteration < self.num_iterations:
            for batch in self.dataloader:
                self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    enabled=self.amp, device_type="cuda", dtype=torch.float16
                ):
                    # Inputs & ground truths
                    images = batch["image"].to(self.device, non_blocking=True)
                    relative_rotations = batch["relative_rotation"].to(
                        self.device, non_blocking=True
                    )
                    crop_params = batch["crop_params"].to(
                        self.device, non_blocking=True
                    )
                    truth = batch["T"].to(self.device, non_blocking=True)

                    # Handle variable number of images
                    n = self.num_images
                    if self.random_num_images:
                        n = np.random.randint(2, self.num_images + 1)
                        n_c = len(get_permutations(n))
                        images = images[:, :n]
                        relative_rotations = relative_rotations[:, :n_c]
                        crop_params = crop_params[:, :n]
                        truth = truth[:, :n]

                    # Forward pass
                    (queries, logits, predicted_translations,) = self.net(
                        images=images,
                        gt_rotation=relative_rotations,
                        crop_params=crop_params,
                        take_softmax=False,
                    )

                    # Rotation loss
                    loss_rot = -torch.mean(torch.log_softmax(logits, dim=-1)[:, :, 0])

                    # Translation loss
                    num_batches = images.shape[0]
                    predicted_translations = predicted_translations.reshape(
                        num_batches * n, 3
                    )
                    truth = truth.reshape(num_batches * n, 3)
                    l1 = L1Loss()
                    loss_trans = l1(predicted_translations, truth)

                loss = loss_rot + loss_trans

                if self.amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.iteration += 1

                # Rank 0 save model at checkpoint, then everyone else load using map
                if self.iteration % self.interval_checkpoint == 0:
                    checkpoint_path = osp.join(
                        self.checkpoint_dir, f"ckpt_{self.iteration:09d}.pth"
                    )
                    if self.rank == 0:
                        self.save_model(checkpoint_path)

                    dist.barrier()
                    map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank}
                    self.net.load_state_dict(
                        torch.load(checkpoint_path, map_location=map_location)[
                            "state_dict"
                        ]
                    )

                # Only rank 0 does housekeeping
                if self.rank == 0:
                    # Print iteration info to terminal
                    if self.iteration % 20 == 0:
                        self.log_info(loss, loss_rot, loss_trans)
                        self.writer.add_scalar(
                            "Loss/train", loss.item(), self.iteration
                        )
                        self.writer.add_scalar(
                            "Loss/rotations", loss_rot.item(), self.iteration
                        )
                        self.writer.add_scalar(
                            "Loss/translations", loss_trans.item(), self.iteration
                        )

                    # Log visualization to board
                    if self.iteration % self.interval_visualize == 0 and n < 5:
                        images = images[: self.num_visualize]
                        relative_rotations = relative_rotations[: self.num_visualize]
                        probabilities = torch.softmax(logits, dim=-1)

                        self.write_visualizations(
                            images=images,
                            rotations=queries,
                            probabilities=probabilities,
                            rotations_gt=relative_rotations,
                            model_id=batch["model_id"],
                            category=batch["category"],
                            ind=batch["ind"],
                        )
                        self.save_histograms(predicted_translations, truth, loss_trans)

                    # Clear old checkpoints
                    if self.iteration % self.interval_delete_checkpoint == 0:
                        self.clear_old_checkpoints(self.checkpoint_dir)

                if self.iteration >= self.num_iterations + 1:
                    break

            self.epoch += 1

    def save_model(self, path):
        elapsed = time.time() - self.start_time if self.start_time is not None else 0
        save_dict = {
            "state_dict": self.net.state_dict(),
            "iteration": self.iteration,
            "epoch": self.epoch,
            "elapsed": elapsed,
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(save_dict, path)

    def load_model(self, path, load_metadata=True):
        save_dict = torch.load(path, map_location=self.device)
        if "state_dict" in save_dict:
            missing, unexpected = self.net.load_state_dict(
                save_dict["state_dict"],
                strict=False,
            )
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
            if load_metadata:
                self.iteration = save_dict["iteration"]
                self.epoch = save_dict["epoch"]
                if "elapsed" in save_dict:
                    time_elapsed = save_dict["elapsed"]
                    self.start_time = time.time() - time_elapsed
            self.optimizer.load_state_dict(save_dict["optimizer"])
        else:
            self.net.load_state_dict(save_dict)

    def clear_old_checkpoints(self, checkpoint_dir):
        print("Clearing old checkpoints")
        checkpoint_files = glob(osp.join(checkpoint_dir, "ckpt_*.pth"))
        for checkpoint_file in checkpoint_files:
            checkpoint = osp.basename(checkpoint_file)
            checkpoint_iteration = int("".join(filter(str.isdigit, checkpoint)))
            if checkpoint_iteration % self.interval_delete_checkpoint != 0:
                os.remove(checkpoint_file)

    def log_info(self, loss, loss_rot, loss_trans):
        if self.start_time is None:
            self.start_time = time.time()
        time_elapsed = np.round(time.time() - self.start_time)
        time_remaining = np.round(
            (time.time() - self.start_time)
            / (self.iteration + 1)
            * (self.num_iterations - self.iteration)
        )

        disp = [
            f"Iter: {self.iteration:d}/{self.num_iterations:d}",
            f"Epoch: {self.epoch:d}",
            f"Loss: {loss.item():.4f}",
            f"Loss Rot: {loss_rot.item():.4f}",
            f"Loss Trans: {loss_trans.item():.4f}",
            f"LR: {self.lr:e}",
            f"Elap: {str(datetime.timedelta(seconds=time_elapsed))}",
            f"Rem: {str(datetime.timedelta(seconds=time_remaining))}",
            f"Output Dir: {self.output_dir}",
        ]
        print(", ".join(disp))

    # Need to join paths safely
    def save_histograms(self, predicted, truth, loss):
        predicted = predicted.detach().cpu().numpy()
        truth = truth.detach().cpu().numpy()

        fig = plt.figure(num=1, figsize=(12, 6))

        ax1 = fig.add_subplot(161)
        ax2 = fig.add_subplot(162)
        ax3 = fig.add_subplot(163)
        ax4 = fig.add_subplot(164)
        ax5 = fig.add_subplot(165)
        ax6 = fig.add_subplot(166)

        ax1.set_title("X Truth")
        _, bins, _ = ax1.hist(truth[:, 0], bins=100, alpha=0.5, color="#FF0000")
        _ = ax1.hist(predicted[:, 0], bins=bins, alpha=0.5, color="#00FF00")

        ax2.set_title("X Prediction")
        _ = ax2.hist(predicted[:, 0], bins=100, alpha=0.5, color="#00FF00")

        ax3.set_title("Y Truth")
        _, bins, _ = ax3.hist(truth[:, 1], bins=100, alpha=0.5, color="#FF0000")
        _ = ax3.hist(predicted[:, 1], bins=bins, alpha=0.5, color="#00FF00")

        ax4.set_title("Y Prediction")
        _ = ax4.hist(predicted[:, 1], bins=100, alpha=0.5, color="#00FF00")

        ax5.set_title("Z Truth")
        _, bins, _ = ax5.hist(truth[:, 2], bins=100, alpha=0.5, color="#FF0000")
        _ = ax5.hist(predicted[:, 2], bins=bins, alpha=0.5, color="#00FF00")

        ax6.set_title("Z Prediction")
        _ = ax6.hist(predicted[:, 2], bins=100, alpha=0.5, color="#00FF00")

        fig.suptitle(
            f"Iteration: {self.iteration} Number of Translations: {truth.shape[0]} Loss: {loss}"
        )

        self.writer.add_figure("Translation Visualization", fig, self.iteration)

        fig.clear()
        plt.close(fig)

    def write_visualizations(
        self,
        images,
        rotations,
        probabilities,
        rotations_gt,
        model_id=None,
        category=None,
        ind=None,
        best_pred=None,
    ):
        visuals2d = make_visualization_n(
            images=images,
            rotations=rotations,
            probabilities=probabilities,
            gt_rotations=rotations_gt,
            model_id=model_id,
            category=category,
            ind=ind,
        )

        for batch in range(self.num_visualize):
            images = []
            for im in range(len(visuals2d)):
                images.append(visuals2d[im][batch])
            full_image = np.hstack(images)
            self.writer.add_image(
                f"Visualization {batch}", full_image, self.iteration, dataformats="HWC"
            )
            print("Visualizing for batch")


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.generate_name:
        name = generate_name(args)
        print(name)
    else:
        torch.backends.cudnn.benchmark = True
        if args.use_tf32:
            torch.set_float32_matmul_precision("medium")
        trainer = Trainer(args)
        trainer.train()
