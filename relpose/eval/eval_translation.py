import json
import os
import os.path as osp

import numpy as np
import torch
from pytorch3d.renderer import FoVPerspectiveCameras
from tqdm.auto import tqdm

from models.util import get_model
from utils import (
    compute_optimal_alignment,
    compute_optimal_translation_alignment,
    get_permutations,
)

from .eval_rotation import ORDER_PATH
from .util import get_dataset


def get_n_consistent_cameras(R_pred, num_frames):
    R_pred_n = torch.zeros(num_frames, 3, 3)
    R_pred_n[0] = torch.eye(3)
    for k, (i, j) in enumerate(get_permutations(num_frames, eval_time=True)):
        if i == 0:
            R_pred_n[j] = R_pred[k]

    return R_pred_n


def full_scene_scale(batch):
    # Calculate centroid of cameras in batch
    cameras = FoVPerspectiveCameras(R=batch["R"], T=batch["T"], device="cuda")
    cc = cameras.get_camera_center()
    centroid = torch.mean(cc, dim=0)

    # Determine distance from centroid to each camera
    diffs = cc - centroid
    norms = torch.linalg.norm(diffs, dim=1)

    # Scene scale is the distance from the centroid to the furthest camera
    furthest_index = torch.argmax(norms).item()
    scale = norms[furthest_index].item()
    return scale


def get_error(mode, R_pred, T_pred, R_gt, T_gt, gt_scene_scale):
    if mode == "cc":
        # Get ground truth and predicted camera centers
        cameras_gt = FoVPerspectiveCameras(R=R_gt, T=T_gt, device="cuda")
        cc_gt = cameras_gt.get_camera_center()
        cameras_pred = FoVPerspectiveCameras(R=R_pred, T=T_pred, device="cuda")
        cc_pred = cameras_pred.get_camera_center()

        # Align the ground truth and predicted camera centers as closely as possible
        A_hat, _, _, _ = compute_optimal_alignment(cc_gt, cc_pred)
        norm = torch.linalg.norm(cc_gt - A_hat, dim=1) / gt_scene_scale

        # Return distances between corresponding ground truth and predicted
        # camera centers
        norms = np.ndarray.tolist(norm.detach().cpu().numpy())
        return norms, A_hat
    elif mode == "t":
        T_A_hat, _, _ = compute_optimal_translation_alignment(T_gt, T_pred, R_pred)
        norm = torch.linalg.norm(T_gt - T_A_hat, dim=1) / gt_scene_scale
        norms = np.ndarray.tolist(norm.detach().cpu().numpy())
        return norms, T_A_hat


def evaluate_category_translation(
    checkpoint_path,
    category,
    mode,
    num_frames,
    use_pbar=False,
    save_dir=None,
    force=False,
    sample_num=0,
    **kwargs,
):
    # Default save directory
    folder_path = f"eval/{mode}-{num_frames:03d}-sample{sample_num}"
    os.makedirs(os.path.join(checkpoint_path, folder_path), exist_ok=True)

    # Check not already existing results
    path = osp.join(checkpoint_path, f"{folder_path}/{category}.json")
    if osp.exists(path) and not force:
        print(f"{path} already exists, skipping")
        with open(path, "r") as f:
            data = json.load(f)
        angular_errors = []
        for d in data.values():
            angular_errors.extend(d["errors"])
        return np.array(angular_errors)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = get_model(model_dir=checkpoint_path, device=device)

    # Load dataset using annotation camera frame
    dataset = get_dataset(
        category=category,
        num_images=num_frames,
        eval_time=True,
    )

    iterable = tqdm(dataset) if use_pbar else dataset
    f = open(order_path.format(sample_num=sample_num, category=category))
    order = json.load(f)
    all_errors = {}
    translation_errors = []
    for metadata in iterable:
        # Calculate scene scale in annotation frame
        sequence_name = metadata["model_id"]
        key_frames = order[sequence_name][:num_frames]
        all_cams_batch = dataset.get_data(
            sequence_name=sequence_name, ids=np.arange(0, metadata["n"]), no_images=True
        )
        gt_scene_scale = full_scene_scale(all_cams_batch)

        # Get model inputs and ground truth pose
        batch = dataset.get_data(sequence_name=sequence_name, ids=key_frames)
        images = batch["image"].to(device).unsqueeze(0)
        crop_params = batch["crop_params"].to(device).unsqueeze(0)
        R_gt = batch["R"].to(device)
        T_gt = batch["T"].to(device)

        # Model forward pass
        with torch.no_grad():
            out = model(
                images=images,
                crop_params=crop_params,
            )
            _, _, T_pred = out

        # Read rotations from input directory
        f = open(
            os.path.join(
                checkpoint_path,
                f"eval/coordinate_ascent-{num_frames:03d}-sample{sample_num}",
                f"{category}.json",
            )
        )
        rotations_json = json.load(f)
        R_pred = torch.from_numpy(
            np.asarray(rotations_json[sequence_name]["R_pred_rel"])
        )
        R_pred_n = get_n_consistent_cameras(R_pred, num_frames).to(images.device)

        # Calculate translation error
        norms, A_hat = get_error(mode, R_pred_n, T_pred, R_gt, T_gt, gt_scene_scale)

        # Append information to be saved
        translation_errors.extend(norms)
        all_errors[sequence_name] = {
            "R_pred": R_pred_n.tolist(),
            "T_pred": T_pred.tolist(),
            "errors": norms,
            "scale": gt_scene_scale,
            "A_hat": A_hat.tolist(),
        }

    # Save to file
    if save_dir is not None:
        with open(path, "w") as f:
            json.dump(all_errors, f)

    print(np.average(np.array(translation_errors) < 0.2))
    return np.array(translation_errors)
