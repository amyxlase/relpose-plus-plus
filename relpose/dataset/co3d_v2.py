import gzip
import json
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image, ImageFile
from pytorch3d.renderer import PerspectiveCameras
from torch.utils.data import Dataset
from torchvision import transforms

from utils.bbox import square_bbox
from utils.misc import get_permutations
from utils.normalize_cameras import first_camera_transform, normalize_cameras

CO3D_DIR = "data/co3d_v2"
CO3D_ANNOTATION_DIR = "data/co3d_v2_annotations"

TRAINING_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

TEST_CATEGORIES = [
    "ball",
    "book",
    "couch",
    "frisbee",
    "hotdog",
    "kite",
    "remote",
    "sandwich",
    "skateboard",
    "suitcase",
]

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Co3dDataset(Dataset):
    def __init__(
        self,
        category=("all",),
        split="train",
        transform=None,
        debug=False,
        random_aug=True,
        jitter_scale=[1.1, 1.2],
        jitter_trans=[-0.07, 0.07],
        num_images=2,
        img_size=224,
        random_num_images=True,
        eval_time=False,
        normalize_cameras=False,
        first_camera_transform=False,
        first_camera_rotation_only=False,
        mask_images=False,
    ):
        """
        Args:
            category (iterable): List of categories to use. If "all" is in the list,
                all training categories are used.
            num_images (int): Default number of images in each batch.
            normalize_cameras (bool): If True, normalizes cameras so that the
                intersection of the optical axes is placed at the origin and the norm
                of the first camera translation is 1.
            first_camera_transform (bool): If True, tranforms the cameras such that
                camera 1 has extrinsics [I | 0].
            first_camera_rotation_only (bool): If True, transforms the cameras such that
                camera 1 has identity rotation.
            mask_images (bool): If True, masks out the background of the images.
        """
        if "all" in category:
            category = TRAINING_CATEGORIES
        category = sorted(category)

        if split == "train":
            split_name = "train"
        elif split == "test":
            split_name = "test"

        if eval_time:
            torch.manual_seed(0)
            random.seed(0)
            np.random.seed(0)

        self.low_quality_translations = []
        self.rotations = {}
        self.category_map = {}
        for c in category:
            annotation_file = osp.join(CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz")
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())

            counter = 0
            for seq_name, seq_data in annotation.items():
                counter += 1
                if len(seq_data) < num_images:
                    continue

                filtered_data = []
                self.category_map[seq_name] = c
                bad_seq = False
                for data in seq_data:
                    # Make sure translations are not ridiculous
                    if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
                        bad_seq = True
                        self.low_quality_translations.append(seq_name)
                        break

                    # Ignore all unnecessary information.
                    filtered_data.append(
                        {
                            "filepath": data["filepath"],
                            "bbox": data["bbox"],
                            "R": data["R"],
                            "T": data["T"],
                            "focal_length": data["focal_length"],
                            "principal_point": data["principal_point"],
                        },
                    )

                if not bad_seq:
                    self.rotations[seq_name] = filtered_data

            print(annotation_file)
            print(counter)

        self.sequence_list = list(self.rotations.keys())
        self.split = split
        self.debug = debug

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(224),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform
        if random_aug and not eval_time:
            self.jitter_scale = jitter_scale
            self.jitter_trans = jitter_trans
        else:
            self.jitter_scale = [1.15, 1.15]
            self.jitter_trans = [0, 0]

        self.random_num_images = random_num_images
        self.num_images = num_images
        self.image_size = img_size
        self.eval_time = eval_time
        self.normalize_cameras = normalize_cameras
        self.first_camera_transform = first_camera_transform
        self.first_camera_rotation_only = first_camera_rotation_only
        self.mask_images = mask_images

        print(
            f"Low quality translation sequences, not used: {self.low_quality_translations}"
        )
        print(f"Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def _jitter_bbox(self, bbox):
        bbox = square_bbox(bbox.astype(np.float32))

        s = np.random.uniform(self.jitter_scale[0], self.jitter_scale[1])
        tx, ty = np.random.uniform(self.jitter_trans[0], self.jitter_trans[1], size=2)

        side_length = bbox[2] - bbox[0]
        center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side_length
        extent = side_length / 2 * s

        # Final coordinates need to be integer for cropping.
        ul = (center - extent).round().astype(int)
        lr = ul + np.round(2 * extent).astype(int)
        return np.concatenate((ul, lr))

    def _crop_image(self, image, bbox, white_bg=False):
        if white_bg:
            # Only support PIL Images
            image_crop = Image.new(
                "RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255)
            )
            image_crop.paste(image, (-bbox[0], -bbox[1]))
        else:
            image_crop = transforms.functional.crop(
                image,
                top=bbox[1],
                left=bbox[0],
                height=bbox[3] - bbox[1],
                width=bbox[2] - bbox[0],
            )
        return image_crop

    def __getitem__(self, index):
        sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        ids = np.random.choice(len(metadata), self.num_images, replace=False)
        return self.get_data(index=index, ids=ids)

    def get_data(self, index=None, sequence_name=None, ids=(0, 1), no_images=False):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        category = self.category_map[sequence_name]

        if no_images:
            annos = [metadata[i] for i in ids]
            rotations = [torch.tensor(anno["R"]) for anno in annos]
            translations = [torch.tensor(anno["T"]) for anno in annos]
            batch = {}
            batch["R"] = torch.stack(rotations)
            batch["T"] = torch.stack(translations)
            return batch

        annos = [metadata[i] for i in ids]
        images = []
        rotations = []
        translations = []
        focal_lengths = []
        principal_points = []
        for anno in annos:
            filepath = anno["filepath"]

            image = Image.open(osp.join(CO3D_DIR, filepath)).convert("RGB")
            if self.mask_images:
                white_image = Image.new("RGB", image.size, (255, 255, 255))
                mask_name = osp.basename(filepath.replace(".jpg", ".png"))

                mask_path = osp.join(
                    CO3D_DIR, category, sequence_name, "masks", mask_name
                )
                mask = Image.open(mask_path).convert("L")

                if mask.size != image.size:
                    mask = mask.resize(image.size)
                mask = Image.fromarray(np.array(mask) > 125)
                image = Image.composite(image, white_image, mask)
            images.append(image)
            rotations.append(torch.tensor(anno["R"]))
            translations.append(torch.tensor(anno["T"]))
            focal_lengths.append(torch.tensor(anno["focal_length"]))
            principal_points.append(torch.tensor(anno["principal_point"]))

        crop_parameters = []

        images_transformed = []
        for i, (anno, image) in enumerate(zip(annos, images)):
            if self.transform is None:
                images_transformed.append(image)
            else:
                w, h = image.width, image.height
                bbox = np.array(anno["bbox"])
                bbox_jitter = self._jitter_bbox(bbox)

                image = self._crop_image(image, bbox_jitter, white_bg=self.mask_images)
                images_transformed.append(self.transform(image))

                crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
                cc = (2 * crop_center / min(h, w)) - 1
                crop_width = 2 * (bbox_jitter[2] - bbox_jitter[0]) / min(h, w)

                crop_parameters.append(
                    torch.tensor([-cc[0], -cc[1], crop_width]).float()
                )

        images = images_transformed

        batch = {
            "model_id": sequence_name,
            "category": category,
            "n": len(metadata),
            "ind": torch.tensor(ids),
        }

        if self.normalize_cameras:
            cameras = PerspectiveCameras(
                focal_length=[data["focal_length"] for data in annos],
                principal_point=[data["principal_point"] for data in annos],
                R=[data["R"] for data in annos],
                T=[data["T"] for data in annos],
            )

            normalized_cameras, _, _, _, _ = normalize_cameras(cameras)

            if self.first_camera_transform or self.first_camera_rotation_only:
                normalized_cameras = first_camera_transform(
                    normalized_cameras,
                    rotation_only=self.first_camera_rotation_only,
                )

            if normalized_cameras == -1:
                print("Error in normalizing cameras: camera scale was 0")
                assert False

            batch["R"] = normalized_cameras.R
            batch["T"] = normalized_cameras.T
            batch["crop_params"] = torch.stack(crop_parameters)
            batch["R_original"] = torch.stack(
                [torch.tensor(anno["R"]) for anno in annos]
            )
            batch["T_original"] = torch.stack(
                [torch.tensor(anno["T"]) for anno in annos]
            )

            if torch.any(torch.isnan(batch["T"])):
                print(ids)
                print(category)
                print(sequence_name)
                assert False

        else:
            batch["R"] = torch.stack(rotations)
            batch["T"] = torch.stack(translations)
            batch["crop_params"] = torch.stack(crop_parameters)

        # Add relative rotations
        permutations = get_permutations(len(ids), eval_time=self.eval_time)
        n_p = len(permutations)
        relative_rotation = torch.zeros((n_p, 3, 3))
        for k, t in enumerate(permutations):
            i, j = t
            relative_rotation[k] = rotations[i].T @ rotations[j]
        batch["relative_rotation"] = relative_rotation

        # Add images
        if self.transform is None:
            batch["image"] = images
        else:
            batch["image"] = torch.stack(images)

        return batch
