import torch

from dataset.co3d_v2 import Co3dDataset


def get_dataloader(
    batch_size=64,
    dataset="co3d",
    category=("table",),
    split="train",
    shuffle=True,
    num_workers=8,
    debug=False,
    num_images=2,
    rank=None,
    world_size=None,
    img_size=224,
    normalize_cameras=False,
    random_num_images=False,
    first_camera_transform=False,
    first_camera_rotation_only=False,
    mask_images=False,
):
    if debug:
        num_workers = 0
    if dataset == "co3d":
        dataset = Co3dDataset(
            category=category,
            split=split,
            num_images=num_images,
            debug=debug,
            img_size=img_size,
            normalize_cameras=normalize_cameras,
            random_num_images=random_num_images,
            first_camera_transform=first_camera_transform,
            first_camera_rotation_only=first_camera_rotation_only,
            mask_images=mask_images,
        )
    else:
        raise Exception(f"Unknown dataset: {dataset}")

    if rank is None:
        sampler = None
    else:
        if world_size is None:
            assert False
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        print(f"Sampler {rank} {world_size}")
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )
