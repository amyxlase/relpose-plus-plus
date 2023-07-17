import json
import os
import os.path as osp

import torch

from .models import RelPose


# Checkpoint dir must be folder with checkpoints and args.json
def get_model(
    model_dir,
    device="cuda:0",
    num_images=8,
):
    # Load weights and args
    checkpoint_dir = osp.join(model_dir, "checkpoints")
    last_checkpoint = sorted(os.listdir(checkpoint_dir))[-1]
    print(f"Loading checkpoint {last_checkpoint}")
    weights_dir = osp.join(checkpoint_dir, last_checkpoint)
    pretrained_weights = torch.load(weights_dir)["state_dict"]

    args_path = osp.join(model_dir, "args.json")
    if osp.exists(args_path):
        with open(args_path, "r") as f:
            args = json.load(f)
    else:
        args = {}

    # Weights may be renamed slightly differently
    pretrained_weights = {
        k.replace("module.", ""): v for k, v in pretrained_weights.items()
    }

    if (
        "feature_extractor.feature_positional_encoding.pos_table_1"
        in pretrained_weights
    ):
        del pretrained_weights[
            "feature_extractor.feature_positional_encoding.pos_table_1"
        ]

    relpose = RelPose(
        num_layers=4,
        num_pe_bases=8,
        hidden_size=256,
        num_images=num_images,
    )
    relpose.to(device)

    # Check incompatible keys
    missing, unexpected = relpose.load_state_dict(pretrained_weights, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    relpose.eval()
    return relpose, args
