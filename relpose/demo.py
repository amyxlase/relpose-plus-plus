"""
Demo script for running RelPose++. RelPose++ can be run on 2-8 images. For the
commandline demo, each image must be associated with a bounding box. These bounding
boxes can either be provided as a JSON file or computed using a directory of masks.

The cameras and images will be visualized in an html file, which can be viewed in any
web browser.

Example using masks:
    python relpose/demo.py  --image_dir examples/robot/images \
        --mask_dir examples/robot/masks --output_path robot.html

Example using bounding boxes:
    python relpose/demo.py  --image_dir examples/robot/images \
        --bbox_path examples/robot/bboxes.json --output_path robot.html
"""

import argparse
import base64
import io
import json
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly
import torch
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene

from dataset import CustomDataset
from eval import evaluate_coordinate_ascent, evaluate_mst
from models import get_model
from utils import view_color_coded_images_from_tensor

HTML_TEMPLATE = """<html><head><meta charset="utf-8"/></head>
<body><img src="data:image/png;charset=utf-8;base64,{image_encoded}"/>
{plotly_html}</body></html>"""


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="examples/robot/images")
    parser.add_argument("--model_dir", type=str, default="weights/relposepp")
    parser.add_argument("--mask_dir", type=str, default="")
    parser.add_argument("--bbox_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="output_cameras.html")
    return parser


def plotly_scene_visualization(R_pred_mst, R_pred_coord_asc, T_pred):
    num_frames = len(R_pred_mst)

    scenes = {
        "Initial Predicted Cameras": {},
        "Final Optimized Cameras": {},
    }

    for i in range(num_frames):
        scenes["Initial Predicted Cameras"][i] = FoVPerspectiveCameras(
            R=R_pred_mst[i, None], T=T_pred[i, None]
        )
        scenes["Final Optimized Cameras"][i] = FoVPerspectiveCameras(
            R=R_pred_coord_asc[i, None], T=T_pred[i, None]
        )

    fig = plot_scene(
        scenes,
        camera_scale=0.03,
        ncols=2,
    )
    fig.update_scenes(aspectmode="data")

    cmap = plt.get_cmap("hsv")
    for i in range(num_frames):
        fig.data[i].line.color = matplotlib.colors.to_hex(cmap(i / (num_frames)))
        fig.data[i + num_frames].line.color = matplotlib.colors.to_hex(
            cmap(i / (num_frames))
        )

    return fig


def main(image_dir, model_dir, mask_dir, bbox_path, output_path):
    device = torch.device("cuda:0")
    model, args = get_model(model_dir, device=device, num_images=8)
    if osp.exists(bbox_path):
        bboxes = json.load(open(bbox_path))
    else:
        bboxes = None
    dataset = CustomDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        bboxes=bboxes,
        mask_images=args.get("mask_images", False),
    )
    num_frames = dataset.n
    batch = dataset.get_data(ids=np.arange(num_frames))
    images = batch["image"].to(device)
    crop_params = batch["crop_params"].to(device)

    if num_frames > 8:
        print("Warning: model expects at most 8 images.")

    # Quickly initialize a coarse set of poses using MST reasoning
    batched_images, batched_crop_params = images.unsqueeze(0), crop_params.unsqueeze(0)

    print("Computing initial MST solution.")
    with torch.no_grad():
        _, hypothesis = evaluate_mst(
            model=model,
            images=batched_images,
            crop_params=batched_crop_params,
        )
    R_pred = np.stack(hypothesis)

    # Regress to optimal translation
    with torch.no_grad():
        _, _, T_pred = model(images=batched_images, crop_params=batched_crop_params)

    print("Computing coordinate ascent solution.")

    # Search for optimal rotation via coordinate ascent.
    with torch.no_grad():
        _, hypothesis = evaluate_coordinate_ascent(
            model=model,
            images=batched_images,
            crop_params=batched_crop_params,
        )
    R_final = np.stack(hypothesis)

    # Visualize cropped and resized images
    fig = plotly_scene_visualization(R_pred, R_final, T_pred)
    html_plot = plotly.io.to_html(fig, full_html=False, include_plotlyjs="cdn")
    s = io.BytesIO()
    view_color_coded_images_from_tensor(images)
    plt.savefig(s, format="png", bbox_inches="tight")
    plt.close()
    image_encoded = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    with open(output_path, "w") as f:
        s = HTML_TEMPLATE.format(
            image_encoded=image_encoded,
            plotly_html=html_plot,
        )
        f.write(s)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
