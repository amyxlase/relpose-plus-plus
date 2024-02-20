"""
Usually, store full co3d dataset on hdd, but want the images on ssd.
"""
import argparse
import os
import shutil
from glob import glob

from tqdm.auto import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default="/data/drive3/jason/co3d")
    parser.add_argument("--target_dir", type=str, default="/data/drive2/jason/co3d")
    parser.add_argument("--category", default="*")
    return parser


def main(args):
    source_dir = args.source_dir
    target_dir = args.target_dir
    all_image_paths = sorted(
        glob(os.path.join(source_dir, args.category, "*", "images", "*.jpg"))
    )
    for image_path in tqdm(all_image_paths):
        target_image_path = os.path.join(
            target_dir, image_path.replace(source_dir, "")[1:]  # drop the first '/'
        )
        target_image_dir = os.path.dirname(target_image_path)
        if not os.path.exists(target_image_dir):
            os.makedirs(target_image_dir, exist_ok=True)
        shutil.copy(image_path, target_image_path)


if __name__ == "__main__":
    main(get_parser().parse_args())
