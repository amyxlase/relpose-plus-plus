"""
python resources/get_tables.py --directory resources/co3d_result
"""
import argparse
import os.path as osp
from glob import glob


TABLE_HTML = """
<div class="item-{ind}">
    <table width="400px" class="border">
        <tr>
            <td>
                <video autoplay loop muted playsinline controls width="100%">
                    <source src="{video_path}"
                        type="video/mp4">
                </video>
            </td>
        </tr>
        <tr>
            <td>
                <img src="{image_path}" width="100%"></img>
            </td>
        </tr>
    </table>
</div>
"""


def get_tables(directory):
    images = sorted(glob(osp.join(directory, "*.jpg")))
    videos = sorted(glob(osp.join(directory, "*.mp4")))
    assert len(images) == len(videos)
    for i, (image, video) in enumerate(zip(images, videos)):
        print(TABLE_HTML.format(ind=i + 1, image_path=image, video_path=video))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True)
    args = parser.parse_args()
    get_tables(args.directory)
