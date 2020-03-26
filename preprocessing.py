import os
import numpy as np
import argparse

from utils.pts import read_pts
from utils.ply import dict2ply

PATH_PREPROCESSED = "data/preprocessed"


def center(coords):
    means = np.mean(coords, axis=0)
    centered = coords - means
    return centered


def preprocess_pts(path, centering, scale):
    print(f"Reading points from {path}")
    data_pts = read_pts(path)

    # center/scale point cloud
    coords = data_pts[:, :3]
    features = data_pts[:, 3:-1].astype(np.uint8)
    labels = data_pts[:, -1].astype(np.uint8)

    if centering:
        coords = center(coords)
    coords = (coords * scale).astype(np.float32)
    data_ply = {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "intensity": features[:, 0],
        "return_number": features[:, 1],
        "number_of_returns": features[:, 2],
        "labels": labels,
    }

    # save preprocessed point cloud
    os.makedirs(PATH_PREPROCESSED, exist_ok=True)
    filename = os.path.split(path)[-1].split(".")[-2]
    path_ply = os.path.join(PATH_PREPROCESSED, filename + ".ply")
    if dict2ply(data_ply, path_ply):
        print(f"PLY point cloud successfully saved to {path_ply}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Center and rescale point cloud"
    )
    parser.add_argument(
        "--files", "-f", type=str, nargs="+", help="Path to point cloud file"
    )
    parser.add_argument(
        "--scale", "-s", type=float, default=1, help="Scale factor"
    )
    parser.add_argument(
        "--centering",
        "-c",
        action="store_true",
        help="Recenter point cloud coordinates",
    )
    args = parser.parse_args()
    # Path of the file

    # Load point cloud
    print(f"Centering : {args.centering}")
    print(f"Scale factor : {args.scale}")
    for path in args.files:
        preprocess_pts(path, args.centering, args.scale)
