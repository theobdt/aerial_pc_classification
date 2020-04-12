import argparse
import numpy as np
import os
import sys
import yaml

from utils.ply import ply2dict, dict2ply
from features import descriptors, region_growing, ground_extraction

AVAILABLE_STEPS = [
    "descriptors",
    "region_growing",
    "ground_extraction",
    "ground_rasterization",
    "height_above_ground",
]

PATH_FEATURES = "data/features"
PATH_GROUND_ONLY = "data/ground_only"
PATH_GROUND_RASTERIZED = "data/ground_rasterized"


def compute_features(path_ply, steps_params):
    filename = os.path.split(path_ply)[-1]

    data = ply2dict(path_ply)
    coords = np.vstack((data["x"], data["y"], data["z"])).T

    grid_ground_3d = None

    for (step, params) in steps_params.items():
        if step == "descriptors":
            print("Computing local descriptors..")
            all_descriptors = descriptors.compute_descriptors(coords, **params)
            data.update(all_descriptors)

        if step == "region_growing":
            print("\nComputing regions..")
            normals = np.vstack((data["nx"], data["ny"], data["nz"])).T
            params_copy = params.copy()

            descriptor_selected = params_copy.pop("descriptor")
            print(
                "* descriptor selected : "
                f"{'min' if params['minimize'] else 'max'} "
                f"{descriptor_selected}"
            )
            print(f"* thresholds : {params['thresholds']}")
            print(f"* radius : {params['radius']}")
            try:
                descriptor_vals = data[descriptor_selected]
                region_labels = region_growing.multi_region_growing(
                    coords, normals, descriptor_vals, **params_copy
                )

                data["regions"] = region_labels
            except KeyError:
                print(
                    f"Descriptor '{descriptor_selected}' has not been computed"
                    ", run 'python3 compute_features.py --descriptors "
                    f"{descriptor_selected}'"
                )
                sys.exit(-1)

        if step == "ground_extraction":
            print("\nExtracting ground from regions..")
            region_labels = data["regions"]
            ground_mask = ground_extraction.stitch_regions(
                coords, region_labels, **params
            )

            ground_only = {
                field: data[field][ground_mask] for field in list(data.keys())
            }

            data["ground"] = ground_mask.astype(np.uint8)

            os.makedirs(PATH_GROUND_ONLY, exist_ok=True)
            path_ground = os.path.join(PATH_GROUND_ONLY, filename)
            if dict2ply(ground_only, path_ground):
                print(f"* PLY ground file successfully saved to {path_ground}")

        if step == "ground_rasterization":
            print("\nComputing ground rasterization..")
            ground_mask = data["ground"].astype(bool)
            grid_ground_3d = ground_extraction.rasterize_ground(
                coords, ground_mask, **params
            )

            ground_rasterized = {
                "x": grid_ground_3d[:, 0],
                "y": grid_ground_3d[:, 1],
                "z": grid_ground_3d[:, 2],
                "ground_altitude": grid_ground_3d[:, 2],
            }

            path_rasterized = os.path.join(PATH_GROUND_RASTERIZED, filename)
            if dict2ply(ground_rasterized, path_rasterized):
                print(
                    "* PLY ground rasterized file successfully saved to "
                    f"{path_rasterized}"
                )

        if step == "height_above_ground":
            print("\nComputing height above ground..")
            if grid_ground_3d is None:
                path_rasterized = os.path.join(
                    PATH_GROUND_RASTERIZED, filename
                )
                print(f"* Loading rasterized ground : {path_rasterized}")
                ground_rasterized = ply2dict(path_rasterized)
                grid_ground_3d = np.vstack(
                    (
                        ground_rasterized["x"],
                        ground_rasterized["y"],
                        ground_rasterized["z"],
                    )
                ).T

            ground_mask = data["ground"].astype(bool)
            heights = ground_extraction.height_above_ground(
                coords, ground_mask, grid_ground_3d
            )
            data["height_above_ground"] = heights
            print("DONE")

    # saving data
    path_output = os.path.join(PATH_FEATURES, filename)
    if dict2ply(data, path_output):
        print(f"\nPLY features file successfully saved to {path_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to extract features form PLY file"
    )
    parser.add_argument(
        "--prefix_path", type=str, default="", help="Path prefix",
    )
    parser.add_argument(
        "--files",
        "-f",
        type=str,
        nargs="+",
        required=True,
        help="Path to point cloud file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cfg/config_features_extraction.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--full_pipeline",
        action="store_true",
        help="Run all steps from the beginning",
    )
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        help=f"List of steps to run, available steps are: {AVAILABLE_STEPS}",
    )
    parser.add_argument(
        "--from_step",
        type=str,
        help="Will run the features extraction pipeline from this step",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(PATH_FEATURES, exist_ok=True)

    if args.full_pipeline:
        steps = AVAILABLE_STEPS
    elif args.from_step:
        assert args.from_step in AVAILABLE_STEPS
        steps = AVAILABLE_STEPS[AVAILABLE_STEPS.index(args.from_step) :]
    elif args.steps:
        assert np.all([s in AVAILABLE_STEPS for s in args.steps])
        steps = args.steps
    else:
        raise ValueError("No input step")
    steps_params = {step: config[step] for step in steps}

    # update path with prefix
    PATH_FEATURES = os.path.join(args.prefix_path, PATH_FEATURES)
    PATH_GROUND_ONLY = os.path.join(args.prefix_path, PATH_GROUND_ONLY)
    PATH_GROUND_RASTERIZED = os.path.join(
        args.prefix_path, PATH_GROUND_RASTERIZED
    )

    for path_file in args.files:
        path_ply = os.path.join(args.prefix_path, path_file)
        print(f"\nComputing features of file {path_ply}")

        data = ply2dict(path_ply)
        compute_features(path_ply, steps_params)
