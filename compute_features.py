import argparse
import numpy as np
import os
import sys

from utils.ply import ply2dict, dict2ply
from features import descriptors, region_growing, ground_extraction

PATH_FEATURES = "data/features"
PATH_GROUND_ONLY = "data/ground_only"
PATH_GROUND_RASTERIZED = "data/ground_rasterized"

DESCRIPTORS = [
    "normals",
    "verticality",
    "linearity",
    "planarity",
    "sphericity",
    "curvature",
]

METHODS_RASTERIZATION = ["closest_neighbor", "delaunay"]


def compute_features(path_ply, steps):
    filename = os.path.split(path_ply)[-1]

    data = ply2dict(path_ply)
    coords = np.vstack((data["x"], data["y"], data["z"])).T

    grid_ground_3d = None

    for (step, params) in steps.items():
        if step == "descriptors":
            print("Computing local descriptors..")
            all_descriptors = descriptors.compute_descriptors(coords, **params)
            data.update(all_descriptors)

        if step == "regions":
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

            path_ground = os.path.join(PATH_GROUND_ONLY, filename)
            if dict2ply(ground_only, path_ground):
                print(f"* PLY ground file successfully saved to {path_ground}")

        if step == "rasterize_ground":
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
    parser = argparse.ArgumentParser(description="Compute features")
    parser.add_argument(
        "--files", "-f", type=str, nargs="+", help="Path to point cloud file"
    )
    parser.add_argument(
        "--full_pipeline", action="store_true", help="Run all steps"
    )
    # DESCRIPTORS
    parser.add_argument(
        "--compute_descriptors",
        "-cd",
        action="store_true",
        help="Compute local descriptors",
    )
    parser.add_argument(
        "--descriptors",
        "-d",
        type=str,
        nargs="+",
        default=["all"],
        help="Descriptors to keep",
    )
    parser.add_argument(
        "--radius_descriptors",
        "-rd",
        type=float,
        default=2,
        help="Radius used to compute descriptors",
    )
    parser.add_argument(
        "--preferred_orientation",
        type=str,
        default="+z",
        help="Preferred normal orientation",
    )
    parser.add_argument(
        "--epsilon_descriptors",
        type=float,
        default=1e-2,
        help="Epsilon added to denominator for some descriptors",
    )
    # REGION GROWING
    parser.add_argument(
        "--region_growing", "-rg", action="store_true", help="Compute regions",
    )
    parser.add_argument(
        "--radius_region",
        "-rr",
        type=float,
        default=1,
        help="Radius used for region growing",
    )
    parser.add_argument(
        "--n_regions",
        "-nr",
        type=int,
        default=50,
        help="Number of regions to grow",
    )
    parser.add_argument(
        "--criterion_region",
        type=str,
        nargs="+",
        default=["max", "planarity"],
        help=(
            "Criterion and descriptor used to compute region. "
            "Must be in the format '[min/max] [descriptor]'"
        ),
    )
    parser.add_argument(
        "--thresh_height",
        type=float,
        default=0.1,
        help=(
            "thresh_height : max difference of height for a new point to be "
            "part of the region\n"
        ),
    )
    parser.add_argument(
        "--thresh_angle",
        type=float,
        default=0.1,
        help=(
            "Max difference of angle (radians) between normals "
            "for a new point to be part of the region\n"
        ),
    )
    parser.add_argument(
        "--thresh_descriptor",
        type=float,
        default=0.1,
        help=(
            "Limit [descriptor] threshold above/below "
            "which a point is discarded"
        ),
    )
    # GROUND EXTRACTION
    parser.add_argument(
        "--ground_extraction",
        "-ge",
        action="store_true",
        help="Extract ground by stitching regions together",
    )
    parser.add_argument(
        "--slope_intra",
        "-sia",
        type=float,
        default=0.1,
        help=(
            "Maximum internal relative slope for a region to be considered"
            "as ground : (diff_height / span)"
        ),
    )
    parser.add_argument(
        "--slope_inter",
        "-sir",
        type=float,
        default=0.2,
        help=(
            "Maximum relative slope between the current ground and the input "
            "region to be considered as ground : "
            "(avg diff_height / avg distance)"
        ),
    )
    parser.add_argument(
        "--percentile_closest",
        "-pc",
        type=float,
        default=0.1,
        help=(
            "Percentile of the closest points of the input region from the "
            "current ground region inspected to get the inter slope "
            "(avg diff_height / avg distance)"
        ),
    )
    # GROUND RASTERIZATION
    parser.add_argument(
        "--rasterize_ground",
        "-rag",
        action="store_true",
        help="Compute ground rasterization",
    )
    parser.add_argument(
        "--step_rasterize",
        type=float,
        default=0.5,
        help="Step size used for rasterization",
    )
    parser.add_argument(
        "--method_rasterize",
        type=str,
        default="delaunay",
        help="Method used for ground rasterization",
    )
    # HEIGHT ABOVE GROUND
    parser.add_argument(
        "--height_above_ground",
        "-hag",
        action="store_true",
        help="Compute height above ground",
    )
    args = parser.parse_args()

    os.makedirs(PATH_FEATURES, exist_ok=True)

    steps = {}
    if args.compute_descriptors or args.full_pipeline:
        selected_descriptors = args.descriptors
        if selected_descriptors == ["all"]:
            selected_descriptors = DESCRIPTORS
        assert np.all([d in DESCRIPTORS for d in selected_descriptors])

        steps["descriptors"] = {
            "descriptors": selected_descriptors,
            "radius": args.radius_descriptors,
            "preferred_orientation": args.preferred_orientation,
            "epsilon": args.epsilon_descriptors,
        }

    if args.region_growing or args.full_pipeline:
        assert args.criterion_region[0] in ["min", "max"]
        assert args.criterion_region[1] in DESCRIPTORS[1:]
        steps["regions"] = {
            "radius": args.radius_region,
            "n_regions": args.n_regions,
            "minimize": args.criterion_region[0] == "min",
            "descriptor": args.criterion_region[1],
            "thresholds": {
                "height": args.thresh_height,
                "angle": args.thresh_angle,
                "descriptor": args.thresh_descriptor,
            },
        }
    if args.ground_extraction or args.full_pipeline:
        steps["ground_extraction"] = {
            "slope_intra_max": args.slope_intra,
            "slope_inter_max": args.slope_inter,
            "percentile_closest": args.percentile_closest,
        }
        os.makedirs(PATH_GROUND_ONLY, exist_ok=True)

    if args.rasterize_ground or args.full_pipeline:
        assert args.method_rasterize in METHODS_RASTERIZATION
        steps["rasterize_ground"] = {
            "step_size": args.step_rasterize,
            "method": args.method_rasterize,
        }
        os.makedirs(PATH_GROUND_RASTERIZED, exist_ok=True)

    if args.height_above_ground or args.full_pipeline:
        steps["height_above_ground"] = {}

    if len(steps.keys()) == 0:
        print("ERROR : No steps to compute")
        sys.exit(-1)

    for path_file in args.files:
        print(f"\nComputing features of file {path_file}")

        data = ply2dict(path_file)
        compute_features(path_file, steps)
