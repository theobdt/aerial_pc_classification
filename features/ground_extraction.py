import numpy as np
from sklearn.neighbors import KDTree
from scipy import interpolate
from tqdm import tqdm


def stitch_regions(
    coords, region_labels, slope_intra_max, slope_inter_max, percentile_closest
):

    N = len(coords)
    ground_mask = np.zeros(N, dtype=bool)

    all_labels = np.unique(region_labels)
    all_labels = all_labels[all_labels > 0]

    spans = np.zeros(len(all_labels), dtype=np.float32)
    heights = np.zeros(len(all_labels), dtype=np.float32)

    # computing slopes intra
    for i, label in enumerate(
        tqdm(all_labels, desc="* Computing intra slopes")
    ):
        coords_label = coords[region_labels == label]
        deltas = np.max(coords_label, axis=0) - np.min(coords_label, axis=0)
        spans[i] = np.linalg.norm(deltas[:2])
        heights[i] = deltas[2]

    slopes_intra = heights / spans

    init_label = all_labels[np.argmax(spans)]
    # print(init_label)

    ground_mask[region_labels == init_label] = 1
    # print(np.sum(ground_mask))

    # computing slopes inter
    coords_ground = coords[ground_mask]
    # print(coords_ground.shape)
    tree = KDTree(coords_ground)
    for i, label in enumerate(
        tqdm(all_labels, desc="* Stitching regions together")
    ):
        if label == init_label:
            continue

        if slopes_intra[i] > slope_intra_max:
            continue

        # print("*****")
        coords_label = coords[region_labels == label]
        # print(len(coords_label))
        dist, idx = tree.query(coords_label, k=1)
        idx = idx.ravel()
        dist = dist.ravel()

        mask_closest = dist < np.percentile(dist, 100 * percentile_closest)
        if np.sum(mask_closest) == 0:
            continue
        # print(np.sum(mask_closest))
        coords_label_closest = coords_label[mask_closest]

        idx_ground_closest = idx[mask_closest]
        coords_ground_closest = coords_ground[idx_ground_closest]

        deltas = np.abs(coords_label_closest - coords_ground_closest)

        dist_xy = np.linalg.norm(deltas[:, :2], axis=1)
        dist_z = deltas[:, 2]
        slopes_inter = dist_z / dist_xy

        mean_slopes_inter = np.mean(slopes_inter)

        if mean_slopes_inter < slope_inter_max:
            ground_mask[region_labels == label] = 1
            coords_ground = coords[ground_mask]
            tree = KDTree(coords_ground)

    return ground_mask


# def interpolate_altitude(coords_ground, coords_queries_xy, k, step):
# tree = KDTree(coords_ground[:, :2])

# dist, idx = tree.query(coords_queries_xy, k=k)
# inv_dist = 1 / (1e-3 + dist)
# weights = inv_dist / np.sum(inv_dist, axis=1).reshape(-1, 1)

# ground_z = coords_ground[:, -1]
# altitude_queries = np.sum(ground_z[idx] * weights, axis=1)
# pairs_ground = (coords_ground[:, :2] // step).astype(np.int16)
# ranges = np.max(coords_ground, axis=0) - np.min(coords_ground, axis=0)
# idx_max = (ranges // step).astype(int)
# print(idx_max)

# print(step)
# altitudes = np.zeros(idx_max[:2])
# z_values = coords_ground[:, -1].astype(np.float32)
# unique_pairs = np.unique(pairs_ground, axis=0)
# heights = np.zeros(len(unique_pairs))
# for i, pair in enumerate(tqdm(unique_pairs)):
#     # print(pair)
#     mask = np.all(pairs_ground == pair, axis=1)
#     # selec = z_values[mask]
#     # print(pairs_ground[mask])
#     # print(selec.shape)
#     avg = np.mean(z_values[mask])
#     # altitudes[tuple(pair)] = avg
#     heights[i] = avg
#     # break
# print(altitudes.shape)
# tuples = tuple([unique_pairs[:, 0], unique_pairs[:, 1]])
# altitudes[tuples] = heights
# import matplotlib.pyplot as plt

# xx, yy = np.mgrid[:idx_max[0], :idx_max[1]]
# print(xx.shape)
# grid = interpolate.griddata(unique_pairs, heights, (xx, yy))


# f = interpolate.interp2d(unique_pairs[:, 0], unique_pairs[:, 1], heights)

# pairs_queries = np.unique(
#     (coords_queries_xy // step).astype(int), axis=0
# )
# # missing =
# altitude_queries = f(pairs_queries[:, 0], pairs_queries[:, 1])
# tuples_queries = tuple([altitude_queries[:, 0], altitude_queries[:, 1]])
# print(altitude_queries.shape)
# altitudes[tuples_queries] = altitude_queries
# plt.matshow(grid)
# plt.show()

# return altitude_queries


def interpolate_altitude(coords_ground, coords_queries_xy):

    # create a KD tree on xy coordinates
    tree = KDTree(coords_ground[:, :2])

    # find closest neighbor on the ground
    _, idx = tree.query(coords_queries_xy, k=1)
    idx = idx.flatten()

    ground_z = coords_ground[:, -1]
    altitude_queries = ground_z[idx]

    return altitude_queries


def height_above_ground(coords, ground_mask):
    heights = np.zeros(len(coords), dtype=np.float32)

    coords_ground = coords[ground_mask]
    coords_queries = coords[~ground_mask]

    altitude_queries = interpolate_altitude(
        coords_ground, coords_queries[:, :2]
    )
    heights[~ground_mask] = coords_queries[:, -1] - altitude_queries

    return heights


def rasterize_ground(coords, ground_mask, step):
    mins = np.min(coords[:, :2], axis=0)
    maxs = np.max(coords[:, :2], axis=0)

    # Create a grid
    grid = np.mgrid[mins[0] : maxs[0] : step, mins[1] : maxs[1] : step].T
    grid_points = grid.reshape(-1, 2)

    # Interpolate altitudes
    grid_altitudes = interpolate_altitude(coords[ground_mask], grid_points)

    grid_3d = np.hstack((grid_points, grid_altitudes.reshape(-1, 1)))

    return grid_3d
