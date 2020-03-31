import numpy as np
from sklearn.neighbors import KDTree
from scipy import spatial
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

    # start with the region with the largest span
    init_label = all_labels[np.argmax(spans)]
    ground_mask[region_labels == init_label] = 1

    # computing slopes inter
    coords_ground = coords[ground_mask]

    tree = KDTree(coords_ground)
    for i, label in enumerate(
        tqdm(all_labels, desc="* Stitching regions together")
    ):
        if label == init_label:
            continue

        if slopes_intra[i] > slope_intra_max:
            continue

        coords_label = coords[region_labels == label]
        dist, idx = tree.query(coords_label, k=1)
        idx = idx.ravel()
        dist = dist.ravel()

        mask_closest = dist < np.percentile(dist, 100 * percentile_closest)
        if np.sum(mask_closest) == 0:
            continue
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


def interpolate_altitude(coords_ground, coords_queries_xy, method="delaunay"):

    if method == "closest_neighbor":
        # create a KD tree on xy coordinates
        tree = KDTree(coords_ground[:, :2])

        # find closest neighbor on the ground
        _, idx_neighbor = tree.query(coords_queries_xy, k=1)
        idx_neighbor = idx_neighbor.flatten()

        z_ground = coords_ground[:, -1]
        z_queries = z_ground[idx_neighbor]
        grid_3d = np.hstack((coords_queries_xy, z_queries.reshape(-1, 1)))

    elif method == "delaunay":
        # create 2D triangulation of ground coordinates
        tri = spatial.Delaunay(coords_ground[:, :2])

        # Find simplex of each query point
        idx_simplices = tri.find_simplex(coords_queries_xy)
        convex_hull_mask = idx_simplices >= 0

        # keep only query points inside convex hull
        idx_simplices = idx_simplices[convex_hull_mask]
        coords_queries_hull = coords_queries_xy[convex_hull_mask]

        # compute weights
        trans = tri.transform[idx_simplices]
        inv_T = trans[:, :-1, :]
        r = trans[:, -1, :]
        diff = (coords_queries_hull - r)[:, :, np.newaxis]
        barycent = (inv_T @ diff).squeeze()
        weights = np.c_[barycent, 1 - barycent.sum(axis=1)]

        # interpolate z values of vertices
        z_vertices = coords_ground[:, -1][tri.simplices][idx_simplices]
        z_queries = np.sum(weights * z_vertices, axis=1)
        grid_3d = np.hstack((coords_queries_hull, z_queries.reshape(-1, 1)))

    else:
        raise ValueError(f"Method '{method}' not found")

    return grid_3d


def rasterize_ground(coords, ground_mask, step_size, method):
    print(f"* method : {method}")
    print(f"* step_size : {step_size}")
    mins = np.min(coords[:, :2], axis=0)
    maxs = np.max(coords[:, :2], axis=0)

    # Create a grid
    grid = np.mgrid[
        mins[0] : maxs[0] : step_size, mins[1] : maxs[1] : step_size
    ].T
    grid_points = grid.reshape(-1, 2)

    # Interpolate altitudes
    grid_3d = interpolate_altitude(coords[ground_mask], grid_points, method)

    return grid_3d


def height_above_ground(coords, ground_mask, grid_ground_3d):
    heights = np.zeros(len(coords), dtype=np.float32)

    coords_queries = coords[~ground_mask]

    tree = KDTree(grid_ground_3d[:, :2])

    # find closest neighbor on the rasterized ground
    _, idx_neighbor = tree.query(coords_queries[:, :2], k=1)
    idx_neighbor = idx_neighbor.flatten()

    # set heights
    z_ground = grid_ground_3d[:, -1]
    z_queries_ground = z_ground[idx_neighbor]
    heights[~ground_mask] = coords_queries[:, -1] - z_queries_ground

    return heights
