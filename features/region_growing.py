from sklearn.neighbors import KDTree
import numpy as np


def geometry_criterion(p1, p2, n1, n2, thresh_height, thresh_angle):

    criterion_height = (p2 - p1) @ n1 < thresh_height

    # dot product
    criterion_angle = np.abs(n2 @ n1) > np.cos(thresh_angle)
    # abs to avoid issues with normals in different directions

    return np.logical_and(criterion_height, criterion_angle)


def descriptor_criterion(descriptor, minimize, thresh_descriptor):
    if minimize:
        return descriptor < thresh_descriptor
    return descriptor > thresh_descriptor


def region_growing(
    coords, normals, radius, descriptor_vals, minimize, thresholds
):

    N = len(coords)
    tree = KDTree(coords)
    region_mask = np.zeros(N, dtype=bool)

    Q_idx = [
        np.argmin(descriptor_vals) if minimize else np.argmax(descriptor_vals)
    ]
    seen_idx = np.zeros(N, dtype=bool)
    i = 0

    while len(Q_idx) > 0:
        print(f"  * N processed neighborhoods : {i}", end="\r")
        i += 1
        seed_idx = Q_idx.pop(0)

        seed_coords = coords[seed_idx]
        seed_normal = normals[seed_idx]

        neighbors_idx = tree.query_radius(seed_coords.reshape(1, -1), radius)
        neighbors_idx = neighbors_idx[0]

        # discard neighbors that have already been processed
        neighbors_idx = neighbors_idx[~seen_idx[neighbors_idx]]

        neighbors_points = coords[neighbors_idx]
        neighbors_normals = normals[neighbors_idx]

        # select neighbors
        geometry_mask = geometry_criterion(
            seed_coords,
            neighbors_points,
            seed_normal,
            neighbors_normals,
            thresholds["height"],
            thresholds["angle"],
        )
        selected_idx = neighbors_idx[geometry_mask]

        # add them to the region
        region_mask[selected_idx] = 1

        # add some of the to the queue
        selected_planarities = descriptor_vals[selected_idx]
        descriptor_mask = descriptor_criterion(
            selected_planarities, minimize, thresholds["descriptor"]
        )

        queue_idx = selected_idx[descriptor_mask]

        # add processed indexes to the seen array
        seen_idx[neighbors_idx] = 1

        Q_idx += list(queue_idx)
        if i > 500000:
            print("Region growing stopped early : n_points > 500.000")
            break

    print(f"  * Total number of points in region : {np.sum(region_mask)}")

    return region_mask


def multi_region_growing(
    coords, normals, descriptor_vals, radius, n_regions, minimize, thresholds
):

    N = len(coords)
    is_region = np.zeros(N, dtype=bool)
    region_labels = -np.ones(N, dtype=np.int32)
    indexes = np.arange(N)

    for i in range(n_regions):
        print(f"* Region {i + 1}/{n_regions}")
        label_i = i + 1
        region_mask = region_growing(
            coords, normals, radius, descriptor_vals, minimize, thresholds
        )

        idx_region = indexes[region_mask]
        region_labels[idx_region] = label_i
        is_region[idx_region] = 1

        coords = coords[~region_mask]
        normals = normals[~region_mask]
        descriptor_vals = descriptor_vals[~region_mask]

        indexes = indexes[~region_mask]
    n_regions_grown = len(np.unique(region_labels) - 1)
    print(f"* Number of valid regions grown : {n_regions_grown}")

    return region_labels
