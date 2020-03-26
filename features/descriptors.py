import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm


ORIENTATIONS = ["+x", "-x", "+y", "-y", "+z", "-z"]


def local_PCA(points):

    eigenvalues = None
    eigenvectors = None

    n = points.shape[0]
    centroids = np.mean(points, axis=0)
    centered = points - centroids

    cov = centered.T @ centered / n

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues.astype(np.float32), eigenvectors.astype(np.float32)


def neighborhood_PCA(query_points, cloud_points, radius):

    # This function needs to compute PCA on the neighborhoods of all
    # query_points in cloud_points
    tree = KDTree(cloud_points)

    print("* Querying radius..", end=" ", flush=True)
    idx_lists = tree.query_radius(query_points, radius)
    print("DONE")

    all_eigenvalues = np.zeros((query_points.shape[0], 3), dtype=np.float32)
    all_eigenvectors = np.zeros(
        (query_points.shape[0], 3, 3), dtype=np.float32
    )

    for i, idx_list in enumerate(
        tqdm(idx_lists, desc="* Processing neighborhoods")
    ):
        if len(idx_list) > 0:
            points = cloud_points[idx_list]
            eigenvalues, eigenvectors = local_PCA(points)
            all_eigenvalues[i, :] = eigenvalues
            all_eigenvectors[i, :, :] = eigenvectors

    return all_eigenvalues, all_eigenvectors


def orient_normals(normals, preferred_orientation="+z"):
    index = ORIENTATIONS.index(preferred_orientation)
    sign = 1 if index % 2 == 0 else -1
    direction = index // 2
    normals[:, direction] = sign * np.abs(normals[:, direction])

    return normals


def compute_descriptors(
    coords, radius, descriptors, preferred_orientation, epsilon
):

    if len(descriptors) == 0:
        return

    print(f"* descriptors: {descriptors}")
    print(f"* radius: {radius}")
    print(f"* epsilon: {epsilon}")
    print(f"* preferred normals orientation: {preferred_orientation}")
    # Compute the features for all points of the cloud
    eigenvalues, eigenvectors = neighborhood_PCA(coords, coords, radius)

    normals = orient_normals(eigenvectors[:, :, 0], preferred_orientation)

    normals_z = normals[:, 2]

    # lambda_1 >= lambda_2 >= lambda_3
    lambda_1 = eigenvalues[:, 2]
    lambda_2 = eigenvalues[:, 1]
    lambda_3 = eigenvalues[:, 0]

    # epsilon = 1e-2 * np.ones(len(normals_z))
    epsilon_array = epsilon * np.ones(len(normals_z), dtype=np.float32)
    all_descriptors = {}

    if "normals" in descriptors:
        all_descriptors["nx"] = normals[:, 0]
        all_descriptors["ny"] = normals[:, 1]
        all_descriptors["nz"] = normals[:, 2]

    if "verticality" in descriptors:
        verticality = 2 * np.arcsin(normals_z) / np.pi
        all_descriptors["verticality"] = verticality

    if "linearity" in descriptors:
        linearity = 1 - (lambda_2 / (lambda_1 + epsilon_array))
        all_descriptors["linearity"] = linearity

    if "planarity" in descriptors:
        planarity = (lambda_2 - lambda_3) / (lambda_1 + epsilon_array)
        all_descriptors["planarity"] = planarity

    if "sphericity" in descriptors:
        sphericity = lambda_3 / (lambda_1 + epsilon_array)
        all_descriptors["sphericity"] = sphericity

    if "curvature" in descriptors:
        curvature = lambda_3 / (lambda_1 + lambda_2 + lambda_3 + epsilon_array)
        all_descriptors["curvature"] = curvature

    return all_descriptors
