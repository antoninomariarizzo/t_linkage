import numpy as np


# Symmetric Transfer Error
def compute_residuals(H, src_pts, dst_pts):

    H_inv = np.linalg.inv(H)

    forward_projection = compute_projections(H, src_pts)
    backward_projection = compute_projections(H_inv, dst_pts)

    e1 = np.sqrt(np.sum(np.square(np.subtract(dst_pts, forward_projection)), axis=1))
    e2 = np.sqrt(np.sum(np.square(np.subtract(src_pts, backward_projection)), axis=1))
    e = e1 + e2

    return e
    # return e1  # use this line to return the traditional geometric distance


def compute_projections(H, pts):

    h11 = H[0][0]
    h12 = H[0][1]
    h13 = H[0][2]
    h21 = H[1][0]
    h22 = H[1][1]
    h23 = H[1][2]
    h31 = H[2][0]
    h32 = H[2][1]
    h33 = H[2][2]

    # The keypoints in output from SIFT have a weird additional dimension, just remove it
    if pts.ndim == 3:
        pts = np.reshape(pts, (pts.shape[0], pts.shape[2]))

    pts_projected = []
    for p in pts:
        p_x, p_y = p

        p_x_projected = (h11 * p_x + h12 * p_y + h13) / (h31 * p_x + h32 * p_y + h33)
        p_y_projected = (h21 * p_x + h22 * p_y + h23) / (h31 * p_x + h32 * p_y + h33)

        pts_projected.append([p_x_projected, p_y_projected])

    return np.array(pts_projected)

