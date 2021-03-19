import math

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random


def show_pref_matrix(pref_m, label_k):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    matr = ax.imshow(pref_m, cmap='Blues', interpolation='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(matr, cax=cax)
    fig.suptitle(label_k)
    fig.tight_layout()
    plt.show()


def compute_errors(clusters_mask, clusters_mask_gt):
    # Recall: misclassification error = num of misclassified points / num of points

    num_of_pts = len(clusters_mask)
    num_of_clusters = 1 + max(clusters_mask)
    num_of_clusters_gt = 1 + clusters_mask_gt[-1]

    # region Initialize empty clusters
    clusters = [set() for i in range(num_of_clusters)]
    clusters_gt = [set() for i in range(num_of_clusters_gt)]
    # endregion

    # region Fill the empty clusters using the clusters masks
    for i in range(num_of_pts):
        clusters[clusters_mask[i]].add(i)
        clusters_gt[clusters_mask_gt[i]].add(i)
    # endregion

    # region Sort the clusters in decreasing order
    clusters.sort(key=len, reverse=True)
    clusters_gt.sort(key=len, reverse=True)
    # endregion

    # region Actually compute ME as the set-difference between two clusters
    errors = 0
    for cl in clusters:
        matched_cl_gt_idx = None
        min_len = 99999
        # min_diff_cl = set()
        for cl_gt_idx, cl_gt in enumerate(clusters_gt):
            diff_cl = cl - cl_gt
            if len(diff_cl) < min_len:
                min_len = len(cl - cl_gt)
                matched_cl_gt_idx = cl_gt_idx
                # min_diff_cl = diff_cl

        if matched_cl_gt_idx is not None:  # once the matching gt cluster is found, we remove it
            del clusters_gt[matched_cl_gt_idx]
        else:  # matched_cl_gt_idx is None when there are no missing clusters
            min_len = len(cl)
        # print(min_len)
        # print(min_diff_cl)
        errors += min_len
    # endregion

    return errors, num_of_pts


def plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k):
    img_src = cv2.cvtColor(cv2.imread(img_i, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img_dst = cv2.cvtColor(cv2.imread(img_j, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    num_of_clusters = 1 + max(clusters_mask)  # cluster labels goes from 0 to N-1

    # region Colors
    # 10 colors
    color = [(0, 0, 0),
             (0, 1, 1),     # light blue
             (1, 0, 1),     # fuchsia
             (.5, 1, 0),    # light green
             (1, .7, 0),    # orange
             (0, 0, 1),     # blue
             (.2, .4, .1),  # green
             (.5, 0, 1),    # violet
             (1, 0, 0),     # red
             (1, 1, 0),     # yellow
             ]

    if num_of_clusters > 10:
        additional_colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
             for i in range(num_of_clusters)]
        color += additional_colors
    # endregion

    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax[0].imshow(img_src, alpha=0.75)
    ax[1].imshow(img_dst, alpha=0.75)

    for cluster_idx in range(num_of_clusters):
        current_cluster = np.where(clusters_mask == cluster_idx)[0]  # tuple of dimension 1

        for i in current_cluster:
            src_p = src_pts[i]
            dst_p = dst_pts[i]

            src_p_x = math.floor(src_p[0])
            src_p_y = math.floor(src_p[1])
            ax[0].plot(src_p_x, src_p_y, 'o', color=color[cluster_idx], markersize=3)

            dst_p_x = math.floor(dst_p[0])
            dst_p_y = math.floor(dst_p[1])
            ax[1].plot(dst_p_x, dst_p_y, 'o', color=color[cluster_idx], markersize=3)

    fig.suptitle(label_k)
    ax[0].axis('off')
    ax[1].axis('off')
    fig.tight_layout()
    plt.show()


def plotMatches(img_i, img_j, kp_src, kp_dst, good_matches):
    img_src = cv2.cvtColor(cv2.imread(img_i, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img_dst = cv2.cvtColor(cv2.imread(img_j, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=0)
    matches_image = cv2.drawMatches(img_dst, kp_dst, img_src, kp_src, good_matches, None, **draw_params)

    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ax.imshow(matches_image)
    ax.axis('off')
    fig.tight_layout()
    plt.show()


# def show_ransac(img_i, img_j, src_pts, dst_pts, preference_matrix):
#     # region Select the column with the largest consensus
#     h_best_idx = np.argmax(np.sum(preference_matrix, axis=0))
#     h_best = preference_matrix[:, h_best_idx]
#
#     kp_src = [cv2.KeyPoint(x=p[0], y=p[1], _size=1.0, _angle=1.0, _response=1.0, _octave=1, _class_id=-1)
#               for p in src_pts]
#     kp_dst = [cv2.KeyPoint(x=p[0], y=p[1], _size=1.0, _angle=1.0, _response=1.0, _octave=1, _class_id=-1)
#               for p in dst_pts]
#
#     img_src = cv2.cvtColor(cv2.imread(img_i, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#     img_dst = cv2.cvtColor(cv2.imread(img_j, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#
#     # region Plot only the inliers wrt the best homography (i.e., plot ransac solution)
#     kp_src_cl = []
#     kp_dst_cl = []
#     for i in range(len(src_pts)):
#         # if h_best[i] == 1:
#         if h_best[i] > 0.8:
#             kp_src_cl.append(kp_src[i])
#             kp_dst_cl.append(kp_dst[i])
#
#     kp_in_img_src = cv2.drawKeypoints(img_src, kp_src_cl, None)
#     kp_in_img_dst = cv2.drawKeypoints(img_dst, kp_dst_cl, None)
#
#     plt.imshow(kp_in_img_src)
#     plt.show()
#     plt.imshow(kp_in_img_dst)
#     plt.show()
#     # endregion


def get_cluster_mask(clusters, num_of_pts, outlier_threshold):
    cluster_mask = [i for i in range(num_of_pts)]
    cl_idx = 1
    for cl in clusters:
        if len(cl) < outlier_threshold:
            for i in cl:
                cluster_mask[i] = 0
        else:
            for i in cl:
                cluster_mask[i] = cl_idx
            cl_idx += 1
    return np.array(cluster_mask)