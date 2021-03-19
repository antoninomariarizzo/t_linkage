import cv2
import numpy as np

from utils_sampling import localized_sampling

K = 6  # the overall number of sampling is computed as K * num_of_points
FM_MSS = 7   # cardinality of the MSS for the fundamental matrix case
INLIER_THRESHOLD = 3.0  # inlier threshold, needed by OpenCV RanSaC


def get_preference_matrix_fm(kp_src, kp_dst, good_matches, tau):

    # region Initialize variables
    num_of_matches = len(good_matches)
    num_of_samplings = int(K * num_of_matches)
    preference_matrix = []
    src_pts = np.float32([kp_src[m.trainIdx].pt for m in good_matches])
    dst_pts = np.float32([kp_dst[m.queryIdx].pt for m in good_matches])
    # endregion

    for m in range(num_of_samplings):
        # region Sample a MSS
        # Uniform sampling
        # g = np.random.Generator(np.random.PCG64())
        # mss_idx = np.array(g.choice(num_of_matches, FM_MSS, replace=False))
        # endregion

        # region Localized sampling
        prob = (1 / num_of_matches) * np.ones(num_of_matches)  # uniform probability (random sampling)
        mss_idx = localized_sampling(src_pts, dst_pts, FM_MSS, prob)
        # endregion

        # region Retrieve source and destination pts from the MSS
        good_matches_current = [good_matches[i] for i in mss_idx]
        src_pts_current = np.float32([kp_src[m.trainIdx].pt for m in good_matches_current]).reshape(-1, 1, 2)
        dst_pts_current = np.float32([kp_dst[m.queryIdx].pt for m in good_matches_current]).reshape(-1, 1, 2)
        # endregion

        # endregion

        # region Compute the fundamental matrix F and the mask of inliers from a single iteration of RANSAC
        F, inliers_mask = cv2.findFundamentalMat(src_pts_current, dst_pts_current, cv2.RANSAC,
                                                 ransacReprojThreshold=INLIER_THRESHOLD, confidence=0.999,
                                                 maxIters=1)
        # endregion

        if F is not None:
            # region Compute residuals
            r = []
            for src_p, dst_p in zip(src_pts, dst_pts):
                src_p = np.array([src_p[0], src_p[1], 1])
                dst_p = np.array([dst_p[0], dst_p[1], 1])
                x = cv2.sampsonDistance(src_p, dst_p, F)
                r.append(x)

            r = np.array(r)
            # endregion

            # region Compute m-th colum of the preference matrix
            preference_column = np.where(r < tau, np.exp(- r / np.array(tau)), 0)  # T-Linkage
            # preference_column = np.where(r < tau, 1, 0)  # J-Linkage
            # endregion

            preference_matrix.append(preference_column)

    # The preference matrix has the points as rows, we need to compute the transpose
    preference_matrix = np.array(preference_matrix).transpose()

    return preference_matrix
