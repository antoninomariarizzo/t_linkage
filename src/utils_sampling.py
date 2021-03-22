import numpy as np


def get_localized_prob(pts, pt, ni):
    d_squared = np.sum(np.square(np.subtract(pts, pt)), axis=1)

    sigma = ni * np.median(np.sqrt(d_squared))
    sigma_squared = sigma ** 2

    prob = np.exp(- (1 / sigma_squared) * d_squared)

    return prob



def localized_sampling(src_pts, dst_pts, k, ni=1 / 3):
    num_of_pts = src_pts.shape[0]
    g = np.random.Generator(np.random.PCG64())

    mss0 = g.choice(num_of_pts, 1)

    prob_local_src = get_localized_prob(src_pts, src_pts[mss0], ni)
    prob_local_dst = get_localized_prob(dst_pts, dst_pts[mss0], ni)

    prob = np.max([prob_local_src, prob_local_dst], axis=0)
    prob[mss0] = 0
    prob = prob / np.sum(prob)

    mss1 = g.choice(num_of_pts, k-1, replace=False, p=prob)

    mss = mss0.tolist() + mss1.tolist()

    return np.array(mss)