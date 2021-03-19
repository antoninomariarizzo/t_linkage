import itertools
import numpy as np


def tanimoto_distance(i, j):
    pq = np.inner(i, j)

    if pq == 0:
        return 1

    p_square = np.inner(i, i)
    q_square = np.inner(j, j)

    t_distance = 1 - pq / (p_square + q_square - pq)
    return t_distance


def measure(x):
    return x[-1]


def clustering(pref_m):
    num_of_pts = pref_m.shape[0]
    pts = range(num_of_pts)
    clusters = [[i] for i in pts]
    new_idx = pref_m.shape[0]

    pos = {i: i for i in pts}

    x0 = list(itertools.combinations(range(num_of_pts), 2))
    x0 = [(cl_i, cl_j, tanimoto_distance(pref_m[cl_i], pref_m[cl_j])) for cl_i, cl_j in x0]

    while pref_m.shape[0] > 1:
        x0.sort(key=measure)
        cl_0, cl_1, min_distance = x0[0]
        if min_distance >= 1:
            break

        new_pf = np.minimum(pref_m[pos[cl_0]], pref_m[pos[cl_1]])  # element-wise min
        new_cluster = clusters[pos[cl_0]] + clusters[pos[cl_1]]

        pref_m = np.delete(pref_m, (pos[cl_0], pos[cl_1]), axis=0)
        pref_m = np.vstack((pref_m, new_pf))
        clusters = [c for idx_c, c in enumerate(clusters) if idx_c not in (pos[cl_0], pos[cl_1])]  # delete C_i and C_j
        clusters = clusters + [new_cluster]
        new_cluster.sort()

        pos0 = pos[cl_0]
        pos1 = pos[cl_1]
        del pos[cl_0]
        del pos[cl_1]

        for k in pos:
            if pos[k] >= pos0:
                pos[k] -= 1
            if pos[k] >= pos1:
                pos[k] -= 1

        pos[new_idx] = pref_m.shape[0] - 1

        pts = [p for p in pts if p not in (cl_0, cl_1)]
        x0 = [(cl_i, cl_j, d) for cl_i, cl_j, d in x0
              if cl_i not in (cl_0, cl_1) and cl_j not in (cl_0, cl_1)]

        new_comb = [(p, new_idx) for p in pts]
        pts.append(new_idx)
        new_idx += 1
        x1 = [(cl_i, cl_j, tanimoto_distance(pref_m[pos[cl_i]], pref_m[pos[cl_j]])) for cl_i, cl_j in new_comb]
        x0 += x1

        print("[CLUSTERING] New cluster: " + str(new_cluster)
              + "\t-\tTanimoto distance: " + str(min_distance))

    return clusters, pref_m

