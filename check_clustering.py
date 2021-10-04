import numpy as np
import json
import pickle

def main():

    # clusters = np.load('c.npy')
#    clusters = np.load('c_granular.npy')
    path = 'out_vader_granular/8.64.0.01.12.2/c.npy'
    path = 'c_granular.2.npy'
    # path = 'c_granular.3.npy'
    # path = 'c_granular.4.npy'
    # path = 'c_granular.5.npy'
    clusters = np.load(path)

    n_clusters = int(max(clusters) + 1)

    d_cluster_to_eids = {}
    for i in range(len(clusters)):
        d_cluster_to_eids.setdefault(clusters[i], set()).add(i)

    for k, v in sorted(d_cluster_to_eids.items()):
        print(k, len(v))

    with open('out/dict.pkl', 'rb') as f:
        # d = json.load(f)
        d = pickle.load(f)

    eids = set()
    with open('out/eids.txt') as f:
        for line in f:
            eids.add(line.rstrip())
    eids = list(sorted(eids))

    icd10_groups = set()
    with open('out/icd10_groups.txt') as f:
        for line in f:
            icd10_groups.add(line.rstrip())
    icd10_groups = list(sorted(icd10_groups))

    icd10_group_counts = {}
    with open("out/icd10_group_counts.txt") as f:
        for line in f:
            k, v = line.rstrip().split()
            icd10_group_counts[k] = int(v)

    # icd10_codes = set()
    # with open('out/icd10_codes.txt') as f:
    #     for line in f:
    #         icd10_codes.add(line.rstrip())
    # icd10_codes = list(sorted(icd10_codes))

    sorted_eids = list(map(int, sorted(eids)))
    n = [len(d_cluster_to_eids.get(i, [])) for i in range(n_clusters)]
    for i_icd10_group, icd10_group in enumerate(icd10_groups):
        score = [0] * n_clusters
        for cluster in range(n_clusters):
            for i_eid in d_cluster_to_eids.get(cluster, []):
                eid = sorted_eids[i_eid]
                if d[eid].get(icd10_group) is None:
                    continue
                score[cluster] += 1

        # # frequency
        score = [score[i] / n[i] if n[i] > 0 else 0 for i in range(n_clusters)]
        # normalized frequency
        # score = [score[i] / (n[i] * icd10_group_counts[icd10_group]) if n[i] > 0 else 0 for i in range(n_clusters)]

        # if max(score) < .5 and i_icd10_group not in ('F20', 'F25', 'F31', 'F33'): continue
        if abs(max(score) - min(score)) < .1 and icd10_group not in ('F20', 'F25', 'F31', 'F33', 'I20-I25'): continue
        # if max(score) < .000005 and i_icd10_group not in ('F20', 'F25', 'F31', 'F33'): continue
        print(icd10_group, '\t'.join([str(round(_, 2)) for _ in score]), sep='\t')

    return

if __name__ == '__main__':
    main()
