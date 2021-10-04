import pandas as pd
import numpy as np
import collections
import re


def main():

    path1 = 'gp_readv2toicd10.txt'
    path2 = 'gp_readv3toicd10.txt'
    path3 = 'hes_three_twodigit_withdup.txt'

    d_id2range = dict()
    d_icd_groups = dict()
    with open('icd10.level2.ranges.txt') as f:
        for line in f:
            d_icd_groups[line[:7]] = line[8:]
            print(line)
            s1 = line[1:3]
            s2 = line[5:7]
            for i in range(int(s1, 16), int(s2, 16) + 1):
                d_id2range[line[0] + format(i, 'X').zfill(2)] = line[:7]

    multimorbidity = dict()
    with open('multimorbidity_icd10.txt') as f:
        for line in f:
            if '-' in line or '–' in line:
                line = line.replace('–', '-')
                _ = line.rstrip().split('-')
                if len(_[0]) == 3 and len(_[1]) == 3:
                    for i in range(int(_[0][1:3]), int(_[1][1:3]) + 1):
                        icd = line[:1] + str(i).zfill(2)
                        multimorbidity.setdefault(icd, set()).add('')
                else:
                    assert _[0][:3] == _[1][:3], line
                    assert len(_[0]) == 5, _[0]
                    assert len(_[1]) == 5, _[1]
                    for i in range(int(_[0][-1]), int(_[1][-1]) + 1):
                        multimorbidity.setdefault(line[:3], set()).add(str(i))
            else:
                assert line[3] in '.\n', line
                multimorbidity.setdefault(line[:3], set()).add(line[3:-1])
                # if len(line) > 4:
                #     print(line)
                #     print(multimorbidity)
                #     stop77

    smi = set(('F20', 'F25', 'F33', 'F31',))

    d = dict()
    eids = set()
    cnt_range = 0
    cnt_non_range = 0
    for path, sep, col in (
        (path1, '|', 'icd10_code'),
        (path2, '|', 'icd10_code'),
        (path3, '\t', 'code'),
        ):
        df = pd.read_csv(path, sep=sep)
        print(path)
        for t in df[['eid', col]].dropna().itertuples():
            eid = t.eid
            eids.add(eid)
            # if getattr(t, col) == np.nan:
            #     print(path, eid, 'nan')
            #     continue
            # if type(getattr(t, col)) == float:
            #     print(path, eid, 'nan')
            #     continue
            for icd in re.split('[, +]', getattr(t, col)):
                if '-' in icd:
                    cnt_range += 1
                    continue
                cnt_non_range += 1
                if len(icd) > 4:
                    assert len(icd) == 5, icd
                    # assert icd[-1] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', icd
                    icd = icd[:4]
                assert len(icd) in (3, 4), (eid, icd)
                if not icd[:3] in smi and (
                    not icd[3:] in multimorbidity.get(icd[:3], set())
                    and not '' in multimorbidity.get(icd[:3], set())
                    ):
                    continue
                d.setdefault(eid, set()).add(icd)

    cnt_group = collections.Counter()
    cnt_icd = collections.Counter()
    for eid, icds in d.items():
        added = set()
        added_icd3 = set()
        for icd in icds:
            group = d_id2range.get(icd[:3], 'Unknown')
            if not group in added:
                cnt_group[group] += 1
            if not icd[:3] in added_icd3:
                cnt_icd[icd[:3]] += 1

    d_cnts = {icd: collections.Counter() for icd in smi}
    d_cnts_semi_normalized = {icd: collections.Counter() for icd in smi}
    d_cnts_normalized = {icd: collections.Counter() for icd in smi}
    cnt_smi_only = 0
    cnt_physical_only = 0
    cnt_smi_and_physical = 0
    for eid, icds in d.items():
        icds3 = set((icd[:3] for icd in icds))
        intersection = icds3 & smi
        if len(intersection) == 0:
            cnt_physical_only += 1
            continue
        if len(icds3 - smi) == 0:
            for icd in intersection:
                d_cnts[icd]['None'] += 1
            cnt_smi_only += 1
            continue
        for icd1 in intersection:
            added = set()
            for icd2 in icds:
                if icd1[:3] == icd2[:3]:
                    continue
                if icd2[:3] in smi:
                    continue
                group = d_id2range.get(icd2[:3], 'Unknown')
                if group in added:
                    continue
                d_cnts[icd1][group] += 1
                d_cnts_semi_normalized[icd1][group] += 1 / (cnt_icd[icd1])
                d_cnts_normalized[icd1][group] += 1 / (cnt_icd[icd1] * cnt_group[group])
                # if icd2[0] != 'T': continue
                # if cnt_icd[icd2[:3]] == 0: continues
                # d_cnts_normalized[icd1][icd2[:3]] += 1 / (cnt_icd[icd1] * cnt_icd[icd2[:3]])
                added.add(group)
            if len(added) > 0:
                d_cnts[icd1]['Physical'] += 1
            elif len(added) == 0:
                d_cnts[icd1]['None'] += 1
                # stop1
        cnt_smi_and_physical += 1

    for icd in sorted(smi):
        print(icd, d_cnts_normalized[icd].most_common(15))
    print('len', len(d.keys()))
    print('eids', len(eids))
    print('cnt_smi_only', cnt_smi_only)
    print('cnt_physical_only', cnt_physical_only)
    print('cnt_smi_and_physical', cnt_smi_and_physical)
    for icd in sorted(smi):
        print(icd, 'total', cnt_icd[icd])
        print(icd, 'only', d_cnts[icd]['None'])
        print(icd, '+ physical', d_cnts[icd]['Physical'])
    print('cnt_range', cnt_range)
    print('cnt_non_range', cnt_non_range)

    # df = pd.read_csv(path1, sep='\t')
    # print(df['code'])

    plot(d_cnts, 'plot_bar_non-normalized')
    plot(d_cnts_normalized, 'plot_bar_normalized')

    return


def plot(d_cnts, affix):

    smi = set(('F20', 'F25', 'F33', 'F31',))

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(16 / 1.5, 9 / 1.5))

    top10 = set()
    for icd in sorted(smi):
        top10 |= set((_ for _, cnt in d_cnts[icd].most_common(8)))
    top10 -= set(('Physical',))

    # create data
    print('top10', set(top10))
    x = np.arange(len(top10))
    width = 1 / (len(smi) + 1)
    for i, (icd1, color) in enumerate(zip(sorted(smi), ('#a6cee3', '#1f78b4', '#b2df8a', '#33a02c'))):
        y = [d_cnts[icd1][icd2] for icd2 in sorted(top10)]
        print(icd1, color, y)
        ax.bar(x + (i - len(smi)/2 + 0.5) * width, y, width, color=color)
      
    # plot data in grouped manner of bar type
    plt.xticks(x, sorted(top10), fontsize='small')
    # plt.xticks(x, (d_icd_groups[_] for _ in sorted(top10)), fontsize='xx-small')
    # plt.xlabel("Physical", fontsize='small')
    plt.ylabel("Count")
    plt.legend([{
        'F20': 'F20 Schizophrenia',
        'F25': 'F25 Schizoaffective disorders',
        'F31': 'F31 Bipolar affective disorder',
        'F33': 'F33 Recurrent depressive disorder',
        }[_] for _ in sorted(smi)])
    ax.tick_params(labelrotation=45)
    fig.savefig('{}.png'.format(affix))

    return


if __name__ == '__main__':
    main()
