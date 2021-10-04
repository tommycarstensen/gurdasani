import pandas as pd
import numpy as np
import collections
import re
import datetime
import numpy as np
import json
import pickle


def main():

    d_id2range = icd_id2range()

    path = 'death_24Jun2021.txt.gz'
    print('reading', path)
    d_deaths = pd.read_csv(path, sep='\t')[[
        'eid', 'date_of_death']].set_index('eid').to_dict()

    d_registrations = parse_registrations()

    multimorbidity = get_multimorbidity_codes()

    smi = set((
        'F20', # Schizophrenia
        'F25', # Schizoaffective disorders
        'F31', # Bipolar affective disorder
        'F33', # Major depressive disorder
        ))

    d = convert_to_array(
        multimorbidity, smi, d_id2range, d_deaths, d_registrations)

    return


def parse_registrations():

    path = 'gp_registrations_24Jun2021.txt.gz'

    # df_registrations = pd.read_csv(
    #     path, sep='\t',
    #     parse_dates=['reg_date', 'deduct_date'], dayfirst=True,
    #     )[[
    #     'eid', 'reg_date', 'deduct_date']].set_index('eid')

    d = {}
    for t in pd.read_csv(
        path, sep='\t',
        parse_dates=['reg_date', 'deduct_date'], dayfirst=True,
        ).itertuples():
        eid = t.eid
        reg_date = t.reg_date
        deduct_date = t.deduct_date
        if reg_date == pd.Timestamp("1902-02-02 00:00:00"):
            reg_date = pd.NaT
        if deduct_date == pd.Timestamp("1902-02-02 00:00:00"):
            deduct_date = pd.NaT
        if deduct_date == pd.Timestamp("2037-07-07 00:00:00"):
            deduct_date = pd.NaT
        if not(deduct_date is pd.NaT or reg_date is pd.NaT):
            if not deduct_date >= reg_date:
                print(eid, reg_date, deduct_date)
                continue
        if not eid in d.keys():
            d[eid] = [None, None]
        if reg_date is not pd.NaT:
            # print(eid, reg_date, deduct_date)
            if d[eid][0] is None:
                d[eid][0] = reg_date
            elif reg_date < d[eid][0]:
                d[eid][0] = reg_date
            # elif d[eid][0][1] >= reg_date:
            #     assert d[eid][0][0] <= reg_date, (d[eid], reg_date, deduct_date)
            #     # d[eid][0][0] = reg_date
            #     pass
            else:
                pass
                # print(d[eid])
                # print(reg_date)
                # print(deduct_date)
                # stop2a
        if deduct_date is not pd.NaT:
            if d[eid][1] is None:
                d[eid][1] = deduct_date
            elif deduct_date > d[eid][1]:
                # assert d[eid][0][1] <= deduct_date, (d[eid], reg_date, deduct_date)
                # print((d[eid][0][1] - reg_date).days)
                d[eid][1] = deduct_date
            # elif d[eid][0][1] >= deduct_date:
            #     print(eid, reg_date, deduct_date)
            else:
                pass
                # print(eid)
                # print(d[eid])
                # print(reg_date)
                # print(deduct_date)
                # stop2b
        # if l[0] == t.eid:
        #     print(t)
        #     print(reg_date)
        #     print(deduct_date)
        #     print(t.eid)
        #     print(l)
        #     delta = (reg_date - l[2]).days
        #     print('delta', delta)
        #     assert delta < 180 or delta != np.nan
            # if t.reg_date is not pd.NaT:
            #     stop1
            # assert t.reg_date >= l[2]
            # if not deduct_date is pd.NaT:
            #     if not l[1] is pd.NaT:
            #         assert deduct_date >= l[1], (t.eid, deduct_date, reg_date, l)
                # if not l[2] is pd.NaT:
                #     assert deduct_date >= l[2], (t.eid, deduct_date, reg_date, l)
        # l = [t.eid, reg_date, deduct_date]

    # d_registrations = pd.read_csv(path, sep='\t', parse_dates=['reg_date'])[[
    #     'eid', 'reg_date']].set_index('eid').dropna().groupby('eid').transform('min').to_dict()['reg_date']

    return d


def convert_to_array(multimorbidity, smi, d_id2range, d_deaths, d_registrations):

    path1 = 'gp_readv2toicd10.txt.gz'
    path2 = 'gp_readv3toicd10.txt.gz'
    path3 = 'hes_three_twodigit_withdup.txt.gz'

    skip_event_dt = (
    # where clinical event or prescription date precedes participant date of birth it has been changed to 01/01/1901
    '01/01/1901',
    # Where the date matches participant date of birth it has been changed to 02/02/1902.
    '02/02/1902',
    # Where the date follows participant date of birth but is in the year of their birth it has been changed to 03/03/1903.
    '03/03/1903',
    # Where the date is in the future (and is presumed to be a placeholder or other system default) it has been changed to 07/07/2037.
    '07/07/2037',
    '01/01/1941',
    )

    f_range = list(range(1, 3 + 1)) + list(range(10, 19 + 1))

    d = {}
    d_eid_to_icd10_codes = {}
    event_dts = set()
    icd10_codes = set()
    icd10_groups = set()
    for path, sep, col_icd10_code, cols_event_dt in (
        (
            path3, '\t', 'code',
            ('epistart', 'elecdate', 'admidate', 'disdate'),
            ),
        (path1, '|', 'icd10_code', ('event_dt',),),
        (path2, '|', 'icd10_code', ('event_dt',),),
        ):
        print('reading', path)
        df = pd.read_csv(path, sep=sep)
        # df = pd.read_csv(path, sep=sep, nrows=1000000)  # tmp!!!
        if path == path3:
            print('reading hesin_24Jun2021.txt.gz')
            df = pd.merge(
                df,
                pd.read_csv('hesin_24Jun2021.txt.gz', sep='\t',)[[
                    'eid', 'ins_index',
                    # 'arr_index',
                    # 'ccstartdate',
                    'epistart',  # Date episode started
                    'elecdate',  # Date of decision to admit
                    'admidate',  # Date of admission
                    'disdate',  # Date of discharge
                    ]],
                # on=['eid', 'ins_index', 'arr_index'],
                on=['eid', 'ins_index'],
                )

        for t in df[[
            'eid', col_icd10_code, *cols_event_dt]].dropna().itertuples():
            eid = t.eid

            for col_event_dt in cols_event_dt:
                event_dt = getattr(t, col_event_dt)
                if event_dt == '':
                    continue
                break
            else:
                continue

            if event_dt in skip_event_dt:
                continue

            # Reformat date to make it sortable.
            event_dt = datetime.datetime.strptime(event_dt, '%d/%m/%Y')#.strftime('%Y-%m-%d')

            for icd10_code in re.split('[, +]', getattr(t, col_icd10_code)):

                if '-' in icd10_code:
                    continue
                if len(icd10_code) > 4:
                    assert len(icd10_code) == 5, icd10_code
                assert len(icd10_code) in (3, 4, 5), (eid, icd10_code)
                assert '.' not in icd10_code, icd10_code
                if icd10_code[0] == 'F' and icd10_code[1:3] == '00':
                    icd10_code = icd10_code[0] + icd10_code[2:]
                if not icd10_code[:3] in smi and (
                    not icd10_code[3:] in multimorbidity.get(icd10_code[:3], set())
                    and '' not in multimorbidity.get(icd10_code[:3], set())
                    ):
                    continue

                event_dts.add(event_dt)
                icd10_codes.add(icd10_code)
                icd10_group = d_id2range.get(icd10_code[:3], 'Unknown')
                if icd10_group == 'Unknown':
                    print('tmp204', icd10_code)

                # ranges are fine, but not for the four SMI diagnoses- they need to be separate for each of the foru
                if icd10_code[:3] in smi:
                    k = icd10_code[:3]
                else:
                    if icd10_group in ('F20-F29', 'F30-F39'):
                        continue
                    if icd10_code[0] == 'F' and int(icd10_code[1:3]) not in f_range:
                        print('tmp226', icd10_code)
                        continue
                    if icd10_group in (
                        'J40-J47', # Chronic lower respiratory diseases
                        'J60-J70', # Lung diseases due to external agents
                        ):
                        icd10_group = 'J40-J47_and_J60-J70'
                    k = icd10_group

                icd10_groups.add(icd10_group)

                # d2.setdefault(eid, {}).setdefault(k, set()).add(event_dt)
                d.setdefault(eid, {}).setdefault(k, set()).add(event_dt)
                d_eid_to_icd10_codes.setdefault(eid, set()).add(icd10_code)

    eids = set(d.keys())

    print('eids', len(eids))
    print('icd10_codes', len(icd10_codes))
    print('icd10_groups', len(icd10_groups))
    print()

    # with open('out/eid_icd10_combined.txt', 'w') as f:
    #     for eid, icd10_codes in d_eid_to_icd10_codes.items():
    #         for icd10_code in icd10_codes:
    #             print(eid, icd10_code, sep='\t', file=f)

    # Get counts prior to non-SMI removal.
    icd10_groups_counts = collections.Counter()
    for eid in set(eids):
        for k in d[eid].keys():
            icd10_groups_counts[k] +=1
    with open("out/icd10_group_counts.txt", 'w') as f:
        for k, v in  icd10_groups_counts.most_common():
            f.write( "{} {}\n".format(k,v) )

    icd10_codes = set()
    icd10_groups = set()
    # Remove individiuals without SMI and get new set of ICD10 codes.
    for eid in set(eids):
        if len(set((
            icd10_code[:3] for icd10_code in d_eid_to_icd10_codes[eid]
            )) & set(smi)) == 0:
            del d[eid]
            eids.remove(eid)
            del d_eid_to_icd10_codes[eid]
        else:
            icd10_groups |= set(d[eid].keys())
            icd10_codes |= set(d_eid_to_icd10_codes[eid])

    with open('out/eids.txt', 'w') as f:
        print('\n'.join(map(str, sorted(eids))),  file=f)

    with open('out/dict.pkl', 'wb') as f:
        # json.dump(d, f)
        pickle.dump(d, f)

    with open('out/icd10_groups.txt', 'w') as f:
        print('\n'.join(sorted(icd10_groups)), file=f)

    with open('out/icd10_codes.txt', 'w') as f:
        print('\n'.join(sorted(icd10_codes)), file=f)

    print('eids', len(eids))
    print('icd10_codes', len(icd10_codes))
    print('icd10_groups', len(icd10_groups), icd10_groups)
    print('min(event_dts)', min(event_dts))
    print('max(event_dts)', max(event_dts))

    year_max = int(max(event_dts).year)
    year_min = int(min(event_dts).year)

    # import collections
    # c = collections.Counter()
    # for i_eid, eid in enumerate(sorted(eids)):
    #     death = datetime.datetime.strptime(
    #         d_deaths.get(eid, '31/12/9999'), '%d/%m/%Y')
    #     for i_icd10_group, icd10_group in enumerate(sorted(icd10_groups)):
    #         dates = d[eid].get(icd10_group)
    #         if dates is None:
    #             continue
    #         c[icd10_group] += 1
    # print(c.most_common())
    # exit()

    X = np.zeros((len(eids), year_max - year_min + 1, len(icd10_groups)))
    W = np.zeros((len(eids), year_max - year_min + 1, len(icd10_groups)))

    for i_eid, eid in enumerate(sorted(eids)):
        death = datetime.datetime.strptime(
            d_deaths.get(eid, '31/12/9999'), '%d/%m/%Y')
        reg_date, deduct_date = d_registrations.get(eid, [pd.NaT, pd.NaT])

        for i_icd10_group, icd10_group in enumerate(sorted(icd10_groups)):
            dates = d[eid].get(icd10_group)

            if dates is not None:
                year0 = min(dates).year
                year1 = min(death.year, year_max)
                for i_year in range(year0 - year_min, year1 - year_min + 1):
                    X[i_eid][i_year][i_icd10_group] = 1

            if not reg_date is pd.NaT and not reg_date is None:
                year0 = max(
                    year_min,
                    reg_date.year,
                    )
            else:
                year0 = year_min
            if not deduct_date is pd.NaT and not deduct_date is None:
                year1 = min(
                    year_max,
                    deduct_date.year,
                    death.year,
                    )
            for i_year in range(year0 - year_min, year1 - year_min + 1):
                W[i_eid][i_year][i_icd10_group] = 1

    # missingness

    # for i_eid, eid in enumerate(sorted(eids)):
    #     print(eid, 1 - W[i_eid,:,:].sum() / np.prod(W[i_eid,:,:].shape) )

    for i_year, year in enumerate(range(year_min, year_max + 1)):
        print(year, 1 - W[:,i_year,:].sum() / np.prod(W[:,i_year,:].shape) )

    # for i_icd10_group, icd10_group in enumerate(sorted(icd10_groups)):
    #     print(icd10_group, 1 - W[:,:,i_icd10_group].sum() / np.prod(W[:,:,i_icd10_group].shape) )

    with open('W_granular.npy', 'wb') as f:
        np.save(f, W)

    with open('X_granular.npy', 'wb') as f:
        np.save(f, X)

    # with open('W.npy', 'wb') as f:
    #     np.save(f, W)

    # with open('X.npy', 'wb') as f:
    #     np.save(f, X)

    return


def get_multimorbidity_codes():

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
                multimorbidity.setdefault(line[:3], set()).add(
                    line[3:-1].replace('.', ''))

    return multimorbidity


def icd_id2range():

    d_id2range = dict()
    with open('icd10.level2.ranges.txt') as f:
        for line in f:
            s1 = line[1:3]
            s2 = line[5:7]
            for i in range(int(s1, 16), int(s2, 16) + 1):
                d_id2range[line[0] + format(i, 'X').zfill(2)] = line[:7]

    d_id2range['E14'] = 'E08-E13' # Diabetes mellitus

    return d_id2range


if __name__ == '__main__':
    main()
