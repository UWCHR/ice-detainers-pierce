#
# :date: 2018-01-28
# :author: PN
# :copyright: GPL v2 or later
#
# ice-detainers-pierce/charges/src/charges.py
#
#

import argparse
import pandas as pd
import sys
import yaml
if sys.version_info[0] < 3:
    raise "Must be using Python 3"


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pierce", required=True)
    parser.add_argument("--booking_charges", required=True)
    parser.add_argument("--fugitive_charges", required=True)
    parser.add_argument("--seriousness", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":

    args = _get_args()
    print(args)

    df = pd.read_csv(args.pierce, sep='|', compression='gzip')
    raw_charge_cols = ['booking_charge_desc', 'booking_charge_txt']
    charge_desc_txt = df[raw_charge_cols].drop_duplicates()
    charge_desc_txt.to_csv('frozen/charge_desc_txt.csv', index=False)

    # print(df.info())

    start_length = len(df)
    start_bookings = len(set(df['booking_id']))
    start_hold_count = sum(df.drop_duplicates(subset='booking_id')['imm_hold'])

    pre = len(df)
    to_drop = df['booking_charge_desc'] == 'ENTRY ERROR'
    df = df[~to_drop]
    post = len(df)
    entry_error_dropped = pre - post
    msg = ' booking charges with value \'ENTRY ERROR\' dropped.'
    print(f'{pre-post}{msg}')

    charge_counts = df.groupby('booking_id')['cause_num'].count()

    df.set_index('booking_id', inplace=True)

    df['charge_count'] = charge_counts

    df.reset_index(inplace=True)

    df['charge_count'].max()

    df['charge_topcount'] = df['charge_count'].copy()
    mask = df['charge_topcount'] >= 10
    df.loc[mask, 'charge_topcount'] = 10

    # Drop records with no associated cause number?
    # predrop = len(df)
    no_charges = df['charge_count'] == 0
    # df = df[~no_charges]
    # postdrop = len(df)
    # print(f'Dropped {predrop - postdrop} records with no charges.')

    # Get charge seriousness
    book = pd.read_csv(args.booking_charges)
    fugit = pd.read_csv(args.fugitive_charges)

    charge_cols = ['booking_charge_desc', 'category', 'type']
    book = book[charge_cols]

    with open(args.seriousness, "rt") as yamlfile:
        seriousness = yaml.load(yamlfile)

    book['seriousness'] = book['category'].replace(to_replace=seriousness)
    fugit['seriousness'] = fugit['category'].replace(to_replace=seriousness)
    # print(fugit.columns)

    fugitive_mask = df['booking_charge_desc'] == 'FUGITIVE'
    fugitive_df = df[fugitive_mask]
    non_fugitive_df = df[~fugitive_mask]

    # print(fugitive_df.info())

    premerge = len(fugitive_df)
    fugitive_df = pd.merge(fugitive_df, fugit, on=['booking_charge_desc', 'booking_charge_txt'], how='left')
    postmerge = len(fugitive_df)
    assert premerge == postmerge
    del premerge, postmerge

    # print(fugitive_df.info())

    # print(non_fugitive_df.info())

    premerge = len(non_fugitive_df)
    non_fugitive_df = pd.merge(non_fugitive_df, book, on=['booking_charge_desc'], how='left')
    postmerge = len(non_fugitive_df)
    assert premerge == postmerge
    del premerge, postmerge

#    print(non_fugitive_df.info())

    preconcat = len(df)
    df = pd.concat([non_fugitive_df, fugitive_df], sort=False)
    postconcat = len(df)
    assert preconcat == postconcat
    del preconcat, postconcat

    df.reset_index(inplace=True, drop=True)
    df = df.sort_values(by=['booking_id'])

    to_drop = df['category'] == 'unknown'
    predrop = len(df)
    df = df[~to_drop]
    postdrop = len(df)
    unknown_category_dropped = predrop - postdrop

    print(f'Dropped {unknown_category_dropped} records of unknown category.')
    del predrop, postdrop

    df['type'] = df['type'].str.strip()
    df['type'] = df['type'].str.replace(' ', '_')

    null_seriousness = df['seriousness'].isnull()
    null = df[null_seriousness]
    null.drop_duplicates(subset='booking_charge_desc').to_csv('output/null.csv')

    predrop = len(df)
    predropbookingset = len(set(df['booking_id']))
    df = df.dropna(subset=['seriousness'])
    postdrop = len(df)
    postdropbookingset = len(set(df['booking_id']))

    null_seriousness_dropped_recs = predrop - postdrop
    null_seriousness_dropped_bookings = predropbookingset - postdropbookingset

    print(f'Dropped {null_seriousness_dropped_recs} records with null seriousness.')
    print(f'Dropped {null_seriousness_dropped_bookings} bookings.')

    del predrop, postdrop, predropbookingset, postdropbookingset

    # Find most serious charge
    df['seriousness'] = df['seriousness'].astype(float)
    max_seriousness = df.groupby('booking_id')['seriousness'].max()

    max_id = df.groupby('booking_id')['seriousness'].idxmax()

    max_type = []
    max_charge_desc = []
    for i in max_id:
        charge_type = df.loc[i, :]['type']
        charge_desc = df.loc[i, :]['booking_charge_desc']
        max_type.append(charge_type)
        max_charge_desc.append(charge_desc)

    seriousness_type = max_seriousness.to_frame(name='max_seriousness')
    seriousness_type['max_charge_type'] = max_type
    seriousness_type['max_charge_desc'] = max_charge_desc

    seriousness_type.reset_index(inplace=True)

    max_type_dummies = pd.get_dummies(seriousness_type['max_charge_type'])

    seriousness_type = pd.concat([seriousness_type, max_type_dummies], axis=1)

    df = pd.merge(df, seriousness_type, on='booking_id', how='left')

    felony_mask = df['max_seriousness'] >= 4
    misdemeanor_mask = (df['max_seriousness'] >= 2) & (df['max_seriousness'] <=3 )

    df['felony'] = 0
    df.loc[felony_mask, 'felony'] = 1

    df['misdemeanor'] = 0
    df.loc[misdemeanor_mask, 'misdemeanor'] = 1

    assert sum(df[df['felony'] == 1]['misdemeanor']) == 0
    assert sum(df[df['misdemeanor'] == 1]['felony']) == 0

    hold_mask = df['imm_hold'] == 1
    df[hold_mask].to_csv('output/allholds.csv', sep='|', compression='gzip', index=False)

    max_charge_other = df['max_charge_type'] == 'other'

    df[max_charge_other].describe().T.to_csv('output/max_charge_other_desc.csv')
    df[~max_charge_other].describe().T.to_csv('output/max_charge_not_other_desc.csv')

    predrop = len((set(df['booking_id'])))
    df[max_charge_other].to_csv('output/max_charge_other.csv', index=False)
    df = df[~max_charge_other]
    postdrop = len((set(df['booking_id'])))
    max_charge_other_dropped = predrop - postdrop

    print(f'Dropped {max_charge_other_dropped} booking records with max charge type "other".')

    df[no_charges].to_csv('output/no_charges.csv')

    end_length = len(df)
    end_bookings = len(set(df['booking_id']))
    end_hold_count = sum(df.drop_duplicates(subset='booking_id')['imm_hold'])

    print(df.info())

    df.to_csv(args.output, sep="|", compression='gzip', index=False)

    data = dict(
        pd_version=pd.__version__,
        sys_version=sys.version,
        end_bookings=end_bookings,
        end_hold_count=end_hold_count,
        end_length=end_length,
        entry_error_dropped=entry_error_dropped,
        max_charge_other_dropped=max_charge_other_dropped,
        null_seriousness_dropped_bookings=null_seriousness_dropped_bookings,
        null_seriousness_dropped_recs=null_seriousness_dropped_recs,
        start_bookings=start_bookings,
        start_hold_count=start_hold_count,
        start_length=start_length,
        unknown_category_dropped=unknown_category_dropped
    )

    with open('output/charge-vars.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

# End.
