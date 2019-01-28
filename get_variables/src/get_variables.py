#
# :date: 2019-01-28
# :author: PN
# :copyright: GPL v2 or later
#
# ice-detainers-pierce/get-variables/src/get-variables.py
#
#

import argparse
import pandas as pd
import numpy as np
import sys
# from ethnicolr import pred_fl_reg_name
if sys.version_info[0] < 3:
    raise "Must be using Python 3"


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pierce", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def booking_mask(df_to_mask, id_col, mask_col, mask_term, new_col):
    mask = df_to_mask[mask_col] == mask_term
    ids_to_mask = set(df_to_mask[mask][id_col])
    ids_in_mask = df_to_mask[id_col].isin(ids_to_mask)
    df_to_mask[new_col] = 0
    df_to_mask.loc[ids_in_mask, new_col] = 1
    return(df_to_mask)


if __name__ == "__main__":

    args = _get_args()
    print(args)

    read_csv_opts = {'sep': '|',
                     'quotechar': '"',
                     'compression': 'gzip',
                     'encoding': 'utf-8'}
    to_csv_opts = {'sep': '|',
                   'quotechar': '"',
                   'index': False,
                   'compression': 'gzip',
                   'encoding': "utf-8"}

    df = pd.read_csv(args.pierce, **read_csv_opts)

    # # Drop records where race is unknown
    # predrop = len(df)
    # df = df[df['unknown'] == 0]
    # df.drop('unknown', axis=1, inplace=True)
    # postdrop = len(df)
    # print(f'Dropped {predrop - postdrop} records with unknown race.')

    df['release_dt'] = pd.to_datetime(df['release_dt'])
    df['booking_dt'] = pd.to_datetime(df['booking_dt'])
    df['charge_release_dt'] = pd.to_datetime(df['charge_release_dt'])
    df['time_detained'] = df['release_dt'] - df['booking_dt']
    df['time_detained'] = df['time_detained'] / np.timedelta64(1, 'D')
    df.loc[:, 'log_time_detained'] = np.log(df['time_detained'])

    booking_mask(df, 'booking_id',
                     'release_disposition_desc',
                     'Administrative Booking',
                     'admin_booking')

    booking_mask(df, 'booking_id',
                     'release_disposition_desc',
                     'Immigration',
                     'release_to_imm')

    booking_mask(df, 'booking_id',
                     'release_disposition_desc',
                     'Cancel Detainer',
                     'cancel_detainer')

    booking_mask(df, 'booking_id',
                     'booking_charge_desc',
                     'HOLD IMMIGRAT',
                     'imm_hold')

    # # Surname analysis: reqiures ethnicolr import
    # name_tokens = df['inmate_name'].str.split(',', expand=True)
    # suffix_mask = name_tokens[2].notnull()
    # suffix = name_tokens[suffix_mask]
    # suffix = suffix.drop(3, axis=1)
    # suffix.columns = ['last_name', 'suffix', 'first_name']
    # f_l_only = name_tokens[~suffix_mask]
    # f_l_only = f_l_only.dropna(axis=1)
    # f_l_only.columns = ['last_name', 'first_name']
    # names = pd.concat([f_l_only, suffix], sort=False)
    # names = names.sort_index()
    # names = pred_fl_reg_name(names, 'last_name', 'first_name')
    # names.columns = ['last_name',
    #                  'first_name',
    #                  'suffix',
    #                  'race_ethnicolr',
    #                  'asian_ethnicolr',
    #                  'hispanic_ethnicolr',
    #                  'nh_black_ethnicolr',
    #                  'nh_white_ethnicolr']
    # df = df.join(names)

    # Find final release disp
    last_rel_idx = df.groupby('booking_id')['charge_release_dt'].idxmax()
    last_rel_charge = df.loc[last_rel_idx]['release_disposition_desc']
    last_rel_idx = last_rel_idx.reset_index().set_index('charge_release_dt')
    temp = last_rel_idx.join(last_rel_charge)
    temp.reset_index(inplace=True)
    temp['final_release_disp'] = temp['release_disposition_desc']
    cols = ['charge_release_dt', 'release_disposition_desc']
    temp.drop(cols, axis=1, inplace=True)
    df = pd.merge(df, temp, on='booking_id', how='left')
    del temp

    print(df.info())

    df.to_csv(args.output, sep="|", compression='gzip', index=False)

# End.
