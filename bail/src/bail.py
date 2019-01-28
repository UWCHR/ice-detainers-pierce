# :date: 2018-08-13
# :author: PN
# :copyright: GPL v2 or later
#
# JailData/county/pierce/bail/src/bail.py
#
#
import argparse
import pandas as pd
import sys

if sys.version_info[0] < 3:
    raise "Must be using Python 3"


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pierce", required=True)
    parser.add_argument("--bail", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":

    args = _get_args()
    print(args)

    df = pd.read_csv(args.pierce, sep='|', compression='gzip')
    bail = pd.read_csv(args.bail, sep='|', compression='gzip')

    g = bail.groupby(['booking_id', 'bail_receipt_id'])

    total_fine = g.agg(lambda x: x.drop_duplicates('bail_receipt_id', keep='first').fine_amt.sum())

    t = total_fine.to_frame('total').reset_index().set_index('booking_id')

    t = t.groupby('booking_id')['total'].sum()

    df.set_index('booking_id', inplace=True)

    df['total_fine'] = t
    mask = df['total_fine'].isnull()
    df.loc[mask, 'total_fine'] = 0

    df.reset_index(inplace=True)

    paid_bail = df['total_fine'] > 0
    df['paid_bail'] = paid_bail

    print(df.info())
    df.to_csv(args.output, index=False, sep='|', compression='gzip')

# END.
