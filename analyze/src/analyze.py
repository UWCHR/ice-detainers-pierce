# :date: 2018-01-28
# :author: PN
# :copyright: GPL v2 or later
#
# ice-detainers-pierce/analyze/src/analyze.py
#

import argparse
import pandas as pd
import numpy as np
import sys
import yaml

import statsmodels.formula.api as smf
import scipy.stats as scipystats

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['figure.figsize'] = (8, 6)

if sys.version_info[0] < 3:
    raise "Must be using Python 3"


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pierce", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def barplot(data, output_path):

    # Create the bar plot
    ax = sns.barplot(
        x='felony_misdemeanor',
        y='time_detained',
        hue='imm_hold_text',
        hue_order=['Detainer', 'No detainer'],
        palette='tab10',
        data=df)

    # title = "Mean jail time for inmates by ICE detainer status"
    # ax.set_title(title, fontsize=18)
    ax.legend(loc=2, fontsize=16)
    ax.set_ylabel('Jail days', fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xlabel('Booking charge category', fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.invert_xaxis()
    plt.savefig(output_path)
    return plt, ax


def test_output_data():

    with open('output/data.yaml', 'rt') as yfile:
        new_data = yaml.load(yfile)

    with open('frozen/data.yaml', 'rt') as yfile:
        old_data = yaml.load(yfile)

    assert new_data == old_data, "Output data has changed, is this expected?"


if __name__ == "__main__":

    args = _get_args()
    print(args)

    df = pd.read_csv(args.pierce, sep='|', compression='gzip', encoding='utf-8')

    df['race_desc'] = df['race_desc'].str.lower()
    df['gender_desc'] = df['gender_desc'].str.lower()
    df['booking_dt'] = pd.to_datetime(df['booking_dt'])
    df['release_dt'] = pd.to_datetime(df['release_dt'])
    df['charge_release_dt'] = pd.to_datetime(df['charge_release_dt'])

    hold_mask = df['imm_hold'] == 1

    total_records = len(df)
    unique_booking_ids = len(set(df.booking_id))
    earliest_booking = df.booking_dt.min()
    latest_booking = df.booking_dt.max()
    earliest_release = df.release_dt.min()
    latest_release = df.release_dt.max()
    min_charges = float(df['charge_count'].min())
    max_charges = float(df['charge_count'].max())
    max_charge_topcout = float(df['charge_topcount'].max())

    start_date = f'{earliest_release.month_name()} {str(earliest_release.year)}'
    end_date = f'{latest_release.month_name()} {str(latest_release.year)}'
    start_date_full = f'{earliest_release.month_name()} {earliest_release.day}, {earliest_release.year}'
    end_date_full = f'{latest_release.month_name()} {latest_release.day}, {latest_release.year}'

    hold_pr = df.loc[hold_mask]['release_disposition_desc'] == 'PR'
    no_hold_pr = df.loc[~hold_mask]['release_disposition_desc'] == 'PR'
    hold_pr_percent = float(sum(hold_pr) / len(df[hold_mask]) * 100)
    no_hold_pr_percent = float(sum(no_hold_pr) / len(df[~hold_mask]) * 100)

    hold_ids = df[df['imm_hold'] == 1]['booking_id'].tolist()

    df['release_dt']= pd.to_datetime(df['release_dt'])
    df['charge_release_dt'] = pd.to_datetime(df['charge_release_dt'])
    df['charge_delta'] = df['charge_release_dt'] - df['booking_dt']
    df['charge_delta'] = df['charge_delta'] / np.timedelta64(1,'h') 
    df['charge_release_delta'] = df['charge_release_dt'] - df['release_dt']
    df['charge_release_delta'] = df['charge_release_delta'] / np.timedelta64(1,'h')

    held_longer_on_detainer = []
    how_much_longer = []

    for b_id in hold_ids:
        booking = df[df['booking_id'] == b_id]
        non_hold_charge_deltas = booking[booking['booking_charge_desc'] != 'HOLD IMMIGRAT']['charge_delta']
        hold_charge_delta = float(booking[booking['booking_charge_desc'] == 'HOLD IMMIGRAT']['charge_delta'])
        hold_release_delta = float(booking[booking['booking_charge_desc'] == 'HOLD IMMIGRAT']['charge_release_delta'])

        if hold_charge_delta > non_hold_charge_deltas.max():
            held_longer_on_detainer.append(b_id)
        how_much_longer.append([hold_charge_delta - non_hold_charge_deltas.max(), hold_release_delta, b_id])

    held_longer_on_detainer_df = df[df['booking_id'].isin(held_longer_on_detainer)]

    held_longer_on_detainer = len(set(held_longer_on_detainer_df['booking_id']))

    # DUPLICATES DROPPED HERE

    df_all = df.copy()
    df = df.drop_duplicates(subset=['booking_id'])
    assert len(df) == len(set(df_all['booking_id']))

    hold_count = sum(df['imm_hold'])
    assert hold_count == 188

    hold_mask = df['imm_hold'] == 1

    hold_median_jail_days = float(df[hold_mask]['time_detained'].median())
    no_hold_median_jail_days = float(df[~hold_mask]['time_detained'].median())
    overall_median_jail_days = float(df['time_detained'].median())
    hold_mean_jail_days = float(df[hold_mask]['time_detained'].mean())
    no_hold_mean_jail_days = float(df[~hold_mask]['time_detained'].mean())
    overall_mean_jail_days = float(df['time_detained'].mean())
    max_jail_days = float(df['time_detained'].max())

    gender_percent = df['gender_desc'].value_counts() / len(df) * 100
    hold_gender_percent = df[hold_mask]['gender_desc'].value_counts() / len(df[hold_mask]) * 100
    male_percent = float(gender_percent['male'])
    hold_male_percent = float(hold_gender_percent['male'])

    race_count = df['race_desc'].value_counts()
    race_percent = race_count/len(df)*100
    hispanic_percent = float(race_percent['hispanic'])
    hold_percent = float(len(df[hold_mask]) / len(df) * 100)
    hold_race_count = df[hold_mask]['race_desc'].value_counts()
    hold_race_percent = hold_race_count/len(df[hold_mask])*100
    hispanic_hold_percent = float(hold_race_percent['hispanic'])

    # Race hold table
    race_ice_holds = df.groupby('race_desc')['imm_hold'].sum()
    n = df['race_desc'].value_counts()
    race_percent = n / len(df) * 100
    race_hold_df = pd.DataFrame([n, race_percent, race_ice_holds]).T
    cols = ['n',
            'race_percent',
            'detainer_count']
    race_hold_df.columns = cols
    race_hold_df.reset_index(inplace=True)
    race_hold_df = race_hold_df.append(race_hold_df.sum(numeric_only=True), ignore_index=True)
    index = ['White',
             'Black',
             'Hispanic',
             'Asian/ \\cr Pacific Islander',
             'American Indian/ \\cr Alaskan Native',
             'Unknown',
             'Total']
    race_hold_df['index'] = index
    race_hold_df = race_hold_df.set_index('index')
    race_hold_df.index.name = 'race'
    int_cols = ['n', 'detainer_count']
    race_hold_df[int_cols] = race_hold_df[int_cols].astype(int)
    holds_as_percent_of_race = race_hold_df['detainer_count']/race_hold_df['n'] * 100
    race_hold_df['detainer_percent'] = holds_as_percent_of_race
    race_hold_df.reset_index(inplace=True)

    race_hold_df.to_csv('output/race_hold_table.csv',
                        index=False,
                        float_format='%.2f')

    # race_hold_df.to_latex('output/race_hold_table.tex',
    #                       column_format='l|r|r|r|r',
    #                       index=False,
    #                       float_format='%.2f')

    hispanic_holds_as_percent_of_race = float(holds_as_percent_of_race['Hispanic'])
    hispanic_chance_of_hold = float(round(100 / holds_as_percent_of_race['Hispanic']))
    api_holds_as_percent_of_race = float(holds_as_percent_of_race['Asian/ \\cr Pacific Islander'])

    # Mean jail days by charge category, detainer status
    df['imm_hold_text'] = df['imm_hold'].replace({0: 'No detainer', 1: 'Detainer'})

    df['felony_misdemeanor'] = df['max_seriousness'].replace(
                             {6: 'Felony',
                              5: 'Felony',
                              4: 'Felony',
                              3: 'Misdemeanor',
                              2: 'Misdemeanor',
                              1: None,
                              0: None})

    felony_percent = sum(df['felony_misdemeanor'] == 'Felony') / len(df) * 100
    misdemeanor_percent = sum(df['felony_misdemeanor'] == 'Misdemeanor') / len(df) * 100

    # Do we need to drop "None" values in table used here?

    table = pd.pivot_table(df, values='time_detained',
                           index='felony_misdemeanor',
                           columns='imm_hold_text',
                           aggfunc=np.mean)
    barplot(table, 'output/MeanJailTime.png')

    table['% increase'] = table['Detainer'] / table['No detainer'] * 100
    table.to_csv('output/felony_misdemeanor.csv', float_format='%.2f')

    levene = scipystats.levene(df[hold_mask]['log_time_detained'], df[~hold_mask]['log_time_detained'])
    # print(levene)

    ttest = scipystats.ttest_ind(df[hold_mask]['log_time_detained'], df[~hold_mask]['log_time_detained'])
    # print(ttest)
    hold_significance = ttest.pvalue
    assert hold_significance < 0.001

    # Bail
    bail_crosstab = pd.crosstab(df.paid_bail, df.imm_hold)
    bail_crosstab_normalized = pd.crosstab(df.paid_bail, df.imm_hold, normalize='columns')*100
    bail_chi2 = scipystats.chi2_contingency(bail_crosstab)

    levene = scipystats.levene(df[hold_mask]['paid_bail'], df[~hold_mask]['paid_bail'])
    # print(levene)

    ttest = scipystats.ttest_ind(df[hold_mask]['paid_bail'], df[~hold_mask]['paid_bail'], equal_var=False)
    bail_significance = ttest.pvalue
    assert bail_significance < 0.001

    no_hold_paid_bail = float(bail_crosstab.loc[True, 0])
    hold_paid_bail = float(bail_crosstab.loc[True, 1])
    no_hold_paid_bail_percent = float(bail_crosstab.loc[True, 0] / len(df[~hold_mask]))
    hold_paid_bail_percent = float(bail_crosstab.loc[True, 1] / len(df[hold_mask]))

    paid_bail = df['paid_bail'] is True

    # Regression
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    data = df[['time_detained',
               'log_time_detained',
               'imm_hold',
               'charge_topcount',
               'male',
               'female',
               'white',
               'black',
               'hispanic',
               'amer_indian_alaskan',
               'asian_pacific_island',
               'unknown',
               'max_seriousness',
               'public_order',
               'felony',
               'misdemeanor',
               'drug',
               'sex',
               'property',
               'violent']]

    data.describe().T.to_csv('descriptive_stats.csv')

    formula = "log_time_detained ~ imm_hold + max_seriousness + charge_topcount + drug + sex + property + violent + male + black + hispanic + amer_indian_alaskan + asian_pacific_island + unknown"
    reg = smf.ols(formula=formula, data=df).fit()

    assert reg.pvalues['imm_hold'] < 0.001

    regression_n = len(df)

    params = reg.params
    params.name = 'coef'

    std_err = reg.bse
    std_err.name = 'std err'

    t_vals = reg.tvalues
    t_vals.name = 't'

    p_vals = reg.pvalues
    p_vals.name = 'P>|t|'

    conf_int = reg.conf_int()
    conf_int.columns = ['.025','.975']

    def join(row):
        ''' given a row, return the concat of 0 and 1 values '''
        codes = [getattr(row, '.025'), getattr(row, '.975')]
        joined = ', '.join(["% 0.2f" % f for f in codes]) 
        return f'({joined})'

    conf_int = conf_int.apply(join, axis=1)
    conf_int.name = '95% CI'

    coef_interpret = (np.power(np.e, reg.params) - 1) * 100
    coef_interpret.name = 'Impact'

    imm_hold_impact = coef_interpret['imm_hold']
    imm_hold_impact_percent = float((imm_hold_impact))
    imm_hold_multiplier = float(np.power(np.e, reg.params['imm_hold']))

    imm_hold_impact_conf_int = np.power(np.e, reg.conf_int().loc['imm_hold'])
    imm_hold_multiplier_lower, imm_hold_multiplier_higher = imm_hold_impact_conf_int

    cols = [params, std_err, t_vals, p_vals, conf_int, coef_interpret]

    conf_int_interpret = (np.power(np.e, reg.conf_int()) - 1)
    conf_int_interpret.columns = ['0.025', '0.975']
    imm_hold_impact_lower = float(conf_int_interpret.loc['imm_hold', '0.025'])
    imm_hold_impact_higher = float(conf_int_interpret.loc['imm_hold', '0.975'])

    rsquared_adj = float(reg.rsquared_adj)

    index = ['Intercept',
             'ICE detainer',
             'Seriousness rank',
             'Number of charges',
             'Drug offense',
             'Sex offense',
             'Property offense',
             'Violent offense',
             'Male',
             'Black',
             'Hispanic',
             'American Indian/ \\cr Alaska Native',
             'Asian/ \\cr Pacific Islander',
             'Unknown race']

    summary = pd.concat(cols, axis=1)

    summary['index'] = index
    summary.set_index('index', inplace=True)

    summary.to_csv('output/regression_summary.csv')

    no_hold_log_time_detained = (reg.params['Intercept'] +
        (reg.params['imm_hold'] * 0) +
        (reg.params['charge_topcount'] * 2) +
        (reg.params['male'] * 1) +
        (reg.params['black'] * 0) +
        (reg.params['hispanic'] * 1) +
        (reg.params['amer_indian_alaskan'] * 0) +
        (reg.params['asian_pacific_island'] * 0) +
        (reg.params['unknown'] * 0) +
        (reg.params['max_seriousness'] * 4) +
        (reg.params['drug'] * 0) +
        (reg.params['sex'] * 0) +
        (reg.params['property'] * 0) +
        (reg.params['violent'] * 0))
    predicted_time_detained_no_hold = float(np.power(np.e, no_hold_log_time_detained))

    hold_log_time_detained = (reg.params['Intercept'] + 
         (reg.params['imm_hold'] * 1) + 
         (reg.params['charge_topcount'] * 2) +
         (reg.params['male'] * 1) +
         (reg.params['black'] * 0) +
         (reg.params['hispanic'] * 1) +
         (reg.params['amer_indian_alaskan'] * 0) +
         (reg.params['asian_pacific_island'] * 0) +
         (reg.params['unknown'] * 0) +
         (reg.params['max_seriousness'] * 4) +
         (reg.params['drug'] * 0) +
         (reg.params['sex'] * 0) +
         (reg.params['property'] * 0) +
         (reg.params['violent'] * 0))
    predicted_time_detained_hold = float(np.power(np.e, hold_log_time_detained))

    cost_per_day_cents = 12600

    df = df.copy()
    df.loc[:, 'approx_cost'] = df['time_detained'] * cost_per_day_cents

    non_detainer_mean_time = df[df['imm_hold'] == 0]['time_detained'].mean()
    non_detainer_avg_cost_cents = non_detainer_mean_time * cost_per_day_cents
    detainer_mean_time = df[df['imm_hold'] == 1]['time_detained'].mean()
    detainer_avg_cost_cents = detainer_mean_time * cost_per_day_cents

    diff = detainer_mean_time - non_detainer_mean_time

    extra_cost_per_detainer_cents = int(detainer_avg_cost_cents - non_detainer_avg_cost_cents)

    total_extra_cost_cents = int(extra_cost_per_detainer_cents * len(df[df['imm_hold'] == 1]))

    total_extra_cost = int(total_extra_cost_cents / 100)
    extra_cost_per_detainer_cents = int(extra_cost_per_detainer_cents)

    data = dict(
        api_holds_as_percent_of_race=api_holds_as_percent_of_race,
        cost_per_day_cents=cost_per_day_cents,
        earliest_booking_year=earliest_booking.year,
        end_date=end_date,
        end_date_full=end_date_full,
        extra_cost_per_detainer_cents=extra_cost_per_detainer_cents,
        felony_percent=felony_percent,
        held_longer_on_detainer=held_longer_on_detainer,
        hispanic_chance_of_hold=hispanic_chance_of_hold,
        hispanic_hold_percent=hispanic_hold_percent,
        hispanic_holds_as_percent_of_race=hispanic_holds_as_percent_of_race,
        hispanic_percent=hispanic_percent,
        hold_count=hold_count,
        hold_male_percent=hold_male_percent,
        hold_mean_jail_days=hold_mean_jail_days,
        hold_median_jail_days=hold_median_jail_days,
        hold_paid_bail=hold_paid_bail,
        hold_paid_bail_percent=hold_paid_bail_percent,
        hold_percent=hold_percent,
        hold_pr_percent=hold_pr_percent,
        imm_hold_impact_higher=imm_hold_impact_higher,
        imm_hold_impact_lower=imm_hold_impact_lower,
        imm_hold_multiplier=imm_hold_multiplier,
        imm_hold_multiplier_lower=imm_hold_multiplier_lower,
        imm_hold_multiplier_higher=imm_hold_multiplier_higher,
        imm_hold_impact_percent=imm_hold_impact_percent,
        male_percent=male_percent,
        max_charge_topcout=max_charge_topcout,
        max_charges=max_charges,
        max_jail_days=max_jail_days,
        min_charges=min_charges,
        misdemeanor_percent=misdemeanor_percent,
        no_hold_mean_jail_days=no_hold_mean_jail_days,
        no_hold_median_jail_days=no_hold_median_jail_days,
        no_hold_paid_bail=no_hold_paid_bail,
        no_hold_paid_bail_percent=no_hold_paid_bail_percent,
        no_hold_pr_percent=no_hold_pr_percent,
        overall_mean_jail_days=overall_mean_jail_days,
        overall_median_jail_days=overall_median_jail_days,
        pd_version=pd.__version__,
        predicted_time_detained_hold=format(predicted_time_detained_hold, '.0f'),
        predicted_time_detained_no_hold=format(predicted_time_detained_no_hold, '.0f'),
        regression_n=regression_n,
        rsquared_adj=rsquared_adj,
        start_date=start_date,
        start_date_full=start_date_full,
        sys_version=sys.version,
        total_extra_cost=total_extra_cost,
        total_records=total_records,
        unique_booking_ids=unique_booking_ids
    )

    with open('output/data.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # test_output_data()

# End.
