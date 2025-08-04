import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
#import ace_tools as tools

# Limit threads for performance
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# -----------------------------
# Step 1: Load and Clean Data
# -----------------------------
file_path = os.path.join(r"/Users/minimac/Downloads/data/", "lead_cost_bkt2.csv")
chunks = []
for chunk in pd.read_csv(file_path, parse_dates=['lead_date'], chunksize=20000, low_memory=True):
    # Ensure valid dates
    chunk['lead_date'] = pd.to_datetime(chunk['lead_date'], errors='coerce')
    chunk = chunk.dropna(subset=['lead_date'])
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

# -----------------------------
# Step 2: Feature Engineering
# -----------------------------
# bin by equitable volume, deciles
ranks = df['acquisition_cost'].rank(method='first', pct=True)
df['tier_decile_rank'] = pd.cut(ranks, bins=10, labels=False, include_lowest=True)

#filter for customer segment (peak/off-peak, NC/VIP)
df1 = df[df['NC_flg'] == 1] #New Customers
df2 = df[(df['NC_flg'] == 0) & (df['Store_Type'] == 'LEAD_GEN')]
df3 = df[(df['NC_flg'] == 1) & (df['Store_Type'] == 'LEAD_GEN')] #NC Peak
df4 = df[(df['NC_flg'] == 1) & (df['Store_Type'] == 'LEAD_GEN_OFF PEAK')] #NC Off-peak
df5 = df[(df['NC_flg'] == 0) & (df['Store_Type'] == 'LEAD_GEN')] #VIP Peak
df6 = df[(df['NC_flg'] == 0) & (df['Store_Type'] == 'LEAD_GEN_OFF PEAK')] #VIP Off-peak

df_backup = df
df = df2 # allows to revert or change primary df for filtered df

# explicit bins for tiers, allows for the cat quartile approach below
#tiers = ['0-10','10-20','20-30','30-40','40-50','50-100','100+']
#bins = [0,10,20,30,40,50,100,np.inf]

# Month period
df['month'] = df['lead_date'].dt.to_period('M').dt.to_timestamp()
# Restrict to actual data range
df = df[df['month'] <= df['month'].max()]

# find min/max of each decile (tier prep)
# per‐decile min & max of acquisition_cost
df['decile_min'] = df.groupby('tier_decile_rank')['acquisition_cost'] \
                     .transform('min')
df['decile_max'] = df.groupby('tier_decile_rank')['acquisition_cost'] \
                     .transform('max')

# combine into str for tier assignment
df['tier'] = (
    df['decile_min'].round(2).astype(str)
    + ' – ' +
    df['decile_max'].round(2).astype(str)
)

# alt for tiers
tiers = sorted(df['tier'].unique().tolist())

# view tier data
print(df[['acquisition_cost','tier_decile_rank','decile_min','decile_max','tier']].head(50))

# Alt cost tier by explicit bucket
#or
#df['tier'] = pd.cut(df['acquisition_cost'], bins=bins, labels=tiers, right=False)

# Flags
#df['is_originated'] = df.get('is_originated', 0).fillna(0) #alt for safe denominator
df['first_payment_default'] = df.get('first_payment_default', 0).fillna(0)

# Default-funded amount
df['fpd_funded'] = df.get('fundedAmount', 0).fillna(0) * df['first_payment_default']

# -----------------------------
# Step 3: Aggregate Metrics by Tier & Month
# -----------------------------
agg = (
    df.groupby(['tier','month'])
      .agg(
        leads=('lead_id','count'),
        originations=('is_originated','sum'),
        cost=('acquisition_cost','sum'),
        funded_total=('fundedAmount','sum'),
        fpd_funded=('fpd_funded','sum'),
        defaults=('first_payment_default','sum'),
        #uw_cost=('underwriting_cost','sum'), # add later or use static placeholder vals
        #bank_cost=('bankVerificationCost','sum'), #add later or use placeholder vals
        interest=('total_interest_paid','sum'),
        fees=('total_fees_paid','sum'),
        payments=('total_principal_paid', 'sum')
      )
      .reset_index()
)

# Safe denominators to control for div/0 issue
agg['originations_safe'] = agg['originations'].replace(0, np.nan)
agg['leads_safe'] = agg['leads'].replace(0, np.nan)

# -----------------------------
# Step 4: Compute Derived Metrics
# -----------------------------
# Net bad debt and default rate
agg['default_rate'] = (agg['defaults'] / agg['originations_safe']).fillna(0)
agg['net_bad_debt_per_loan'] = (agg['fpd_funded'] / agg['originations_safe']).fillna(0)
agg['bad_principal'] = (agg['fpd_funded']).fillna(0) # allows scaling, to add Never Pay or another loss feature
# Conversion
agg['conversion_rate'] = (agg['originations'] / agg['leads_safe']).fillna(0)
agg['conversion_trend'] = agg.groupby('tier')['conversion_rate'].pct_change()
# Costs per loan
agg['lead_cost_per_loan'] = (agg['cost'] / agg['originations_safe']).fillna(0)
#agg['bank_cost_per_loan'] = (agg['bank_cost'] / agg['originations_safe']).fillna(0)
#agg['uw_cost_per_loan'] = (agg['uw_cost'] / agg['originations_safe']).fillna(0)
agg['cost_per_loan'] = agg['lead_cost_per_loan'] #+ agg['uw_cost_per_loan'] + agg['bank_cost_per_loan']
# Net margin per loan, based on mature historical fundings
agg['net_margin_per_loan'] = ((agg['interest'] + agg['fees'] + agg['payments'] - agg['bad_principal']) /
                              agg['originations_safe']).fillna(0)
# Total variable cost and ROI
agg['total_variable_cost'] = agg['cost'] #+ agg['uw_cost'] + agg['bank_cost']
agg['ROI'] = ((agg['interest'] + agg['fees'] + agg['payments'] - agg['bad_principal']) /
               agg['total_variable_cost'].replace(0, np.nan)).fillna(0)
# Break-even metrics
agg['break_even_originations'] = (agg['total_variable_cost'] / agg['net_margin_per_loan'].
                                  replace(0, np.nan)).fillna(0)
agg['break_even_leads'] = (agg['break_even_originations'] / agg['conversion_rate'].
                           replace(0, np.nan)).fillna(0)
agg['break_even_conversion'] = (agg['break_even_leads'] / agg['leads']).fillna(0)

# -----------------------------
# Step 4.1: Default Rate Quartiles
# -----------------------------
# optional quartiling for categorical bins
# agg['default_rate_quartile'] = (
#     agg.groupby('tier')['default_rate']
#        .transform(lambda s: pd.qcut(s, 4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop'))
#        .fillna('Q0')
# )

# -----------------------------
# Step 4.2: ROI-based Optimal Volume Analysis with MOIC Floor
# -----------------------------
MOIC = 1.4  # Minimum acceptable ROI or money multiple
# calcs the required funding level and qty to achieve MOIC in each tier per month

opt_results = []
for tier, grp in agg.groupby('tier'):
    g = grp.dropna(subset=['ROI','conversion_trend', 'conversion_rate']).copy()
    if g['originations'].nunique() < 2:
        continue
    g['vol_decile'] = pd.qcut(g['originations'], 10, labels=False, duplicates='drop')
    dec_stats = g.groupby('vol_decile').agg(
        avg_ROI=('ROI','mean'),
        median_volume=('originations','median'),
        avg_net_bad=('net_bad_debt_per_loan','mean'),
        avg_conv_trend=('conversion_trend','mean'),
        avg_conv_rate=('conversion_rate', 'mean')
    )
    # Filter by MOIC
    dec_stats = dec_stats[dec_stats['avg_ROI'] >= MOIC]
    if dec_stats.empty:
        continue
    best = dec_stats.loc[dec_stats['avg_ROI'].idxmax()]
    opt_results.append({
        'tier': tier,
        'Optimal Loans': int(best['median_volume']),
        'ROI': best['avg_ROI'],
        'Avg Bad Debt/Loan': best['avg_net_bad'],
        'Avg Conv Trend': best['avg_conv_trend'],
        'Avg Conv Rate': best['avg_conv_rate']
    })

opt_df = pd.DataFrame(opt_results).set_index('tier')

#tools.display_dataframe_to_user('Optimal Volume by Cost Tier (ROI>=MOIC)', opt_df)

# --- Step 4.4 alt: calendar month MOIC Targets per Tier ---
# 4.4.1: Extract abbreviated calendar month and order it

agg['month_of_year'] = agg['month'].dt.month_name().str[:3]
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
agg['month_of_year'] = pd.Categorical(agg['month_of_year'], categories=month_order, ordered=True)

# 4.4.2: agg average cost & margin, and compute avg funded amount per loan
calendar = (
    agg.groupby(['month_of_year','tier'])
       .agg(
           avg_cost=('total_variable_cost','mean'),
           avg_margin=('net_margin_per_loan','mean'),
           total_funded=('funded_total','sum'),
           total_originations=('originations','sum')
       )
       .reset_index()
)
calendar['avg_funded_per_loan'] = calendar['total_funded'] / calendar['total_originations']

# 4.4.3 calc MOIC‐based targets
calendar['profit_target'] = calendar['avg_cost'] * (MOIC - 1)
calendar['loans_to_MOIC'] = (calendar['profit_target']) / calendar['avg_margin']
calendar['funding_to_MOIC'] = calendar['loans_to_MOIC'] * calendar['avg_funded_per_loan']

# 4.4.4 for df
opt_df_monthly = calendar[['month_of_year','tier','loans_to_MOIC','funding_to_MOIC']].rename(
    columns={
        'month_of_year':'Month',
        'tier':'Cost Tier',
        'loans_to_MOIC':'Loans to MOIC',
        'funding_to_MOIC':'Funding to MOIC ($)'
    }
)

# visualize loans and funding side-by-side
pivot_loans   = opt_df_monthly.pivot(
    index='Month',   columns='Cost Tier', values='Loans to MOIC').reindex(month_order)
pivot_funding = opt_df_monthly.pivot(
    index='Month',   columns='Cost Tier', values='Funding to MOIC ($)').reindex(month_order)

def add_line_labels(ax, fmt="{y:.0f}"):
    for line in ax.get_lines():
        xdata, ydata = line.get_xdata(), line.get_ydata()
        for x, y in zip(xdata, ydata):
            ax.text(x, y, fmt.format(x=x, y=y), ha='center', va='bottom', fontsize=7)

fig, ax1 = plt.subplots(figsize=(10,5))
for tier in tiers:
    ax1.plot(month_order, pivot_loans[tier], marker='o', label=tier)
add_line_labels(ax1, fmt="{y:.0f}")
ax1.set_title(f'Loans Required per Calendar Month to Achieve MOIC={MOIC}')
ax1.set_xlabel('Month')
ax1.set_ylabel('Loans to MOIC')
ax1.legend(title='Cost Tier', bbox_to_anchor=(1.05,1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/minimac/Downloads/data/calendar_month_loans_to_moic_NC_MCI.png')

fig, ax2 = plt.subplots(figsize=(10,5))
for tier in tiers:
    ax2.plot(month_order, pivot_funding[tier], marker='s', label=tier)
add_line_labels(ax2, fmt="${y:,.0f}")
ax2.set_title(f'Funding Required per Calendar Month to Achieve MOIC={MOIC}')
ax2.set_xlabel('Month')
ax2.set_ylabel('Funding to MOIC ($)')
ax2.legend(title='Cost Tier', bbox_to_anchor=(1.05,1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/minimac/Downloads/data/calendar_month_funding_to_moic_NC_MCI.png')


# -----------------------------
# Step 5: STL Decomposition of Lead Volumes
# -----------------------------
vol_df = agg.pivot(index='month', columns='tier', values='leads').fillna(0) # for tiers

# stl_series = {'Week': pd.date_range(start='2024-01-01', periods=73, freq='W'),
#                                 'Tier': df.lead_cost}


# STL using tiers
fig, axes = plt.subplots(len(tiers),1,figsize=(8,4*len(tiers)), sharex=True)
for ax, tier in zip(axes, tiers):
    ts = vol_df[tier]
    res = STL(ts, period=13, robust=True).fit()
    ax.plot(ts.index, ts, label='Original')
    ax.plot(res.trend.index, res.trend, label='Trend')
    ax.plot(res.seasonal.index, res.seasonal, label='Seasonal')
    ax.set_title(f'STL Decomposition – {tier}')
    ax.legend()
axes[-1].set_xlabel('Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/minimac/Downloads/data/leads_stl_decomposition_NC_MCI.png')

# # STL using lead costs by week aggregated
# stl_df = pd.DataFrame(stl_series)
# stl_df.set_index('Week', inplace=True)
#
# stl = STL(stl_series, period=52, seasonal=13)
# res = stl.fit()

# -----------------------------
# Step 6: Unified Monthly Break-even Conversion Goal
# -----------------------------
total = agg.groupby('month').agg(
    leads_total=('leads','sum'),
    be_leads_total=('break_even_leads','sum')
)
total['break_even_conv_all'] = (total['be_leads_total'] / total['leads_total']).fillna(0)
#tools.display_dataframe_to_user('Aggregated Break-even Conversion All Tiers',
# total['break_even_conv_all'].reset_index(name='Conversion'))
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(total.index, total['break_even_conv_all'], marker='o')
for x,y in zip(total.index, total['break_even_conv_all']): ax.text(x,y,f"{y:.2%}",ha='center',va='bottom',fontsize=7)
ax.set_title('Overall Monthly Break-even Conversion Goal')
ax.set_xlabel('Month'); ax.set_ylabel('Conversion Rate')
plt.xticks(rotation=45); plt.tight_layout(); plt.savefig('/Users/minimac/Downloads/data/overall_be_conversion_NC_MCI.png')

# -----------------------------
# Step 7: Histogram of Break-even Conversion
# -----------------------------
plt.figure(figsize=(8,4))
plt.hist(agg['break_even_conversion'], bins=30, edgecolor='k')
plt.title('Histogram of Break-even Conversion Rates')
plt.xlabel('Break-even Conversion'); plt.ylabel('Frequency')
plt.tight_layout(); plt.savefig('/Users/minimac/Downloads/data/break_even_conversion_histogram_NC_MCI.png')

# -----------------------------
# Step 8: Chart of Monthly Break-even Conversion by Tier
# -----------------------------
# Pivot by month-of-year to show seasonality
# may want to add data labels
agg['month_of_year'] = agg['month'].dt.month_name().str[:3]
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
agg['month_of_year'] = pd.Categorical(agg['month_of_year'], categories=month_order, ordered=True)
df_cal = agg.groupby(['month_of_year','tier'])['break_even_conversion'].mean().unstack().fillna(0)
fig, ax = plt.subplots(figsize=(10,6))
for tier in tiers:
    ax.plot(month_order, df_cal[tier].reindex(month_order), marker='o', label=tier)
ax.set_title('Avg Monthly Break-even Conversion by Tier')
ax.set_xlabel('Month'); ax.set_ylabel('Conversion Rate')
ax.legend(title='Tier', bbox_to_anchor=(1.05,1), loc='upper left')
plt.xticks(rotation=45); plt.tight_layout();
plt.savefig('/Users/minimac/Downloads/data/calendar_month_be_conversion_by_tier_NC_MCI.png')

#tools.display_dataframe_to_user('Calendar Month Avg Break-even Conversion by Tier',df_cal.reset_index())

# -----------------------------
# Step 9: Ridge Regression
# -----------------------------
# Aggregate deseasonalized deltas for input
ridge_results = []
alphas = np.logspace(-3, 3, 13)
for tier, grp in agg.groupby('tier'):
    g = grp.sort_values('month').copy()
    # Compute deltas on deseasonalized metrics
    g['d_originations'] = g['originations_safe'].diff()
    g['d_net_bad_debt'] = g['net_bad_debt_per_loan'].diff()
    g['d_default_rate'] = g['default_rate'].diff()
    g['d_lead_cost'] = g['lead_cost_per_loan'].diff()
    #g['d_uw_cost'] = g['uw_cost_per_loan'].diff()
    g['d_total_cost'] = g['cost_per_loan'].diff()
    g['d_net_margin'] = g['net_margin_per_loan'].diff()
    df_r = g.dropna(subset=['d_originations','d_net_bad_debt','d_default_rate',
                            'd_lead_cost', 'd_total_cost','d_net_margin']) # add 'd_uw_cost' back when available!
    if len(df_r) < 5:
        continue
    X = df_r[['d_originations','d_net_bad_debt','d_default_rate',
              'd_lead_cost','d_total_cost']].values # add 'd_uw_cost' back when available!
    y = df_r['d_net_margin'].values
    ridge = RidgeCV(alphas=alphas).fit(X, y)
    rec = {'tier': tier, 'alpha': ridge.alpha_, 'R2_ridge': ridge.score(X, y)}
    for name, coef in zip(['d_originations','d_net_bad_debt',
                           'd_default_rate','d_lead_cost','d_total_cost'], ridge.coef_):
        rec[f'ridge_coef_{name}'] = coef # add 'd_uw_cost' back when available!
    ridge_results.append(rec)
ridge_df = pd.DataFrame(ridge_results).set_index('tier')
# Display Ridge results
#tools.display_dataframe_to_user('Ridge Regression Summary by Tier', ridge_df)
# -----------------------------
# Step 11: Ridge Regression with Seasonal Spline Features
# -----------------------------
from sklearn.preprocessing import SplineTransformer
# Configure spline transformer
spline_transformer = SplineTransformer(degree=3, n_knots=5, include_bias=False)

spline_results = []
for tier, grp in agg.groupby('tier'):
    # Seasonal component on originations via STL
    ts = grp.sort_values('month').set_index('month')['originations']
    stl_res = STL(ts, period=12, robust=True).fit()
    seasonal = stl_res.seasonal.values.reshape(-1,1)
    # Create spline basis
    spline_feats = spline_transformer.fit_transform(seasonal)
    # Compute delta features as in Step 10
    g = grp.sort_values('month').copy()
    g['d_originations'] = g['originations_safe'].diff()
    g['d_net_bad_debt'] = g['net_bad_debt_per_loan'].diff()
    g['d_default_rate'] = g['default_rate'].diff()
    g['d_lead_cost'] = g['lead_cost_per_loan'].diff()
    #g['d_uw_cost'] = g['uw_cost_per_loan'].diff()
    g['d_total_cost'] = g['cost_per_loan'].diff()
    g['d_net_margin'] = g['net_margin_per_loan'].diff()
    df_r = g.dropna(subset=['d_originations','d_net_bad_debt','d_default_rate',
                            'd_lead_cost','d_total_cost','d_net_margin']) #add 'd_uw_cost' var when avail!
    if len(df_r) < 5:
        continue
    # Combine delta and spline features
    X_delta = df_r[['d_originations','d_net_bad_debt','d_default_rate',
                    'd_lead_cost','d_total_cost']].values # add 'd_uw_cost' back when available!
    spline_aligned = spline_feats[-len(df_r):]
    X = np.hstack([X_delta, spline_aligned])
    y = df_r['d_net_margin'].values
    # Fit Ridge with spline
    ridge_spline = RidgeCV(alphas=alphas).fit(X, y)
    rec = {'tier': tier, 'alpha_spline': ridge_spline.alpha_, 'R2_spline': ridge_spline.score(X, y)}
    # Record coefficients
    feat_names = ['d_originations','d_net_bad_debt','d_default_rate','d_lead_cost','d_total_cost'] \
                 + [f'spline_{i}' for i in range(spline_feats.shape[1])] # add 'd_uw_cost' back when available!
    for name, coef in zip(feat_names, ridge_spline.coef_):
        rec[f'ridge_coef_{name}'] = coef
    spline_results.append(rec)
# Create DataFrame
spline_df = pd.DataFrame(spline_results).set_index('tier')
# Display results
#tools.display_dataframe_to_user('Ridge with Seasonal Spline Features', spline_df)
# Plot coefficients
fig, ax = plt.subplots(figsize=(10,6))
for tier in spline_df.index:
    coefs = spline_df.loc[tier, [c for c in spline_df.columns if c.startswith('ridge_coef')]]
    ax.plot(coefs.index.str.replace('ridge_coef_',''), coefs.values, marker='o', label=tier)
ax.set_title('Ridge (with Seasonal Splines) Coefficients by Tier')
ax.set_xlabel('Feature')
ax.set_ylabel('Coefficient')
ax.legend(title='Tier', bbox_to_anchor=(1.05,1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ridge_spline_coefficients_by_tier.png')
# Plot Ridge Coefficients
fig, ax = plt.subplots(figsize=(8,4))
for tier in ridge_df.index:
    coefs = ridge_df.loc[tier, [c for c in ridge_df.columns if c.startswith('ridge_coef')]]
    ax.plot(coefs.index.str.replace('ridge_coef_',''), coefs.values, marker='o', label=tier)
ax.set_title('Ridge Regression Coefficients by Tier')
ax.set_xlabel('Predictor')
ax.set_ylabel('Coefficient')
ax.legend(title='Tier', bbox_to_anchor=(1.05,1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/minimac/Downloads/data/ridge_coefficients_by_tier_NC_MCI.png')



