#!/usr/bin/env python3
"""Compare generated ratios with the benchmark targets. Reads ONLY the data files (never truth.json)."""
import sys, numpy as np, pandas as pd
d = sys.argv[1]
w = pd.read_csv(f"{d}/weekly_geo.csv"); p = pd.read_csv(f"{d}/platform_reported.csv")
c = pd.read_csv(f"{d}/customers.csv", parse_dates=["first_order_date"])
o = pd.read_csv(f"{d}/orders.csv", parse_dates=["date"])
CH = ["Meta", "Google_Search", "YouTube", "QC_Ads", "Influencers", "Trade_Promo"]
sc = ["spend_" + k + "_lakh" for k in CH]
maxw = w.week.max()
last = w[w.week > maxw - 52]
rev = last.revenue_lakh.sum() / 100
spend = last[sc].sum()
print(f"weeks available: {maxw}")
print(f"trailing-52wk revenue: Rs {rev:.1f} Cr   (target ~120 at full 104 weeks)")
if maxw == 104:
    print(f"YoY growth: {w[w.week>52].revenue_lakh.sum()/w[w.week<=52].revenue_lakh.sum()-1:.1%}  (target ~45%)")
print(f"paid spend trailing-52wk: Rs {spend.sum()/100:.1f} Cr   (target ~34)")
print(f"marketing % of revenue: {spend.sum()/100/rev:.1%}   (target ~28%)")
print("spend shares:", (spend / spend.sum()).round(3).rename(lambda s: s[6:-5]).to_dict())
print(f"QC ad spend as % of assumed QC sales (45% of revenue): {spend.iloc[3]/100/(0.45*rev):.1%}  (benchmark: up to 15%)")
nat = w.groupby("week")[sc].sum()
win = nat.loc[20:49]
print(f"Meta-YouTube spend corr, wks 20-49: {np.corrcoef(win.iloc[:,0], win.iloc[:,2])[0,1]:.2f}  (target ~0.8)")
print(f"Meta-YouTube spend corr, other wks: {np.corrcoef(nat.drop(range(20,50)).iloc[:,0], nat.drop(range(20,50)).iloc[:,2])[0,1]:.2f}")
wk = w.groupby("week").revenue_lakh.sum()
b1, b0 = np.polyfit(nat.index, np.log(wk.values), 1)
rv = wk / np.exp(b0 + b1 * nat.index)
mon = pd.to_datetime(w.drop_duplicates("week").set_index("week").week_start).dt.month
print("seasonality (detrended revenue vs overall mean): summer Apr-Jun %+.0f%%, festive Oct-Nov %+.0f%%, monsoon Jul-Sep %+.0f%%" % tuple(
    100 * (rv[mon.isin(m)].mean() / rv.mean() - 1) for m in ([4, 5, 6], [10, 11], [7, 8, 9])))
web = last[[sc[i] for i in [0, 1, 2, 4, 5]]].sum().sum() * 1e5
print(f"web-facing paid spend / new customers (CAC proxy): Rs {web/last.new_customers.sum():,.0f}  (benchmark 800-1,200, low confidence)")
print(f"ledger: {len(o):,} orders, {len(c):,} customers (target 70-90k orders, 25-35k customers); mean gross value Rs {o.gross_value.mean():.0f} (target 600-700)")
oc = o.merge(c[["customer_id", "acquisition_channel", "first_order_date"]], on="customer_id")
cut = o.date.max() - pd.Timedelta(days=90)
elig = c[c.first_order_date <= cut]
first = oc.groupby("customer_id").date.min().rename("fd")
oc = oc.join(first, on="customer_id")
rep = oc[(oc.date > oc.fd) & (oc.date <= oc.fd + pd.Timedelta(days=90))].customer_id.unique()
elig = elig.assign(rep=elig.customer_id.isin(rep))
print("90-day repeat rate by channel:", elig.groupby("acquisition_channel").rep.mean().round(3).to_dict())
tot_rev = w.revenue_lakh.sum()
plat = p.groupby("channel").agg(spend=("spend_lakh", "sum"), rev=("platform_revenue_lakh", "sum"))
print(f"platform-claimed revenue / actual revenue: {plat.rev.sum()/tot_rev:.2f}x   (brief's example: 1.6x)")
print("platform ROAS by channel:", (plat.rev / plat.spend).round(2).to_dict())
