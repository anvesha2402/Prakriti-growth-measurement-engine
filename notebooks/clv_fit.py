"""BG/NBD + Gamma-Gamma CLV by acquisition channel (PyMC-Marketing 1.1, MAP) with customer-level bootstrap intervals."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, time, sys
from pymc_marketing.clv import BetaGeoModel, GammaGammaModel
END = pd.Timestamp('2024-10-07') + pd.Timedelta(weeks=104)
def rfm():
    c = pd.read_csv('../data/stage2/customers.csv'); o = pd.read_csv('../data/stage2/orders.csv', parse_dates=['date'])
    g = o.groupby('customer_id').agg(first=('date','min'), last=('date','max'), n=('date','size'))
    rep = o.sort_values('date').groupby('customer_id').apply(lambda x: x.contribution.iloc[1:].mean() if len(x) > 1 else np.nan)
    d = c.set_index('customer_id').join(g).join(rep.rename('monetary_value'))
    d['frequency'] = d.n - 1; d['recency'] = (d['last'] - d['first']).dt.days / 7; d['T'] = (END - d['first']).dt.days / 7
    first_c = o.sort_values('date').groupby('customer_id').contribution.first()
    d['first_contribution'] = first_c
    return d.reset_index()
def fit_channel(d, horizon_months=(12, 24), disc_monthly=0.01, seed=0):
    x = d[['customer_id','frequency','recency','T']].copy(); x['customer_id'] = np.arange(len(x))
    bg = BetaGeoModel(); bg.fit(x, method='map', random_seed=seed)
    r = d[d.frequency > 0]; mg = pd.DataFrame({'customer_id': np.arange(len(r)), 'frequency': r.frequency.values, 'monetary_value': r.monetary_value.values})
    gg = GammaGammaModel(); gg.fit(mg, method='map', random_seed=seed)
    out = {}
    for h in horizon_months:
        clv = gg.expected_customer_lifetime_value(transaction_model=bg, data=x.assign(monetary_value=d.monetary_value.fillna(d.monetary_value.mean()).values),
                                                  future_t=h, discount_rate=disc_monthly, time_unit='W')
        out[h] = float(np.asarray(clv).mean())
    return out
if __name__ == "__main__":
    d = rfm(); ch = sys.argv[1] if len(sys.argv) > 1 else 'Trade_Promo'; t0 = time.time()
    print(ch, fit_channel(d[d.acquisition_channel == ch].reset_index(drop=True)), 'seconds', round(time.time() - t0))
