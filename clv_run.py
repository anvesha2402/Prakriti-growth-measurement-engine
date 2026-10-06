import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json, time
from clv_fit import rfm, fit_channel
d = rfm(); B = 10; rows = []; rng = np.random.default_rng(11); t0 = time.time()
for ch in ['Influencers','YouTube','Organic','Google_Search','Meta','Trade_Promo']:
    g = d[d.acquisition_channel == ch].reset_index(drop=True)
    pt = fit_channel(g)
    bs = []
    for b in range(B):
        idx = rng.integers(0, len(g), len(g))
        try: bs.append(fit_channel(g.iloc[idx].reset_index(drop=True), seed=b))
        except Exception as e: print('bootstrap fail', ch, b, str(e)[:80])
    rows.append({'channel': ch, 'customers': len(g), 'clv12': pt[12], 'clv24': pt[24],
                 'clv12_p5': np.percentile([x[12] for x in bs], 5), 'clv12_p95': np.percentile([x[12] for x in bs], 95),
                 'clv24_p5': np.percentile([x[24] for x in bs], 5), 'clv24_p95': np.percentile([x[24] for x in bs], 95), 'n_boot': len(bs)})
    pd.DataFrame(rows).to_csv('../model/clv_by_channel.csv', index=False); print(ch, 'done', round(time.time() - t0), 's', flush=True)
