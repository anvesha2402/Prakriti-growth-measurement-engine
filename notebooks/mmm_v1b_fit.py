"""MMM v1b (national): fixes for v1 failures. Run: python mmm_v1b_fit.py  (about 25 min on 1 CPU)"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, time, os, sys
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior
sys.path.insert(0,'.'); from mmm_v1_fit import load, CH
CTRL=['price','temp','festive','comp','promo','t']          # dark-store count dropped: corr 0.998 with trend nationally
SIG_BETA=0.15
def prep():
    nat=load()
    tr=nat.iloc[:79]
    mu,sd=tr[CTRL].mean(),tr[CTRL].std()                      # standardise controls with TRAIN stats only
    nat[CTRL]=(nat[CTRL]-mu)/sd
    return nat
def make_model():
    cfg={"intercept":Prior("HalfNormal",sigma=0.5),
         "saturation_beta":Prior("HalfNormal",sigma=SIG_BETA,dims="channel"),
         "gamma_control":Prior("Normal",mu=0,sigma=0.3,dims="control")}
    return MMM(date_column='date',channel_columns=CH,target_column='y',
               adstock=GeometricAdstock(l_max=26),saturation=HillSaturation(),
               control_columns=CTRL,yearly_seasonality=2,model_config=cfg)
if __name__=="__main__":
    nat=prep(); tr=nat.iloc[:79]; m=make_model(); t0=time.time()
    if '--prior' in sys.argv:
        m.build_model(tr.drop(columns='y'),tr['y'])
        import pymc as pm
        with m.model: pr=pm.sample_prior_predictive(300,random_seed=1)
        cc=pr.prior['channel_contribution'].values*float(m.model['target_scale'].eval() if hasattr(m.model['target_scale'],'eval') else 293.5)
        share=cc.sum(axis=(2,3)).reshape(-1)/tr.y.sum()
        print('PRIOR media share of revenue: median %.2f, p5 %.2f, p95 %.2f'%(np.median(share),*np.percentile(share,[5,95]))); sys.exit()
    m.fit(tr.drop(columns='y'),tr['y'],chains=4,draws=1000,tune=1000,target_accept=0.97,cores=1,random_seed=42,progressbar=False)
    os.makedirs('../model',exist_ok=True); m.idata.to_netcdf('../model/mmm_v1b_national.nc'); print('trace saved',round(time.time()-t0),'s',flush=True)
    full=nat.iloc[:87]
    pp=m.sample_posterior_predictive(full.drop(columns='y'),extend_idata=False,combined=False,random_seed=42)
    pp.to_netcdf('../model/mmm_v1b_pred_wk1_87.nc'); print('done',round(time.time()-t0),'s',flush=True)
