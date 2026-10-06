"""MMM v2 (national, all 104 weeks) = v1b specification + the Meta geo-experiment as a lift-test calibration.
Run: python mmm_v2_fit.py      (smoke test: python mmm_v2_fit.py --smoke)"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, time, os, sys, json
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_extras.prior import Prior
CH=['Meta','Google_Search','YouTube','QC_Ads','Influencers','Trade_Promo']
CTRL=['price','temp','festive','comp','promo','t']
TG=['Ahmedabad','Pune']
def load():
    w=pd.read_csv('../data/stage2/weekly_geo.csv',parse_dates=['week_start'])
    nat=w.groupby('week').agg(date=('week_start','first'),y=('revenue_lakh','sum'),price=('price_index','mean'),ds=('dark_store_count','sum'),temp=('temp_c','mean'),festive=('festive_flag','first'),comp=('competitor_index','first'),promo=('promo_flag','first'),**{c:('spend_'+c+'_lakh','sum') for c in CH}).reset_index(drop=True)
    nat['t']=np.arange(1,len(nat)+1)
    nat[CTRL]=(nat[CTRL]-nat[CTRL].mean())/nat[CTRL].std()
    return nat,w
def lift_test(nat,w):
    """Pre-registered readout (design.md): DiD on log revenue, interval half-width 0.0404 (notebook 03). Rebuilt from data."""
    R=w.pivot(index='week',columns='geo',values='revenue_lakh'); MS=w.pivot(index='week',columns='geo',values='spend_Meta_lakh')
    CG=[g for g in R.columns if g not in TG]
    lr=np.log(R[TG].sum(axis=1))-np.log(R[CG].sum(axis=1)); tau=lr.loc[88:95].mean()-lr.loc[36:87].mean(); c90=0.0404
    foregone_wk=MS.loc[80:87,TG].sum(axis=1).mean(); T=R.loc[88:95,TG].sum(axis=1).mean()   # per week
    lost=lambda t: T*np.exp(-t)-T
    d=dict(tau=float(tau),delta_y_wk=float(lost(tau)),delta_y_lo_wk=float(lost(tau+c90)),delta_y_hi_wk=float(lost(tau-c90)),
           delta_x_wk=float(foregone_wk),x_national_wk=float(MS.loc[80:87].sum(axis=1).mean()))
    d['sigma']=float((d['delta_y_hi_wk']-d['delta_y_lo_wk'])/(2*1.645))
    return d
def make_model():
    cfg={"intercept":Prior("HalfNormal",sigma=0.5),
         "saturation_beta":Prior("HalfNormal",sigma=0.15,dims="channel"),
         "gamma_control":Prior("Normal",mu=0,sigma=0.3,dims="control")}
    return MMM(date_column='date',channel_columns=CH,target_column='y',adstock=GeometricAdstock(l_max=26),
               saturation=HillSaturation(),control_columns=CTRL,yearly_seasonality=2,model_config=cfg)
if __name__=="__main__":
    nat,w=load(); X=nat.drop(columns='y'); y=nat['y']
    L=lift_test(nat,w); os.makedirs('../model',exist_ok=True); json.dump(L,open('../model/lift_test.json','w'),indent=2)
    df=pd.DataFrame([{'channel':'Meta','x':L['x_national_wk'],'delta_x':-L['delta_x_wk'],'delta_y':-L['delta_y_wk'],'sigma':L['sigma']}])
    m=make_model(); m.build_model(X,y); m.add_lift_test_measurements(df)
    smoke='--smoke' in sys.argv; t0=time.time()
    m.fit(X,y,chains=2 if smoke else 4,draws=100 if smoke else 1000,tune=100 if smoke else 1000,target_accept=0.97,cores=1,random_seed=42,progressbar=False) if False else None
    import pymc as pm
    with m.model:
        idata=pm.sample(draws=100 if smoke else 1000,tune=100 if smoke else 1000,chains=2 if smoke else 4,target_accept=0.97,cores=1,random_seed=42,progressbar=False)
    m.idata=idata
    print('sampled',round(time.time()-t0),'s',flush=True)
    if smoke: print('lift variable present:',[v for v in m.model.named_vars if 'lift' in v]); sys.exit()
    idata.to_netcdf('../model/mmm_v2_national.nc'); print('trace saved',flush=True)
    pp=m.sample_posterior_predictive(X,extend_idata=False,combined=False,random_seed=42); pp.to_netcdf('../model/mmm_v2_pred.nc'); print('done',round(time.time()-t0),'s',flush=True)
