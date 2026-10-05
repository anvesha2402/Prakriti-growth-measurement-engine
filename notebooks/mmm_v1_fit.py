"""MMM v1 (national, weeks 1-79 train; 80-87 holdout). PyMC-Marketing 1.1. Run: python mmm_v1_fit.py"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, time, os
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
CH=['Meta','Google_Search','YouTube','QC_Ads','Influencers','Trade_Promo']
CTRL=['price','ds','temp','festive','comp','promo','t']
def load():
    w=pd.read_csv('../data/generated/weekly_geo.csv',parse_dates=['week_start'])
    nat=w.groupby('week').agg(date=('week_start','first'),y=('revenue_lakh','sum'),price=('price_index','mean'),ds=('dark_store_count','sum'),temp=('temp_c','mean'),festive=('festive_flag','first'),comp=('competitor_index','first'),promo=('promo_flag','first'),**{c:('spend_'+c+'_lakh','sum') for c in CH}).reset_index(drop=True)
    nat['t']=np.arange(1,len(nat)+1)
    return nat
def make_model():
    return MMM(date_column='date',channel_columns=CH,target_column='y',
               adstock=GeometricAdstock(l_max=26), saturation=HillSaturation(),   # named explicitly (brief 6.4)
               control_columns=CTRL, yearly_seasonality=2)
if __name__=="__main__":
    nat=load(); tr=nat.iloc[:79]
    m=make_model(); t0=time.time()
    m.fit(tr.drop(columns='y'),tr['y'],chains=4,draws=1000,tune=1000,target_accept=0.9,cores=1,random_seed=42,progressbar=False)
    os.makedirs('../model',exist_ok=True); m.idata.to_netcdf('../model/mmm_v1_national_raw.nc'); print('raw trace saved',round(time.time()-t0),'s',flush=True)
    m.sample_posterior_predictive(tr.drop(columns='y'),extend_idata=True,combined=False,random_seed=42)
    m.add_original_scale_contribution_variable(var=["channel_contribution"])
    os.makedirs('../model',exist_ok=True)
    m.idata.to_netcdf('../model/mmm_v1_national.nc')
    # holdout: predict weeks 1-87 so adstock has its history, keep the last 8 weeks
    full=nat.iloc[:87]
    pp=m.sample_posterior_predictive(full.drop(columns='y'),extend_idata=False,combined=False,random_seed=42)
    pp.to_netcdf('../model/mmm_v1_national_pred_wk1_87.nc')
    print('done in',round(time.time()-t0),'s')
