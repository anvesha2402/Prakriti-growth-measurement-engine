# Data card (draft v0.1) - Prakriti Foods synthetic data
Purpose: validate a marketing-measurement method against a planted truth. Fictional brand; synthetic data.
Process: 8 geos x 104 weeks. Baseline demand (trend, season, temperature, price, dark stores, competitor shocks) plus six channels, each with geometric adstock (l_max 26) and Hill saturation; betas solved so each channel's drawn ROI holds over the full period; multiplicative log-normal noise; 5% measurement error on reported spend.
Known unrealism: no competitor reactions, constant creative quality, geo-level spend split is a noisy share of national spend, platform-reported revenue = true incremental x drawn multiple (+noise).
Units: Rs lakh unless a column says otherwise.
Ledger: customers.csv/orders.csv are a ~28k-customer extract of own-website customers. new_customers in weekly_geo.csv is the full website population (the ledger is a ~10% sample).
Regenerate: python notebooks/generate_data.py --seed-file <vault>/seed.txt --stage 1 --out data/generated --vault <vault>
## Calibration (stage-1 data, weeks 1-87) - see data/calibration_stage1.txt
Gaps to document: marketing share, platform over-claim ratio, ledger size (stage 1 holds weeks 1-87 only; full 104 weeks will add orders).
