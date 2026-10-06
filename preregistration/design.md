# Pre-registration: Meta geo-holdout experiment (Prakriti Foods, fictional)
Written before stage 2 (weeks 88-104) was generated. Based only on stage-1 data (weeks 1-87). The sealed truth has not been opened.
*Prakriti Foods is a fictional company; all data is synthetic.*

## Question
Is Meta advertising incremental, and by how much? (Platform dashboards report ROAS of about 3x.)

## Design
- Treatment: Meta spend set to zero in **Ahmedabad and Pune**, weeks **88-95** (8 weeks).
- Control: Bengaluru, Chennai, Delhi NCR, Hyderabad, Kolkata, Mumbai (Meta unchanged).
- Pre-period: weeks 36-87. Washout observed: weeks 96-104.
- Geo selection rule (fixed in advance): lowest pre-period tracking error of log(treated/control) around its linear trend; ties broken by lowest share of revenue at risk. Ahmedabad + Pune and Delhi NCR + Mumbai tied at 0.060; Ahmedabad + Pune chosen (14% vs 38% of revenue).
- No other channel's spend plan is changed in the treated cities.

## Hypotheses
- H0: pausing Meta in treated cities does not change their revenue relative to control cities.
- H1: pausing Meta lowers treated-city revenue relative to control.

## Metric and estimators
- Primary metric: weekly revenue, log scale, treated cities combined vs control cities combined.
- Primary estimator: difference-in-differences, tau = mean(log T/C, weeks 88-95) - mean(log T/C, weeks 36-87).
- Secondary estimator: synthetic control (non-negative weights summing to 1 on the six control cities, fitted on weeks 36-87 only).
- Inference: 90% interval from the pre-period placebo windows and the simulated AR(1) null; also report 95%.
- Placebo check: the same DiD on three pre-period windows with no pause.

## Power (computed on weeks 36-87)
- Minimum detectable drop in treated-city revenue at 80% power, 5% two-sided: about 7% (simulation 7%, empirical windows 6%).
- MMM v1b implies Meta is worth about 10% of revenue (90% interval 0-30%). If the true effect is below about 7%, the test is likely to return "not detected".

## Outputs and decision rule
1. Implied Meta ROI = (counterfactual revenue - actual revenue in treated cities, weeks 88-95) / (Meta spend foregone). Foregone spend = mean weekly Meta spend in treated cities over weeks 80-87 x 8.
2. If the 90% interval for tau excludes zero: use the implied ROI and its interval as a lift-test calibration input for MMM v2.
3. If it includes zero: report "no effect detected above the MDE" and use the interval's upper end as a cap on Meta's ROI prior in MMM v2.
4. If DiD and synthetic control disagree: report both, calibrate with the more conservative.

## Commitments
- No change to the analysis after seeing stage 2. Any deviation is reported alongside the pre-registered result, with the reason.
- A competitor-shock week inside the window affects all cities alike and is handled by the DiD; no weeks are dropped.
- Known limitations: only two treated cities (both newer geos); possible ad-delivery spillover across cities; the carry-over of Meta lasts 1-2 weeks, so the first test weeks mix pause and decay.
