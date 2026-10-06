#!/usr/bin/env python3
"""
Prakriti Foods - synthetic data generator (Capstone 3).
Prakriti Foods is fictional; all data is synthetic, calibrated to public benchmarks.

Usage
  dev run   : python generate_data.py --seed 7 --stage 2 --out dev_out --vault dev_vault
  final run : python generate_data.py --seed-file vault/seed.txt --stage 1 --out data/generated --vault vault
  stage 2   : python generate_data.py --seed-file vault/seed.txt --stage 2 --design preregistration/design.json ...

Units: money in Rs lakh (1 Cr = 100 lakh) unless a column says otherwise.
The true parameters are written to <vault>/truth.json and are NEVER printed.
"""
import argparse, datetime as dt, hashlib, json, os
import numpy as np, pandas as pd

CH = ["Meta", "Google_Search", "YouTube", "QC_Ads", "Influencers", "Trade_Promo"]
ACQ = ["Meta", "Google_Search", "YouTube", "Influencers", "Trade_Promo", "Organic"]
GEOS = ["Delhi NCR", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]
GW = np.array([.20, .18, .16, .12, .10, .10, .08, .06])
SHARE = np.array([.25, .15, .12, .25, .10, .13])
NEWGEO = [3, 6, 7]            # Hyderabad, Pune, Ahmedabad: influencers stronger
NW = 104
LMAX = 26
AOV = 650.0
START = dt.date(2024, 10, 7)  # Monday of week 1

RANGES = {
    "lam": [(.5, .7), (.1, .3), (.85, .9), (.1, .3), (.4, .6), (0., .2)],
    "K":   [(.6, .9), (1., 1.5), (1.5, 2.5), (.8, 1.2), (1.2, 2.), (1., 1.5)],
    "s":   [(1., 2.)] * 6,
    "roi": [(1.2, 1.8), (.8, 1.4), (1., 2.), (1.2, 2.), (1.3, 2.2), (.6, 1.)],
}
MULT = [(1.8, 2.2), (2.5, 3.5), (.6, .9), None, (.5, 1.), None]
REPEAT90 = {"Meta": (.30, .35), "Google_Search": (.30, .35), "YouTube": (.35, .40),
            "Influencers": (.40, .45), "Trade_Promo": (.20, .25), "Organic": (.40, .40)}


def draw_truth(r):
    T = {}
    for k in ["lam", "K", "s", "roi"]:
        T[k] = [float(r.uniform(*RANGES[k][i])) for i in range(6)]
    cann = float(r.uniform(.5, .7))
    T["cannibalisation_share"] = cann
    mult = []
    for i, rg in enumerate(MULT):
        if i == 3:
            mult.append(float(np.clip((1 / (1 - cann)) * r.uniform(.95, 1.05), 2.0, 3.5)))
        elif rg is None:
            mult.append(None)
        else:
            mult.append(float(r.uniform(*rg)))
    T["platform_multiple"] = mult
    T["baseline_share"] = float(r.uniform(.55, .65))
    T["season"] = {"summer": float(r.uniform(.15, .25)), "festive": float(r.uniform(.08, .12)),
                   "monsoon": float(r.uniform(.05, .10))}
    T["price_elasticity"] = float(r.uniform(-1.5, -.8))
    T["temp_coef"] = float(r.uniform(.10, .20))
    T["ds_growth"] = float(r.uniform(.25, .35))
    T["comp_weeks"] = sorted(int(x) for x in r.choice(np.arange(5, 100), 2, replace=False))
    T["comp_shock"] = [float(r.uniform(.08, .12)) for _ in range(2)]
    T["sigma_geo"] = [float(r.uniform(.06, .10)) for _ in range(8)]
    T["pullforward"] = float(r.uniform(.2, .4))
    T["new_geo_boost"] = float(r.uniform(1.2, 1.5))
    T["cro_relative_lift"] = float(r.uniform(.05, .08))
    clv = {}
    for a in ACQ:
        tgt = float(r.uniform(*REPEAT90[a])) if REPEAT90[a][0] != REPEAT90[a][1] else REPEAT90[a][0]
        pbar = 0.40 if a == "Trade_Promo" else 0.30
        shape = 0.8
        x = 1 - tgt / (1 - pbar)
        u = x ** (1 / shape)
        alpha = 13 * u / (1 - u)
        clv[a] = {"repeat90_target": tgt, "p_mean": pbar, "r": shape, "alpha": float(alpha),
                  "aov_mult": float(r.uniform(.9, 1.1))}
    T["clv"] = clv
    T["cm_rate"] = float(r.uniform(.35, .45))
    return T


def adstock(x, lam):
    A = np.zeros_like(x)
    for i in range(x.shape[1]):
        prev = A[:, i - 1] if i > 0 else 0.0
        A[:, i] = x[:, i] + lam * prev
        if i >= LMAX:
            A[:, i] -= lam ** LMAX * x[:, i - LMAX]
    return A


def response(spend, T, ref):
    f = np.zeros_like(spend)
    for c in range(6):
        A = adstock(spend[c], T["lam"][c])
        Ah = A / ref[c][:, None]
        s, K = T["s"][c], T["K"][c]
        f[c] = Ah ** s / (Ah ** s + K ** s) * GW[:, None]
        if c == 4:
            f[c][NEWGEO] *= T["new_geo_boost"]
        if c == 5:
            g = f[c].copy()
            p1 = np.concatenate([np.zeros((8, 1)), g[:, :-1]], 1)
            p2 = np.concatenate([np.zeros((8, 2)), g[:, :-2]], 1)
            f[c] = g - T["pullforward"] * (p1 + p2) / 2
    return f


def build(seed, stage, design):
    ss = np.random.SeedSequence(seed)
    rt, re_, rs, rn, rc, rp, ri, rx, rb = [np.random.default_rng(s) for s in ss.spawn(9)]
    T = draw_truth(rt)
    t = np.arange(1, NW + 1)
    dates = [START + dt.timedelta(weeks=int(i - 1)) for i in t]
    mid = [d + dt.timedelta(days=3) for d in dates]
    month = np.array([d.month for d in mid])
    doy = np.array([d.timetuple().tm_yday for d in mid])
    trend = 1.40 ** (t / 52)
    se = T["season"]
    season = np.ones(NW)
    season[np.isin(month, [4, 5, 6])] += se["summer"]
    season[np.isin(month, [10, 11])] += se["festive"]
    season[np.isin(month, [7, 8, 9])] -= se["monsoon"]
    summer = np.isin(month, [4, 5, 6]); fest = np.isin(month, [10, 11])
    base_t = np.array([25, 28, 24, 27, 29, 27, 25, 28], float)
    temp = base_t[:, None] + 8 * np.sin(2 * np.pi * (doy - 60) / 365)[None, :] + re_.normal(0, 1.5, (8, NW))

    # ---- spend plans (national) ----
    S = trend * (3400 / trend[52:].sum())
    tilt = np.ones((6, NW))
    tilt[0] += .10 * (summer | fest); tilt[3] += .15 * (summer | fest); tilt[5] += .15 * fest
    plan = SHARE[:, None] * S[None, :] * tilt
    noise = rs.normal(0, .2, (6, NW))
    win = (t >= 20) & (t <= 49)
    common = rs.normal(0, .50, NW); idio = rs.normal(0, .15, (2, NW)); indep = rs.normal(0, .25, (2, NW))
    noise[0] = np.where(win, common + idio[0], indep[0])
    noise[2] = np.where(win, common + idio[1], indep[1])
    x = plan * np.exp(noise)
    x *= 3400 / x[:, 52:].sum()
    gs = GW[None, :, None] * np.exp(rs.normal(0, .10, (6, 8, NW)))
    gs /= gs.sum(1, keepdims=True)
    spend = x[:, None, :] * gs                                   # c,g,t  (true billed spend)
    promo_nat = (x[5] > np.quantile(x[5], .85)).astype(int)

    # ---- controls ----
    rw = np.cumsum(re_.normal(0, .004, NW)); rw -= rw.mean()
    price_nat = 1 + np.clip(rw, -.04, .04) - .08 * promo_nat
    price = price_nat[None, :] * (1 + re_.normal(0, .01, (8, NW)))
    ds0 = re_.uniform(35, 45, 8) * GW * 8
    ds = ds0[:, None] * (1 + T["ds_growth"]) ** (t / 52)[None, :]
    shock = np.zeros(NW)
    for w, sh in zip(T["comp_weeks"], T["comp_shock"]):
        shock[w - 1] = sh
    Bu = (GW[:, None] * trend[None, :] * season[None, :] * (1 + T["temp_coef"] * (temp - 27) / 10)
          * price ** T["price_elasticity"] * (ds / ds[:, [0]]) ** .3 * (1 - shock)[None, :])

    # ---- calibrate betas so drawn ROI holds over the full period ----
    ref = np.stack([adstock(spend[c], T["lam"][c]).mean(1) for c in range(6)])
    fb = response(spend, T, ref)
    beta = np.array([T["roi"][c] * spend[c].sum() / fb[c].sum() for c in range(6)])
    M = sum(T["roi"][c] * spend[c].sum() for c in range(6))
    B = Bu * (M * T["baseline_share"] / (1 - T["baseline_share"]) / Bu.sum())

    # ---- experiment design (stage 2) ----
    sp = spend.copy()
    if design:
        w0, w1 = design["weeks"]
        for g in design["treated_geos"]:
            sp[0, GEOS.index(g), w0 - 1:w1] = 0.0
    f = response(sp, T, ref)
    incr = beta[:, None, None] * f                               # c,g,t true incremental revenue
    incr_np = beta[:, None, None] * fb                           # same, with NO pause (counterfactual)
    eps = rn.normal(0, 1, (8, NW)) * np.array(T["sigma_geo"])[:, None]
    y = (B + incr.sum(0)) * np.exp(eps)                          # g,t revenue (lakh)
    y_np = (B + incr_np.sum(0)) * np.exp(eps)
    meas = np.exp(rn.normal(0, .05, (6, 8, NW)))
    spend_obs = sp * meas

    tend = 87 if stage == 1 else 104
    # ---- weekly_geo ----
    rows = []
    newc = np.round(y * 1e5 * .20 / AOV * .60).astype(int)
    for g in range(8):
        for i in range(tend):
            d = {"week": i + 1, "week_start": str(dates[i]), "geo": GEOS[g],
                 "revenue_lakh": round(float(y[g, i]), 3),
                 "orders": int(round(y[g, i] * 1e5 / AOV)), "new_customers": int(newc[g, i])}
            for c in range(6):
                d["spend_" + CH[c] + "_lakh"] = round(float(spend_obs[c, g, i]), 3)
            d.update({"price_index": round(float(price[g, i]), 4), "promo_flag": int(promo_nat[i]),
                      "dark_store_count": int(round(ds[g, i])), "temp_c": round(float(temp[g, i]), 2),
                      "festive_flag": int(fest[i]), "competitor_index": int(shock[i] > 0)})
            rows.append(d)
    weekly = pd.DataFrame(rows)

    # ---- platform_reported ----
    rows = []
    for c in [0, 1, 2, 3, 4]:
        inc_n = incr[c].sum(0)
        rep = T["platform_multiple"][c] * inc_n * np.exp(rp.normal(0, .10, NW))
        sp_n = sp[c].sum(0)
        nz = np.concatenate([rp.normal(0, .08, 87), rb.normal(0, .08, NW - 87)])
        cpm = [160, 0, 110, 0, 250][c]; ctr = [.011, .045, .004, .018, .008][c]; cpc = [0, 14, 0, 9, 0][c]
        for i in range(tend):
            rs_ = sp_n[i] * 1e5
            imp = rs_ / cpm * 1000 if cpm else rs_ / (cpc * ctr)
            imp *= float(np.exp(nz[i]))
            clicks = imp * ctr
            conv = rep[i] * 1e5 / AOV
            rows.append({"week": i + 1, "channel": CH[c], "spend_lakh": round(float(sp_n[i]), 3),
                         "impressions": int(imp), "clicks": int(clicks), "ctr": round(ctr, 4),
                         "cpc_rs": round(rs_ / clicks, 2), "cpm_rs": round(rs_ / imp * 1000, 2),
                         "platform_conversions": int(conv),
                         "platform_revenue_lakh": round(float(rep[i]), 3),
                         "platform_roas": round(float(rep[i] / sp_n[i]), 3)})
    platform = pd.DataFrame(rows)

    # ---- customers & orders (website ledger extract) ----
    def sim_ledger(rng, newc_arr, fs, incr_arr, id_start):
        n_s = rng.binomial(newc_arr, fs)
        wts = np.stack([incr_arr[0], incr_arr[1], incr_arr[2], incr_arr[4], np.maximum(incr_arr[5], 0), B], 0)
        pr = wts / wts.sum(0)
        cust = []
        for g in range(8):
            for i in range(NW):
                n = n_s[g, i]
                if n:
                    ch = rng.choice(6, n, p=pr[:, g, i])
                    for k in range(n):
                        cust.append((g, i, ch[k]))
        cg = np.array([a_ for a_, _, _ in cust], int); cw = np.array([b_ for _, b_, _ in cust], int)
        cc = np.array([c_ for _, _, c_ in cust], int)
        t0 = cw + rng.random(len(cw))
        ordl = []
        for ci, a_ in enumerate(ACQ):
            idx = np.where(cc == ci)[0]
            n = len(idx)
            if n == 0:
                continue
            P = T["clv"][a_]
            lam = rng.gamma(P["r"], 1 / P["alpha"], n)
            p = rng.beta(P["p_mean"] * 4, (1 - P["p_mean"]) * 4, n)
            cur = t0[idx].copy()
            alive = rng.random(n) > p
            ordl.append((idx, cur.copy(), np.ones(n, bool)))
            for _ in range(80):
                cur = cur + rng.exponential(1 / lam)
                ok = alive & (cur < NW)
                if ok.any():
                    ordl.append((idx[ok], cur[ok], np.zeros(ok.sum(), bool)))
                alive = ok & (rng.random(n) > p)
                if not alive.any():
                    break
        cid = np.array(["C%06d" % (id_start + i + 1) for i in range(len(cg))])
        oi = np.concatenate([a_ for a_, _, _ in ordl]); ot = np.concatenate([b_ for _, b_, _ in ordl])
        ofirst = np.concatenate([c_ for _, _, c_ in ordl])
        o_ch = cc[oi]
        mult = np.array([T["clv"][ACQ[k]]["aov_mult"] for k in o_ch])
        gross = AOV * mult * np.exp(rng.normal(0, .3, len(oi)) - .045)
        trade_first = (o_ch == 4) & ofirst
        disc = np.where(trade_first, rng.uniform(.15, .25, len(oi)), np.where(rng.random(len(oi)) < .3, .05, 0)) * gross
        cm = np.clip(T["cm_rate"] + rng.normal(0, .02, len(oi)), .3, .5)
        cdf = pd.DataFrame({"customer_id": cid,
                            "first_order_date": [START + dt.timedelta(days=int(v * 7)) for v in t0],
                            "acquisition_channel": [ACQ[i] for i in cc], "geo": [GEOS[i] for i in cg],
                            "acquisition_week": cw + 1})
        odf = pd.DataFrame({"customer_id": cid[oi],
                            "date": [START + dt.timedelta(days=int(v * 7)) for v in ot],
                            "gross_value": gross.round(2), "discount": disc.round(2),
                            "contribution": (cm * (gross - disc)).round(2), "_t": ot})
        return cdf, odf

    newc_np = np.round(y_np * 1e5 * .20 / AOV * .60).astype(int)
    fs = 28000 / newc_np.sum()
    # Part A: customers acquired in weeks 1-87 come from the NO-PAUSE simulation (repeat buying does not depend on ad spend),
    # so stage-1 and stage-2 ledgers agree exactly on weeks 1-87.
    cdfA, odfA = sim_ledger(rc, newc_np, fs, incr_np, 0)
    nA = int(cdfA.customer_id.str[1:].astype(int).max())
    cdfA = cdfA[cdfA.acquisition_week <= 87]
    odfA = odfA[odfA.customer_id.isin(set(cdfA.customer_id))]
    if tend > 87:
        # Part B: customers acquired in weeks 88+ come from the actual (paused) scenario, separate random stream
        nb = newc.copy(); nb[:, :87] = 0
        cdfB, odfB = sim_ledger(rb, nb, fs, incr, nA)
        cdfB = cdfB[cdfB.acquisition_week <= tend]
        customers = pd.concat([cdfA, cdfB], ignore_index=True)
        o1 = odfA[odfA["_t"] < 87].sort_values("_t")
        o2 = pd.concat([odfA[odfA["_t"] >= 87], odfB[odfB["_t"] < tend]]).sort_values("_t")
        orders = pd.concat([o1, o2], ignore_index=True)
    else:
        customers = cdfA
        orders = odfA[odfA["_t"] < tend].sort_values("_t")
    orders = orders.drop(columns="_t")
    orders.insert(0, "order_id", ["O%06d" % (i + 1) for i in range(len(orders))])
    customers = customers[customers.customer_id.isin(set(orders.customer_id))]

    # ---- influencers (weeks 1-52, 26 fortnights) ----
    nI, nF = 150, 26
    nfake = int(round(ri.uniform(.08, .12) * nI))
    fake = np.sort(ri.choice(nI, nfake, replace=False))
    base = ri.lognormal(np.log(40000), .9, nI)
    growth = ri.normal(.006, .003, (nI, nF)); jump = np.zeros((nI, nF))
    er = ri.uniform(.025, .06, nI); cr = ri.uniform(.02, .05, nI)
    gen = ri.uniform(.05, .2, nI); match = ri.uniform(.5, .8, nI)
    for i in fake:
        pats = ri.choice(5, ri.integers(2, 6), replace=False)
        if 0 in pats:
            for f_ in ri.choice(np.arange(2, nF), ri.integers(1, 4), replace=False):
                jump[i, f_] = ri.uniform(.2, .5)
        if 1 in pats: er[i] = ri.uniform(.07, .15)
        if 2 in pats: cr[i] = ri.uniform(.002, .01)
        if 3 in pats: gen[i] = ri.uniform(.4, .8)
        if 4 in pats: match[i] = ri.uniform(.1, .35)
    foll = np.zeros((nI, nF)); foll[:, 0] = base
    for f_ in range(1, nF):
        foll[:, f_] = foll[:, f_ - 1] * (1 + growth[:, f_] + jump[:, f_])
    likes = foll * er[:, None] * ri.lognormal(0, .25, (nI, nF))
    comm = likes * cr[:, None] * ri.lognormal(0, .25, (nI, nF))
    chs = np.array([sp[4][:, 2 * f_:2 * f_ + 2].sum() for f_ in range(nF)])
    rows = []
    for i in range(nI):
        for f_ in range(nF):
            rows.append({"creator_id": "INF%03d" % (i + 1), "fortnight": f_ + 1, "followers": int(foll[i, f_]),
                         "likes": int(likes[i, f_]), "comments": int(comm[i, f_]),
                         "follower_growth": round(float(foll[i, f_] / foll[i, f_ - 1] - 1) if f_ else 0.0, 4),
                         "generic_comment_share": round(float(np.clip(gen[i] + ri.normal(0, .03), 0, 1)), 3),
                         "audience_city_match": round(float(np.clip(match[i] + ri.normal(0, .03), 0, 1)), 3),
                         "spend_lakh": round(float(chs[f_] * foll[i, f_] / foll[:, f_].sum()), 4)})
    infl = pd.DataFrame(rows)
    T["influencer_fake_ids"] = ["INF%03d" % (i + 1) for i in fake]

    # ---- CRO sessions (stage 2 only) ----
    cro = None
    if stage == 2:
        n = 28 * 4000
        day = np.repeat(np.arange(28), 4000)
        var = np.where(rx.random(n) < .5, "one_page", "three_step")
        pc = np.where(var == "one_page", .03 * (1 + T["cro_relative_lift"]), .03)
        conv = (rx.random(n) < pc).astype(int)
        d0 = dates[96]
        cro = pd.DataFrame({"session_id": ["S%07d" % (i + 1) for i in range(n)],
                            "date": [d0 + dt.timedelta(days=int(d)) for d in day], "variant": var,
                            "converted": conv,
                            "order_value": np.where(conv == 1, (AOV * np.exp(rx.normal(0, .3, n) - .045)).round(2), 0.0)})
    return T, weekly, platform, customers, orders, infl, cro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int); ap.add_argument("--seed-file")
    ap.add_argument("--stage", type=int, default=1); ap.add_argument("--design")
    ap.add_argument("--out", required=True); ap.add_argument("--vault", required=True)
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else int(open(a.seed_file).read().strip())
    design = json.load(open(a.design)) if a.design else None
    T, weekly, platform, customers, orders, infl, cro = build(seed, a.stage, design)
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.vault, exist_ok=True)
    weekly.to_csv(f"{a.out}/weekly_geo.csv", index=False)
    platform.to_csv(f"{a.out}/platform_reported.csv", index=False)
    customers.to_csv(f"{a.out}/customers.csv", index=False)
    orders.to_csv(f"{a.out}/orders.csv", index=False)
    infl.to_csv(f"{a.out}/influencers.csv", index=False)
    if cro is not None:
        cro.to_csv(f"{a.out}/cro_sessions.csv", index=False)
    blob = json.dumps(T, sort_keys=True, indent=2).encode()
    open(f"{a.vault}/truth.json", "wb").write(blob)
    h = hashlib.sha256(blob).hexdigest()
    open(f"{a.vault}/HASH.txt", "w").write(h + "\n")
    print("stage", a.stage, "| files written to", a.out, "| truth SHA-256:", h)


if __name__ == "__main__":
    main()
