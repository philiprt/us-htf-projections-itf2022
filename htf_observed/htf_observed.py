# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import xarray as xr
import json
import os

# ---------------------------------------------------------------------------

fname = "../data/stations.pickle"
stations = pd.read_pickle(fname)

# -----------------------------------------------------------------------

# # only do select stations
# select = ["8638901"]
# keep = [n for n in stations.index if stations.loc[n, "id"] in select]
# stations = stations.loc[keep]

# ---------------------------------------------------------------------------

n = 0
Nsta = stations.index.size
for idx, sta in stations.iterrows():

    n += 1

    print(
        "Station "
        + str(sta["id"])
        + ": "
        + sta["name"]
        + " ("
        + str(n)
        + " of "
        + str(Nsta)
        + ")"
    )

    # if n < 40:
    #     continue

    with open("./levels/" + str(sta["id"]) + ".json") as f:
        lev = json.load(f)

    try:
        tg = xr.open_dataset("../data/tide_gauge/" + str(sta["id"]) + ".nc")
    except:
        print("    Skipped. No tide gauge data file.")
        continue

    tg = tg.observed.to_pandas()

    dmax = tg.groupby(pd.Grouper(freq="D")).max()
    dmin = tg.groupby(pd.Grouper(freq="D")).min()
    day_min_max = pd.DataFrame({"max": dmax, "min": dmin})

    yrs = dmax.index.year.unique().values[1:]
    met_yrs = [[str(y - 1) + "-05-01", str(y) + "-04-30"] for y in yrs]

    yrs_complete = [
        int(y)
        for y, my in zip(yrs, met_yrs)
        if dmax.loc[my[0] : my[1]].dropna().count() >= 292
    ]

    analyze_long_term = True if len(yrs_complete) >= 30 else False

    if analyze_long_term:

        f10_yr1 = max([yrs_complete[0], 1970])
        f10_yr2 = f10_yr1 + 9
        l10_yr2 = yrs_complete[-1]
        l10_yr1 = l10_yr2 - 9

        f10_msl = tg.loc[f"{f10_yr1-1}-05":f"{f10_yr2}-04"].mean()
        l10_msl = tg.loc[f"{l10_yr1-1}-05":f"{l10_yr2}-04"].mean()
        msl_change = float(np.round(l10_msl - f10_msl, 2))

        f10_dmax = dmax.loc[f"{f10_yr1-1}-05":f"{f10_yr2}-04"]
        f10_Nyrs = f10_dmax.dropna().count() / 365.25

        l10_dmax = dmax.loc[f"{l10_yr1-1}-05":f"{l10_yr2}-04"]
        l10_Nyrs = l10_dmax.dropna().count() / 365.25

        # --------------------------------------------------------------------
        # get random pairs of 10-year periods

        yrs1 = yrs_complete[:-10]

        ryrs1 = [yrs1[k] for k in np.random.randint(low=0, high=len(yrs1), size=100)]

        rpairs10 = []  # random pairs of 10-year periods
        ryp = []
        for k, y1 in enumerate(ryrs1):

            yrs2 = [y for y in yrs1 if np.abs(y - y1) > 10]
            if len(yrs2) == 0:
                continue

            y2 = yrs2[np.random.randint(low=0, high=len(yrs2))]
            if ([y1, y2] in ryp) or ([y2, y1] in ryp):
                continue

            tg1 = tg.loc[f"{y1-1}-05":f"{y1+9}-04"]
            dmax1 = dmax.loc[f"{y1-1}-05":f"{y1+9}-04"] - tg1.mean() + f10_msl
            dmax1_Nyrs = dmax1.dropna().count() / 365.25

            tg2 = tg.loc[f"{y2-1}-05":f"{y2+9}-04"]
            dmax2 = dmax.loc[f"{y2-1}-05":f"{y2+9}-04"] - tg2.mean() + f10_msl
            dmax2_Nyrs = dmax2.dropna().count() / 365.25

            ryp.append([y1, y2])
            rpairs10.append(
                dict(
                    dmax1=dmax1,
                    dmax2=dmax2,
                    dmax1_Nyrs=dmax1_Nyrs,
                    dmax2_Nyrs=dmax2_Nyrs,
                )
            )

    # --------------------------------------------------------------------

    path = f"./htf_observed/{sta['id']}/"
    os.makedirs(path, exist_ok=True)

    thresholds = [h / 100 for h in range(0, 306)]
    for h in thresholds:

        fld_dys = {}

        ann_cnt = [
            (dmax.loc[my[0] : my[1]] > h).sum()
            if dmax.loc[my[0] : my[1]].dropna().count() >= 292
            else None
            for my in met_yrs
        ]
        fld_dys["annual"] = {
            "counts": [int(c) if c is not None else None for c in ann_cnt],
            "years": [int(y) if y is not None else None for y in yrs],
        }

        mo_cnt = dmax.groupby(pd.Grouper(freq="MS")).apply(
            lambda x: (x > h).sum() if x.dropna().count() >= 24 else None
        )

        fld_dys["monthly"] = {
            str(y): [
                int(mo_cnt.loc[f"{y}-{m:02d}"].values[0])
                if m in mo_cnt.loc[f"{y}"].dropna().index.month
                else None
                for m in range(1, 13)
            ]
            for y in mo_cnt.index.year.unique()
        }

        fld_dys["climatology"] = [
            int(c) for c in (dmax > h).groupby(dmax.index.month).sum().to_list()
        ]

        if analyze_long_term:

            f10_htf = int((f10_dmax > h).sum())
            f10_htf_pyr = float(np.round(f10_htf / f10_Nyrs, 2))

            l10_htf = int((l10_dmax > h).sum())
            l10_htf_pyr = float(np.round(l10_htf / l10_Nyrs, 2))

            htf_change = int(l10_htf - f10_htf)
            htf_pyr_change = int(np.round(l10_htf_pyr - f10_htf_pyr))

            htf_pyr_diff_dist = pd.Series(
                [
                    (rp["dmax2"] > h).sum() / rp["dmax2_Nyrs"]
                    - (rp["dmax1"] > h).sum() / rp["dmax1_Nyrs"]
                    for rp in rpairs10
                ]
            )
            htf_pyr_diff_dist = htf_pyr_diff_dist.abs()

            htf_pyr_diff_prob = int(
                np.round(
                    100
                    * (htf_pyr_diff_dist > np.abs(htf_pyr_change)).sum()
                    / len(rpairs10)
                )
            )

            fld_dys["long_term"] = dict(
                first_10_span=[f10_yr1, f10_yr2],
                last_10_span=[l10_yr1, l10_yr2],
                msl_change=np.round(msl_change, 4),
                f10_htf=f10_htf,
                f10_htf_pyr=f10_htf_pyr,
                l10_htf=l10_htf,
                l10_htf_pyr=l10_htf_pyr,
                htf_change=htf_change,
                htf_pyr_change=htf_pyr_change,
                htf_pyr_diff_prob=htf_pyr_diff_prob,
            )

        else:

            fld_dys["long_term"] = None

        fname = f"{path}{int(round(100 * h)):0>3}.json"
        with open(fname, "w") as f:
            json.dump(fld_dys, f)

    # -----------------------------------------------------------------------

    path = "./day_min_max/"
    os.makedirs(path, exist_ok=True)
    fname = path + str(sta["id"]) + ".csv"
    day_min_max.to_csv(fname)

    # -----------------------------------------------------------------------
