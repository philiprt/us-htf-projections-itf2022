import os
import pickle
import json

import numpy as np
import pandas as pd

import scipy.special as spcl
import scipy.stats as stats


def station_string(sta):
    """Create a station-specific string for creating directories and filenames."""

    loc_str = "".join(
        [
            c if c != " " else "_"
            for c in [c for c in sta["tool_name"].lower() if c not in [",", "."]]
        ]
    )
    return loc_str + "_" + str(sta["id"])


def experiment_string(msl_scenario, threshold, time_of_day):
    """Create a string describing an exceedance days experiment."""
    tod_str = "h" + "{:0>2}".format(time_of_day[0]) + "{:0>2}".format(time_of_day[1])
    return msl_scenario + "_" + threshold + "_" + tod_str


def station_list(select=None, exclude=None, sortby="lon"):
    """Return a dataframe where each row contains information about a NOS
    tide station."""

    fname = "../data/stations.pickle"
    stations = pd.read_pickle(fname)

    if select is None:

        if isinstance(exclude, str):
            exclude = [exclude]

        if exclude is not None:
            keep = [n for n in stations.index if stations.loc[n, "id"] not in exclude]
            stations = stations.loc[keep]

        # prioritize certain stations in the order of completion
        first = ["1612340", "8443970", "8658120", "8545240", "9414290"]
        stations["sort"] = [
            first.index(stations.loc[n, "id"])
            if stations.loc[n, "id"] in first
            else n + len(first)
            for n in stations.index
        ]
        stations = stations.sort_values(by="sort")
        stations.index = range(stations.index.size)

    else:

        if isinstance(select, str):
            select = [select]

        stations = stations.loc[
            [n for n in stations.index if stations.loc[n, "id"] in select]
        ]

    stations.loc[stations["lon"] < 0, "lon"] += 360
    stations.sort_values(sortby, inplace=True)
    stations.index = range(stations.shape[0])

    return stations


# -----------------------------------------------------------------------
# Define functions describing mean (mu) of beta distributions

# gaussian CDF (GCDF)
def gcdf(x0, loc0, sc0):
    return 0.5 * (1 + spcl.erf((x0 - loc0) / (sc0 * np.sqrt(2))))


def logistic(x0, rt0, z0):
    # rt is the slope of transition, x0 is the location
    return 1 / (1 + np.exp(-rt0 * (x0 - z0)))


# -----------------------------------------------------------------------


def xd_from_e99(e99, trace, x_nrm, rt, Ndy):

    """Function for calculating exceedance days from eta99 above threshold."""

    # get height relative to threshold and normalize scale using x_nrm
    xp = (e99.values - x_nrm[0]) / (x_nrm[1] - x_nrm[0])

    # calculate mean (mu) of beta distributions
    mu_bta = np.zeros([xp.shape[0], trace["mu_loc1"].size])
    N_xd = np.zeros([xp.shape[0], trace["mu_loc1"].size])
    for k in range(trace["mu_loc1"].size):
        mu1 = logistic(xp[:, k], -rt, trace["z"][k]) * gcdf(
            xp[:, k], trace["mu_loc1"][k], trace["mu_sc1"][k]
        )
        mu2 = logistic(xp[:, k], rt, trace["z"][k]) * gcdf(
            xp[:, k], trace["mu_loc2"][k], trace["mu_sc2"][k]
        )
        mu_bta[:, k] = mu1 + mu2

    # calculate alpha/beta parameters of beta distributions
    v_bta = trace["nu"][None, :] * mu_bta * (1 - mu_bta)
    z = v_bta > 1e-15  # don't divide by zero
    z_min = ~z & (mu_bta < 0.5)
    z_max = ~z & (mu_bta > 0.5)
    mu_bta = mu_bta[z]
    v_bta = v_bta[z]
    kappa = mu_bta * (1 - mu_bta) / v_bta - 1
    a_bta = kappa * mu_bta
    b_bta = kappa * (1 - mu_bta)

    # get beta-distributed binomial probabilities
    p = np.zeros(xp.shape)
    p[z] = stats.beta.rvs(a_bta, b_bta)

    # get threshold exceedance days
    xd0 = stats.binom.rvs(Ndy, p).astype(float)
    xd0[z_min] = 0
    xd0[z_max] = Ndy
    xd = pd.DataFrame(xd0, index=e99.index).astype(int)

    return xd


def xdys_monthly_ensemble(e99_abv_thrsh, bbmodels):
    """Load arrays of monthly exceedance days ensembles into a single DataFrame."""

    time_of_day = [0, 23]
    rt = 1 / 0.1

    n_days = {
        m: 30 if m in [4, 6, 9, 11] else (28 if m == 2 else 31) for m in range(1, 13)
    }

    xd = []

    for m in range(1, 13):
        xd_mo = xd_from_e99(
            e99_abv_thrsh.loc[e99_abv_thrsh.index.get_level_values("month") == m],
            bbmodels[m]["trace"],
            bbmodels[m]["x_nrm"],
            rt,
            n_days[m],
        )
        xd.append(xd_mo)

    return pd.concat(xd).sort_index()


def ensemble_percentiles(df, percentiles):

    return pd.DataFrame(
        np.percentile(df, percentiles, axis=1).T,
        index=df.index,
        columns=percentiles,
    )


def monthly_calculations(ensemble, station, scenario, threshold, percentiles=None):

    if percentiles is None:
        percentiles = [5, 10, 17, 50, 83, 90, 95]

    ens_pctl = ensemble_percentiles(ensemble, percentiles).astype(int)

    years = ens_pctl.index.get_level_values(0).unique()
    ens_pctl_dict = {
        y: {p: ens_pctl.loc[(y, slice(None)), p].to_list() for p in percentiles}
        for y in years
    }

    # loop over monthly counts of interest
    mo_counts = [5, 10, 15, 20]
    mo_prob = pd.DataFrame(0, index=years, columns=mo_counts)
    mo_prob_frst_yr = pd.DataFrame(0, index=years, columns=mo_counts)
    for mc in mo_counts:

        # find probability of at least one monthly count in each year being greater than
        # some number
        mo_prob.loc[:, mc] = (
            ensemble.groupby("year").apply(lambda x: (x >= mc).any(axis=0)).sum(axis=1)
            / ensemble.shape[1]
        )

        # find the first monthly count exceeding some number in each ensemble member
        frst = pd.DataFrame([imx[0] for imx in (ensemble >= mc).idxmax().to_list()])

        # find probabilities of that first count occurring in each year
        p = frst.value_counts() / frst.shape[0]
        p.index = p.index.get_level_values(0)
        mo_prob_frst_yr.loc[p.index, mc] = p.values

    mo_prob_dict = dict(years=mo_prob.index.tolist())
    mo_prob_dict["prob_at_least_n"] = {c: mo_prob.loc[:, c].tolist() for c in mo_counts}
    mo_prob_dict["prob_first_year"] = {
        c: mo_prob_frst_yr.loc[:, c].tolist() for c in mo_counts
    }

    return ens_pctl_dict, mo_prob_dict


def annual_calculations(ensemble, station, scenario, threshold, percentiles=None):

    if percentiles is None:
        percentiles = [5, 10, 17, 50, 83, 90, 95]

    ann_ens = ensemble.groupby("year").sum()
    ann_ens_pctl = ensemble_percentiles(ann_ens, percentiles).astype(int)

    aep_dict = dict(
        years=ann_ens_pctl.index.to_list(),
        percentiles={p: ann_ens_pctl[p].to_list() for p in ann_ens_pctl.columns},
    )

    yoi = yoi_steps(ann_ens_pctl.loc[:, 50])

    # loop over annual counts of interest
    yr_counts = [10, 20, 26, 30, 50, 100]
    ann_prob = pd.DataFrame(0, index=ann_ens.index, columns=yr_counts)
    ann_prob_frst_yr = pd.DataFrame(0, index=ann_ens.index, columns=yr_counts)
    for yc in yr_counts:

        # find probability of annual counts being greater than some number
        ann_prob.loc[:, yc] = (ann_ens >= yc).sum(axis=1) / ann_ens.shape[1]

        # find the first annual count exceeding some number in each ensemble member
        frst = pd.DataFrame((ann_ens >= yc).idxmax().to_list())

        # find probabilities of that first count occurring in each year
        p = frst.value_counts() / frst.shape[0]
        p.index = p.index.get_level_values(0)
        ann_prob_frst_yr.loc[p.index, yc] = p.values

    ann_prob_dict = dict(years=ann_prob.index.tolist())
    ann_prob_dict["prob_at_least_n"] = {
        c: ann_prob.loc[:, c].tolist() for c in yr_counts
    }
    ann_prob_dict["prob_first_year"] = {
        c: ann_prob_frst_yr.loc[:, c].tolist() for c in yr_counts
    }

    return aep_dict, yoi, ann_prob_dict


def pentad_calculations(ensemble, station, scenario, threshold, percentiles=None):

    if percentiles is None:
        percentiles = [5, 10, 17, 50, 83, 90, 95]

    ensemble = ensemble.loc[ensemble.index.get_level_values("year") < 2100]

    # group by pentad
    pent = (np.floor(ensemble.index.get_level_values("year") / 5) * 5).astype(int)
    pent_grp = ensemble.groupby(pent)

    def make_dict(calc):
        return dict(
            pentads=calc.index.to_list(),
            percentiles={p: calc[p].to_list() for p in calc.columns},
        )

    # get average count per month in each pentad
    mean_mo = pent_grp.apply(lambda x: x.mean(axis=0))
    mean_mo_pctl = ensemble_percentiles(mean_mo, percentiles).round().astype(int)
    mean_mo_pctl_dict = make_dict(mean_mo_pctl)

    # calculate number of events in worst month of each pentad
    xd_pent_mxmo = pent_grp.apply(lambda x: x.max(axis=0))
    max_mo_pctl = ensemble_percentiles(xd_pent_mxmo, percentiles).astype(int)
    max_mo_pctl_dict = make_dict(max_mo_pctl)

    # calculate events per month in worst season of each pentad
    xd_ssn = ensemble.rolling(3, center=True).mean()
    pent_grp_ssn = xd_ssn.groupby(pent)
    xd_ssn_pent_mx = pent_grp_ssn.apply(lambda x: x.max(axis=0))
    max_ssn_pctl = ensemble_percentiles(xd_ssn_pent_mx, percentiles).astype(int)
    max_ssn_pctl_dict = make_dict(max_ssn_pctl)

    return mean_mo_pctl_dict, max_mo_pctl_dict, max_ssn_pctl_dict


def station_analysis(
    station,
    threshold=50,
    scenario="int",
):
    """
    Perform statistical analysis of exceedance days ensemble.

    The calculations produce a list of dictionaries (1 dict per station) with
    the following entries:

        "xdys_mo_ptl": Percentiles of monthly counts of exceedance days

        "xdys_mxmo_ptl": Percentiles of yearly max monthly counts

        "prob_mxmo": Probability that the highest monthly count in each year
            exceeds some number

        "prob_mxmo_frst_yr": Probability that each year contains the first
            monthly count to exceed some number

        "prob_mxmo_frst_mo": Probability that the first monthly count to exceed
            some number occurs during each calendar month

        "xdys_ann_ptl": Percentiles of annual counts of exceedance days from the
            ensemble

        "ratio_mx_ann_ptl": Percentiles of the ratio between the max monthly
            count and the annual count for each year

        "prob_ann": Probability that the annual count in each year exceeds some
            number

        "prob_ann_frst_yr": Probability that each year is the first year that
            the annual count exceeds some number

        "xdys_dec_md_pctl": Percentiles of the median annual count in each
            decade

        "xdys_dec_mn_pctl": Percentiles of the mean annual count in each decade

        "xdys_dec_mx_pctl": Percentiles of the maximum annual count in each
            decade

        "xdys_pent_md_pctl": Percentiles of the median annual count in each
            pentad

        "xdys_pent_mn_pctl": Percentiles of the mean annual count in each
            pentad

        "xdys_pent_mx_pctl": Percentiles of the maximum annual count in each
            pentad

    """

    if percentiles is None:
        percentiles = [5, 10, 17, 50, 83, 90, 95]

    # create dictionary to store analysis
    A = {"station": station}

    # load the ensemble of monthly counts of exceedance days;
    # function defined above
    xd = xdys_monthly_ensemble(
        station,
        threshold=threshold,
        scenario=scenario,
    )

    # calculate percentiles
    A["xdys_mo_ptl"] = monthly_percentiles(
        xd, station, scenario, threshold, percentiles=percentiles
    )

    # group by pentad
    pent = (np.floor(xd.index.get_level_values("year") / 5) * 5).astype(int)
    pent_grp = xd.groupby(pent)

    # calculate percentiles of the percentage of days in each pentad accounted
    # for by the top 10 percent of months
    xd_pent_sum = pent_grp.apply(lambda x: x.sum(axis=0))
    xd_pent_sum_top10pct = pent_grp.apply(
        lambda x: pd.Series(np.sum(np.sort(x.values, axis=0)[::-1][:6, :], axis=0))
    )
    xd_pent_sum[xd_pent_sum == 0] = None
    xd_pent_top10pct = xd_pent_sum_top10pct / xd_pent_sum
    xd_pent_top10pct = xd_pent_top10pct.iloc[:-1, :]
    y = xd_pent_sum.median(axis=1)
    z = (y == 0) | (y > 180 * 5)
    xd_pent_top10pct.loc[z, :] = None
    xd_pent_top10pct_ptl = xd_pent_top10pct.quantile(
        q=[p / 100 for p in percentiles], axis=1
    ).T
    xd_pent_top10pct_ptl.columns = percentiles
    itoss = (xd_pent_top10pct_ptl[50] == 1).idxmin() - 10
    xd_pent_top10pct_ptl.loc[:itoss, :] = None
    A["xdys_pent_top10pct_ptl"] = xd_pent_top10pct_ptl

    # calculate number of events in worst month of each pentad
    xd_pent_mxmo = pent_grp.apply(lambda x: x.max(axis=0))
    A["xdys_pent_mxmo_ptl"] = ensemble_percentiles(xd_pent_mxmo, percentiles)

    # calculate number of events in worst season of each year and pentad
    xd_ssn = xd.rolling(3, center=True).sum()

    xd_ssn_ann_mx = xd_ssn.groupby("year").apply(lambda x: x.max(axis=0))
    A["xdys_ann_mxssn_ptl"] = ensemble_percentiles(xd_ssn_ann_mx, percentiles)

    pent_grp_ssn = xd_ssn.groupby(pent)
    xd_ssn_pent_mx = pent_grp_ssn.apply(lambda x: x.max(axis=0))
    A["xdys_pent_mxssn_ptl"] = ensemble_percentiles(xd_ssn_pent_mx, percentiles)

    # get the maximum monthly count in each year; calculate percentiles
    xd_mxmo = xd.groupby("year").apply(lambda x: x.max(axis=0))
    A["xdys_mxmo_ptl"] = ensemble_percentiles(xd_mxmo, percentiles)

    # loop over monthly counts of interest
    A["prob_mxmo"] = pd.DataFrame(0, index=xd_mxmo.index, columns=mo_counts)
    A["prob_mxmo_frst_yr"] = pd.DataFrame(0, index=xd_mxmo.index, columns=mo_counts)
    A["prob_mxmo_frst_mo"] = pd.DataFrame(0, index=range(1, 13), columns=mo_counts)
    for mc in mo_counts:

        # find probability of max monthly counts being greater than some number
        A["prob_mxmo"].loc[:, mc] = (xd_mxmo >= mc).sum(axis=1) / xd.shape[1]

        # find the first monthly count exceeding some number in each ensemble member
        frst = pd.DataFrame((xd >= mc).idxmax().to_list(), columns=["year", "month"])

        # find probabilities of that first month occurring in each year
        p = frst.groupby("year").count() / frst.shape[0]
        A["prob_mxmo_frst_yr"].loc[p.index, mc] = p.values

        # ... in each month of the year
        p = frst.groupby("month").count() / frst.shape[0]
        A["prob_mxmo_frst_mo"].loc[p.index, mc] = p.values

    # find the total count in each year; calculate percentiles
    xd_ann = xd.groupby("year").sum()
    A["xdys_ann_ptl"] = ensemble_percentiles(xd_ann, percentiles)

    # get the ratio of max monthly to annual counts; calculate percentiles
    rat_mx_ann = xd_mxmo / xd_ann
    A["ratio_mx_ann_ptl"] = ensemble_percentiles(rat_mx_ann, percentiles)

    # loop over annual counts of interest
    A["prob_ann"] = pd.DataFrame(0, index=xd_ann.index, columns=yr_counts)
    A["prob_ann_frst_yr"] = pd.DataFrame(0, index=xd_ann.index, columns=yr_counts)
    for yc in yr_counts:

        # find probability of annual counts being greater than some number
        A["prob_ann"].loc[:, yc] = (xd_ann >= yc).sum(axis=1) / xd.shape[1]

        # find the first annual count exceeding some number in each ensemble member
        frst = pd.DataFrame((xd >= yc).idxmax().to_list(), columns=["year", "month"])

        # find probabilities of that first count occurring in each year
        p = frst.groupby("year").count() / frst.shape[0]
        A["prob_ann_frst_yr"].loc[p.index, yc] = p.values

    # find the median, mean, and max annual count in each decade
    dec = (np.floor(xd_ann.index / 10) * 10).astype(int)
    xd_dec_md = xd_ann.groupby(dec).median()
    xd_dec_mn = xd_ann.groupby(dec).mean()
    xd_dec_mx = xd_ann.groupby(dec).apply(lambda x: x.max(axis=0))

    # calculate percentiles of the above quantities
    A["xdys_dec_md_ptl"] = ensemble_percentiles(xd_dec_md, percentiles)
    A["xdys_dec_mn_ptl"] = ensemble_percentiles(xd_dec_mn, percentiles)
    A["xdys_dec_mx_ptl"] = ensemble_percentiles(xd_dec_mx, percentiles)

    # find the median, mean, and max annual count in each pentad
    pent = (np.floor(xd_ann.index / 5) * 5).astype(int)
    xd_pent_md = xd_ann.groupby(pent).median()
    xd_pent_mn = xd_ann.groupby(pent).mean()
    xd_pent_mn_nomxssn = (xd_pent_mn * 5 - xd_ssn_pent_mx) / (5 * 12 - 3)
    xd_pent_mx = xd_ann.groupby(pent).apply(lambda x: x.max(axis=0))

    # calculate percentiles of the above quantities
    A["xdys_pent_md_ptl"] = ensemble_percentiles(xd_pent_md, percentiles)
    A["xdys_pent_mn_ptl"] = ensemble_percentiles(xd_pent_mn, percentiles)
    A["xdys_pent_mn_nomxssn_ptl"] = ensemble_percentiles(
        xd_pent_mn_nomxssn, percentiles
    )
    A["xdys_pent_mx_ptl"] = ensemble_percentiles(xd_pent_mx, percentiles)

    # record threshold and scenario
    A["experiment"] = {
        "threshold": threshold,
        "scenario": scenario,
    }

    return A


def load_analysis_results(slr_scenarios, thresholds, time_of_day):

    # load analysis for each station, scenario, and threshold
    analysis = {}
    stations = []

    init = True

    for scn in slr_scenarios:
        for thrsh in thresholds:

            fname = (
                "./analysis_results_"
                + experiment_string(scn, thrsh, time_of_day)
                + ".pickle"
            )

            with open(fname, "rb") as f:
                a = pd.read_pickle(f)

            for n in range(len(a)):

                stn = a[n]["station"]["name"]

                if init:
                    analysis[stn] = {
                        scn: {thrsh: {} for thrsh in thresholds}
                        for scn in slr_scenarios
                    }
                    stations.append(a[n]["station"])

                analysis[stn][scn][thrsh] = a[n]

            init = False

    stations = pd.concat(stations, axis=1).T
    stations = stations_by_reg(stations)

    return stations, analysis


def stations_by_reg(stations):

    # organize stations by region and sort to make the ordering intuitive
    stn_by_reg = {}

    stn_by_reg["Pacific Islands"] = stations[stations["lon"] < 230].sort_values(
        by="lon", ascending=True
    )

    stn_by_reg["California"] = stations[
        (stations["lon"] > 230) & (stations["lon"] < 250) & (stations["lat"] <= 43)
    ].sort_values(by="lat", ascending=True)

    stn_by_reg["Oregon & Washington"] = stations[
        (stations["lon"] > 230) & (stations["lon"] < 250) & (stations["lat"] > 43)
    ].sort_values(by="lat", ascending=True)

    stn_by_reg["Gulf of Mexico, West"] = stations[
        (stations["lon"] > 250) & (stations["lon"] < 272)
    ].sort_values(by="lon", ascending=True)

    stn_by_reg["Gulf of Mexico, Florida"] = stations[
        (stations["lon"] > 272) & (stations["lon"] < 278.4)
    ].sort_values(by="lat", ascending=False)

    stn_by_reg["Caribbean"] = stations[stations["lon"] > 292].sort_values(
        by="lon", ascending=True
    )

    stn_by_reg["Atlantic Coast, South"] = stations[
        (stations["lon"] > 278.4) & (stations["lon"] < 282.5)
    ].sort_values(by="lat", ascending=True)

    atl_nrth = stations[
        (stations["lon"] > 282.5) & (stations["lon"] < 292)
    ].sort_values(by="lat", ascending=True)

    atl_nrth_states_1 = ["NC", "VA", "DC", "MD", "DE", "PA"]
    stn_by_reg["Atlantic Coast, North 1"] = pd.concat(
        [
            stn
            for state in [
                [stn[1] for stn in atl_nrth.iterrows() if stn[1]["name"][-2:] == st]
                for st in atl_nrth_states_1
            ]
            for stn in state
        ],
        axis=1,
    ).T

    atl_nrth_states_2 = ["NJ", "NY", "CT", "RI", "MA", "ME"]
    stn_by_reg["Atlantic Coast, North 2"] = pd.concat(
        [
            stn
            for state in [
                [stn[1] for stn in atl_nrth.iterrows() if stn[1]["name"][-2:] == st]
                for st in atl_nrth_states_2
            ]
            for stn in state
        ],
        axis=1,
    ).T

    for reg in stn_by_reg:
        stn_by_reg[reg]["Region"] = reg

    stn_by_reg = pd.concat([stn_by_reg[reg] for reg in stn_by_reg], axis=0)
    stn_by_reg.set_index("name", inplace=True)
    stn_by_reg = stn_by_reg.loc[:, ["id", "Region", "lat", "lon"]]
    stn_by_reg.columns = ["id", "Region", "Latitude", "Longitude"]
    stations.index.name = "Name"

    return stn_by_reg


# def analysis_by_reg_subsets(a_by_reg, subsets):

#     analysis = {}

#     for reg in subsets:

#         for s in subsets[reg]:
#             s["steps"] = s["steps"] if "steps" in s.keys() else None

#         analysis[reg] = [
#             [
#                 {**a, **{"steps": s["steps"]}}
#                 for a in a_by_reg[s["threshold"]][reg]
#                 if a["station"]["name"] == s["name"]
#             ][0]
#             for s in subsets[reg]
#         ]

#     return analysis


def yoi_steps(prjn, search_range=None):

    if search_range is not None:
        prjn = prjn.loc[search_range[0] - 10 : search_range[1] + 10]

    deltas = prjn.rolling(21, center=True).apply(
        lambda x: (x[20] - x[10]) - (x[10] - x[0]), raw=True
    )
    multipliers = prjn.rolling(21, center=True).apply(lambda x: x[20] / x[10], raw=True)

    deltas = deltas[(prjn > 1) & (prjn <= 30) & deltas.notna()]
    multipliers = multipliers.loc[deltas.index]
    drnk = deltas.rank()
    mrnk = multipliers.rank()
    avgrnk = (drnk + mrnk) / 2
    try:
        yoi = avgrnk.idxmax()
        return [int(yoi + d) for d in [-10, 0, 10]]
    except:
        return None


def yoi_analysis(analysis, stations, search_range=None, combine_regions=None):

    yoi = []

    for stn in analysis:
        for scn in analysis[stn]:
            for thrsh in analysis[stn][scn]:

                qty = analysis[stn][scn][thrsh]["xdys_ann_ptl"].loc[:, 50]
                steps = yoi_steps(qty, search_range=search_range)

                base = {
                    "Name": stn,
                    "id": stations.loc[stn, "id"],
                    "Region": stations.loc[stn, "Region"],
                    "Scenario": scn,
                    "Threshold": thrsh,
                }

                if (steps is not None) and ~np.isnan(steps[1]):

                    steps = qty.loc[steps]

                    calc = {
                        "YOI": steps.index[1],
                        "10-year increase": steps.diff().iloc[2],
                        "10-year multiplier": (
                            np.round(100 * steps.iloc[2] / steps.iloc[1]) / 100
                        ),
                    }

                else:

                    calc = {
                        "YOI": None,
                        "10-year increase": None,
                        "10-year multiplier": None,
                    }

                yoi.append({**base, **calc})

    yoi_df = pd.DataFrame(yoi)

    if combine_regions is not None:
        for new_reg in combine_regions:
            for old_reg in combine_regions[new_reg]:
                z = yoi_df["Region"] == old_reg
                yoi_df.loc[z, "Region"] = new_reg

    return yoi_df


def first_month_analysis(count, analysis, stations, combine_regions=None):

    ann_count_lookup = {5: 60, 10: 120, 15: 180}
    ann_count = ann_count_lookup[count]

    P = [25, 50]

    fmo = []

    for stn in analysis:
        for scn in analysis[stn]:
            for thrsh in analysis[stn][scn]:

                a = analysis[stn][scn][thrsh]

                cum_p_fmo = a["prob_mxmo_frst_yr"][count].cumsum().loc[2020:]
                p_first_mo = [(cum_p_fmo > p / 100).idxmax() for p in P]

                calc = {
                    "Year": [
                        p_first_mo[0],
                        p_first_mo[1],
                        a["prob_mxmo"][count].ge(0.5).idxmax(),
                        a["xdys_ann_ptl"][50].ge(ann_count).idxmax(),
                    ],
                    "Quantity": [
                        "P" + str(P[0]),
                        "P" + str(P[1]),
                        "Expected monthly",
                        "Expected annually",
                    ],
                }

                N = len(calc["Year"])
                base = {
                    "Name": [stn] * N,
                    "Region": [stations.loc[stn, "Region"]] * N,
                    "Scenario": [scn] * N,
                    "Threshold": [thrsh] * N,
                }

                fmo.append(pd.DataFrame({**base, **calc}))

    fmo_df = pd.concat(fmo, axis=0, ignore_index=True)

    if combine_regions is not None:
        for new_reg in combine_regions:
            for old_reg in combine_regions[new_reg]:
                z = fmo_df["Region"] == old_reg
                fmo_df.loc[z, "Region"] = new_reg

    return fmo_df


def first_month_by_region(count, a_by_reg, yoi_df):

    ann_count_lookup = {5: 60, 10: 120, 15: 180}
    ann_count = ann_count_lookup[count]

    fmo_by_reg = {}

    for reg in [r for r in a_by_reg["minor"]]:

        fmo_by_reg[reg] = {}

        for mnr, mdt in zip(a_by_reg["minor"][reg], a_by_reg["moderate"][reg]):

            # check that mnr and mdt refer to same station
            if mnr["station"]["name"] != mdt["station"]["name"]:
                raise Exception("Station match failed.")

            thrsh = yoi_df.loc[yoi_df["Name"] == mnr["station"]["name"]][
                "Threshold"
            ].iloc[0]

            if thrsh == "Minor":
                stn = mnr
            elif thrsh == "Moderate":
                stn = mdt

            cum_p_fmo = stn["prob_mxmo_frst_yr"][count].cumsum().loc[2020:]

            prb = [10, 25, 50, 75, 90]

            fmo_by_reg[reg][mnr["station"]["name"]] = {
                "p_first_mo": pd.Series(
                    [(cum_p_fmo > p / 100).idxmax() for p in prb], index=prb
                ),
                "xpctd_mo": stn["prob_mxmo"][count].ge(0.5).idxmax(),
                "xpctd_ann": stn["xdys_ann_ptl"][50].ge(ann_count).idxmax(),
                "threshold": thrsh,
            }

    return fmo_by_reg


def first_month_aggregate_dataframe(fmo_by_reg, combine_regions=None):

    P = [25, 50]

    fmo_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "Name": [loc for n in range(4)],
                    "Region": [reg for n in range(4)],
                    "Threshold": [fmo_by_reg[reg][loc]["threshold"] for n in range(4)],
                    "Year": [
                        fmo_by_reg[reg][loc]["p_first_mo"].loc[P[0]],
                        fmo_by_reg[reg][loc]["p_first_mo"].loc[P[1]],
                        fmo_by_reg[reg][loc]["xpctd_mo"],
                        fmo_by_reg[reg][loc]["xpctd_ann"],
                    ],
                    "Quantity": [
                        "P" + str(P[0]),
                        "P" + str(P[1]),
                        "Expected monthly",
                        "Expected annually",
                    ],
                }
            )
            for reg in fmo_by_reg
            for loc in fmo_by_reg[reg]
        ],
        axis=0,
        ignore_index=True,
    )

    if combine_regions is not None:
        for new_reg in combine_regions:
            for old_reg in combine_regions[new_reg]:
                z = fmo_df["Region"] == old_reg
                fmo_df.loc[z, "Region"] = new_reg

    return fmo_df


def peak_season_analysis(analysis, stations, combine_regions=None):

    pks = []

    for stn in analysis:
        for scn in analysis[stn]:
            for thrsh in analysis[stn][scn]:

                a = analysis[stn][scn][thrsh]

                d = {
                    "average": a["xdys_pent_mn_ptl"][50] / 12,
                    "peak_ssn": a["xdys_pent_mxssn_ptl"][50] / 3,
                    "peak_mo": a["xdys_pent_mxmo_ptl"][50],
                }
                qty = d["peak_mo"] / d["average"]
                #                 qty = (3 * d["peak_ssn"]) / (12 * 5 * d["average"])
                qty.loc[d["average"] < 1] = None
                N = qty.values.size

                calc = {
                    "Year": qty.index.values,
                    "Quantity": qty.values,
                }

                base = {
                    "Name": [stn] * N,
                    "Region": [stations.loc[stn, "Region"]] * N,
                    "Scenario": [scn] * N,
                    "Threshold": [thrsh] * N,
                }

                pks.append(pd.DataFrame({**base, **calc}))

    pks_df = pd.concat(pks, axis=0, ignore_index=True)

    if combine_regions is not None:
        for new_reg in combine_regions:
            for old_reg in combine_regions[new_reg]:
                z = pks_df["Region"] == old_reg
                pks_df.loc[z, "Region"] = new_reg

    return pks_df


def top10_percent_analysis(analysis, stations, combine_regions=None):

    t10 = []

    for stn in analysis:
        for scn in analysis[stn]:
            for thrsh in analysis[stn][scn]:

                a = analysis[stn][scn][thrsh]

                calc = {
                    "Year": a["xdys_pent_top10pct_ptl"].index.values,
                    "Quantity": a["xdys_pent_top10pct_ptl"][50].values,
                    "Total": a["xdys_pent_mn_ptl"][50].values[:-1] * 5,
                }

                N = calc["Year"].size
                base = {
                    "Name": [stn] * N,
                    "Region": [stations.loc[stn, "Region"]] * N,
                    "Scenario": [scn] * N,
                    "Threshold": [thrsh] * N,
                }

                t10.append(pd.DataFrame({**base, **calc}))

    t10_df = pd.concat(t10, axis=0, ignore_index=True)

    if combine_regions is not None:
        for new_reg in combine_regions:
            for old_reg in combine_regions[new_reg]:
                z = t10_df["Region"] == old_reg
                t10_df.loc[z, "Region"] = new_reg

    return t10_df


class ProgressBar:
    def __init__(self, total, description=None):

        self.total = total
        self.decimals = 1
        self.length = 50
        self.fill = ">"
        self.iteration = 0
        self.intervals = np.ceil(np.linspace(1, self.total, self.length))
        self.percent = np.linspace(0, 100, self.length + 1)[1:].astype(int)

        unq, cnt = np.unique(self.intervals, return_counts=True)
        idx = np.array([np.where(self.intervals == u)[0][-1] for u in unq])
        self.intervals = self.intervals[idx]
        self.percent = self.percent[idx]
        self.cumcnt = np.cumsum(cnt)

        if description:
            print(description)
        print("|%s| 0%% complete" % (" " * self.length), end="\r")

    def update(self):

        self.iteration += 1

        if self.iteration in self.intervals:
            prog = np.where(self.intervals == self.iteration)[0][-1]
            pctstr = str(self.percent[prog]) + "%"
            bar = self.fill * self.cumcnt[prog] + " " * (
                self.length - self.cumcnt[prog]
            )
            print("\r|%s| %s complete" % (bar, pctstr), end="\r")
            if self.iteration == self.total:
                print()  # blank line on completion
