# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "../htf_projections/")
from data_functions import (
    station_data,
    station_string,
    lowpass_wts,
)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def main():

    # -----------------------------------------------------------------------

    fname = "../data/stations.pickle"
    with open(fname, "rb") as f:
        stations = pickle.load(f)

    # # restart
    # stations = stations.loc[36:, :]

    # -----------------------------------------------------------------------

    # remove stations to skip
    skip = ["8638901"]
    keep = [n for n in stations.index if stations.loc[n, "id"] not in skip]
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

    # select = ["8443970"]
    # keep = [n for n in stations.index if stations.loc[n, "id"] in select]
    # stations = stations.loc[keep]

    # -----------------------------------------------------------------------

    for idx, sta in stations.iterrows():
        tg = collect_data(sta)
        ssnl_tidal_range(sta)
        make_figures(sta, tg)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def collect_data(sta):

    # -----------------------------------------------------------------------

    # print('\n' + sta['name'] + ' (NOAA ID: ' + str(sta['noaa id']) + ')\n')

    # -----------------------------------------------------------------------
    # load station data ('sl' in cm)
    # also includes stock NOAA tidal hindcast ('td')

    tg = station_data(sta)

    # -----------------------------------------------------------------------
    # initial tide fit to distinguish major/minor constituents
    # adds two tidal height time series to 'tg':
    #   'utd_epch' is tidal heights reconstructed from all tidal constituents
    #         with signal:noise > 2 in fit during 5 year epoch
    #   'utd_epch_mnr' is tidal heights reconstructed from minor tidal
    #         constituents only

    # tg, mjr_cnst, coef_epch = initial_tide_fit(sta, tg)

    # -----------------------------------------------------------------------
    # load probabilistic tidal heights

    sta_str = station_string(sta)
    out_path = "./output/" + sta_str + "/"

    fname = out_path + "tide_prediction_mn_std.pickle"
    with open(fname, "rb") as f:
        ptd = pickle.load(f)

    tg["ptd_mean"] = ptd["obs"]["mn"].reindex(tg.index)
    tg["ptd_stdev"] = ptd["obs"]["sd"].reindex(tg.index)

    # -----------------------------------------------------------------------

    return tg


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def ssnl_tidal_range(sta):

    sta_str = station_string(sta)
    out_path = "./output/" + sta_str + "/"

    fname = out_path + "./tide_99th_percentile_monthly.pickle"
    with open(fname, "rb") as f:
        tdm = pickle.load(f)

    ssnl_td = tdm["obs"].groupby(tdm["obs"].index.month).mean().mean(axis=1)

    mx = "{:0.2f}".format(ssnl_td.max())
    mn = "{:0.2f}".format(ssnl_td.min())
    rng = "{:0.2f}".format(ssnl_td.max() - ssnl_td.min())

    print(sta_str)
    print("[" + mn + ", " + mx + ", " + rng + "]")
    print("")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def make_figures(sta, tg):

    # -----------------------------------------------------------------------

    sta_str = station_string(sta)

    out_path = "./output/" + sta_str + "/"

    fig_path = "./figures/" + sta_str + "/"
    os.makedirs(fig_path, exist_ok=True)

    # -----------------------------------------------------------------------

    wts = lowpass_wts(48)[0]
    res = tg.sl - tg.td
    res -= np.convolve(res.values, wts, mode="same")
    res_asd = res.groupby(pd.Grouper(freq="A")).apply(lambda x: x.std())
    res_asd.index = res_asd.index.year

    fname = out_path + "residuals_stdev_annual.pickle"
    with open(fname, "rb") as f:
        pres_asd = pickle.load(f)

    # -----------------------------------------------------------------------

    col = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig0 = plt.figure(num="std_res", figsize=[10, 5])
    fig0.clf()
    ax = plt.gca()
    res_asd.plot(ax=ax, color=col[1], label="traditional tide prediction (NOAA)")
    ax.fill_between(
        pres_asd.index,
        pres_asd[5.0],
        pres_asd[95.0],
        color=col[0],
        alpha=0.5,
        lw=0,
        label="probabilistic 90% credible interval",
    )
    pres_asd[50.0].plot(ax=ax, color="k", label="probabilistic prediction median")
    plt.xlim([1920, 2020])
    ymx = res_asd.max()
    ymn = pres_asd[50.0].min()
    yrng = ymx - ymn
    plt.ylim([np.floor(ymn - 0.1 * yrng), np.ceil(ymx + 0.2 * yrng)])
    plt.ylabel("cm")
    plt.title(
        sta["name"]
        + "\nAnnual standard deviation of tidal residuals "
        + r"($T\,\,\lesssim\,\,48$ hours)"
    )
    plt.legend(loc="upper left", ncol=2, frameon=False)
    plt.tight_layout()
    # fig0.show()

    fig0_name = fig_path + "annual_stdev_residuals.pdf"
    fig0.savefig(fig0_name)

    # -----------------------------------------------------------------------

    fname = out_path + "tide_99th_percentile_annual.pickle"
    with open(fname, "rb") as f:
        t99a = pickle.load(f)

    qtl = [0.05, 0.17, 0.5, 0.83, 0.95]
    t99a_ptl_obs = t99a["obs"].quantile(q=qtl, axis=1).T
    t99a_ptl_prjn = t99a["int_od"].quantile(q=qtl, axis=1).T

    fig1 = plt.figure(num="tide_99th_percentile_annual", figsize=[8, 4])
    fig1.clf()
    ax = plt.gca()

    t99a_ptl_prjn[0.5].plot(ax=ax, color=col[3], label="Projected median")
    ax.fill_between(
        t99a_ptl_prjn.index,
        t99a_ptl_prjn[0.05],
        t99a_ptl_prjn[0.95],
        color=col[1],
        alpha=0.4,
        lw=0,
        label="Projected 90% CI",
    )

    t99a_ptl_obs[0.5].plot(ax=ax, color="k", label="Observed median")
    ax.fill_between(
        t99a_ptl_obs.index,
        t99a_ptl_obs[0.05],
        t99a_ptl_obs[0.95],
        color=col[0],
        alpha=0.4,
        lw=0,
        label="Observed 90% CI",
    )

    # ax.fill_between(t99a_ptl.index, t99a_ptl[0.17], t99a_ptl[0.83],
    #     color=col[0], alpha=0.7, lw=0, label='67% credible interval')
    plt.xlim([1920, 2100])
    plt.xlabel("year")
    plt.ylabel("cm")
    plt.title(sta["name"] + "\nAnnual 99th percentile of tidal height")
    plt.legend(loc="upper left", ncol=2, frameon=False)
    plt.tight_layout()
    # fig1.show()

    fig1_name = fig_path + "annual_99th_percentile.pdf"
    fig1.savefig(fig1_name)

    # -----------------------------------------------------------------------

    # fname = out_path + 'tide_99th_percentile_monthly.pickle'
    # with open(fname, 'rb') as f:
    #     t99m = pickle.load(f)
    #
    # qtl = [0.05, 0.5, 0.95]
    # t99m_ptl = t99m.quantile(q=qtl, axis=1).T
    #
    # fig2 = plt.figure(num='tide_99th_percentile_monthly')
    # fig2.clf()
    # ax = plt.gca()
    # t99m_ptl[0.5].plot(ax=ax, color='k', label='50th percentile')
    # ax.fill_between(t99m_ptl.index, t99m_ptl[0.05], t99m_ptl[0.95],
    #     color=col[0], alpha=0.5, lw=0, label='90% credible interval')
    # plt.ylabel('cm')
    # plt.title(sta['name'] + ': Annual 99th percentile of tidal height')
    # plt.legend()
    # fig2.show()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
