# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

import os
import pickle
import shutil
import sys
import time
from multiprocessing import Pool

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import utide
from pandarallel import pandarallel
from scipy.interpolate import splrep, splev

from data_functions import station_data

# below must be done before importing numpy
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.insert(0, "../htf_projections/")
from support_functions import station_string, LoopInfo


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def main():
    # mode = "parallel"
    mode = "series"

    # -----------------------------------------------------------------------

    fname = "../data/stations.pickle"
    with open(fname, "rb") as f:
        stations = pickle.load(f)

    # # restart
    # stations = stations.loc[72:, :]

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

    # only keep stations that do not have a specific output file
    def fpath(sta):
        sta_str = station_string(sta)
        return "./output/" + sta_str + "/tide_99th_percentile_monthly.pickle"

    keep = [n for n in stations.index if not os.path.isfile(fpath(stations.loc[n, :]))]
    stations = stations.loc[keep]

    # # only do select stations
    # select = ["1612340"]
    # keep = [n for n in stations.index if stations.loc[n, "id"] in select]
    # stations = stations.loc[keep]

    # -----------------------------------------------------------------------
    # series loop

    if mode == "series":

        lp = LoopInfo(stations.index.size)
        for _, sta in stations.iterrows():
            lp.begin_iteration(sta)
            probabilistic_tides([sta, mode])
            lp.end_iteration()

    # -----------------------------------------------------------------------
    # parallel processing

    elif mode == "parallel":

        try:
            shutil.rmtree("./progress/")
        except:
            pass
        os.makedirs("./progress/")

        pool_input = [[sta, mode] for _, sta in stations.iterrows()]

        agents = 24
        chunksize = 1
        with Pool(processes=agents) as pool:
            result = pool.map(probabilistic_tides, pool_input, chunksize)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# convert posterior amplitudes and phases into tidal variations;
# use utide to deal with greenwich phase lags;
# function below handles a single realization/sample from the posterior


def constituent_realizations(loop_args):
    global jyr_pred, jhr_pred, Ep, Em

    pstr_amp, pstr_phs = loop_args[0], loop_args[1]

    amp_rep = splrep(jyr_pred, pstr_amp)
    amp_ev = splev(jhr_pred, amp_rep)

    phs_rep = splrep(jyr_pred, pstr_phs)
    phs_ev = splev(jhr_pred, phs_rep)

    ap = 0.5 * amp_ev * np.exp(-1j * phs_ev * np.pi / 180)
    am = np.conj(ap)

    mod_td_cnst = np.real(Ep * ap + Em * am)

    return mod_td_cnst


# function for initializing parallel workers to avoid passing large arrays
# for every iteration of above function
def pool_initializer(base_args):
    global jyr_pred, jhr_pred, Ep, Em
    jyr_pred, jhr_pred, Ep, Em = (
        base_args[0],
        base_args[1],
        base_args[2],
        base_args[3],
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def lowpass_wts(Tc, N=None):
    """
    Return weights for a windowed-sinc low-pass filter.

    Args:
        Tc (int): Filter cut-off period in units of the data time step.
        N (int): Length of the filter.

    Returns:
        wts: Filter weights normalized to unit sum.
        B: Limits of the transition band from full to zero power. Given as a
            two element array of periods in units of the data time step
    """
    fc = 1 / Tc  # cut-off frequency

    if N is None:
        N = 3 * Tc
    if not N % 2:
        N += 1  # make sure that N is odd

    # compute sinc filter
    n = np.arange(N)
    wts = np.sinc(2 * fc * (n - (N - 1) / 2.0))

    # compute Hamming window
    win = np.hamming(N)

    # multiply sinc filter and window
    wts = wts * win

    # normalize to get unity gain
    wts = wts / np.sum(wts)

    # calculate transition band
    b = 3.3 / N  # width of transition band in frequency for hamming window
    B = 1 / np.array([fc - b / 2, fc + b / 2])

    return wts, B


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def probabilistic_tides(params):
    sta, mode = params[0], params[1]

    # -----------------------------------------------------------------------
    # set up for io

    sta_str = station_string(sta)

    out_path = "./output/" + sta_str + "/"
    os.makedirs(out_path, exist_ok=True)

    fig_path = "./figures/" + sta_str + "/"
    os.makedirs(fig_path, exist_ok=True)

    # -----------------------------------------------------------------------
    # set up for logging

    log_file = "./progress/" + sta_str + "_log.txt"
    lf = open(log_file, "w") if mode == "parallel" else None

    if mode == "parallel":
        print(sta["tool_name"] + " (NOAA ID: " + str(sta["id"]) + ")", file=lf)

    # -----------------------------------------------------------------------
    # load station data ('sl' in cm)
    # also includes stock NOAA tidal hindcast ('td')

    # tg.sl has units of cm above mhhw
    tg, acyc = station_data(sta, reuse_acyc_fit=True)

    # -----------------------------------------------------------------------
    # load info from initial tide fit

    out_name = out_path + "initial_tide_fit.pickle"
    with open(out_name, "rb") as f:
        ifit = pickle.load(f)

    mjr_cnst, coef_epch = ifit["mjr_cnst"], ifit["coef_epch"]

    # get minor constituents
    mnr_cnst = [
        c
        for c, p, s in zip(
            coef_epch.name, coef_epch.diagn["PE"], coef_epch.diagn["SNR"]
        )
        if (c not in mjr_cnst) and (s >= 2.0) and (c not in ["SA", "SSA"])
    ]

    # -------------------------------------------------------------------
    # gather arguments for tidal projections

    ngflgs = [
        coef_epch["aux"]["opt"]["nodsatlint"],
        coef_epch["aux"]["opt"]["nodsatnone"],
        coef_epch["aux"]["opt"]["gwchlint"],
        coef_epch["aux"]["opt"]["gwchnone"],
    ]

    tref = coef_epch["aux"]["reftime"]
    lat = coef_epch["aux"]["lat"]

    # -----------------------------------------------------------------------
    # setup hourly time vector

    dt_pred = pd.date_range(
        pd.Timestamp("1920-01-01"), pd.Timestamp("2101-01-01"), freq="H"
    )[:-1]

    if sta["id"] == "1612340":  # honolulu
        z = dt_pred < pd.Timestamp("1947-06-13")
        zidx = dt_pred[z] + pd.Timedelta("30 m")
        dt_pred = zidx.append(dt_pred[~z])

    jhr_pred = mdates.date2num(dt_pred)

    # -----------------------------------------------------------------------
    # generate prediction of minor constituents for jhr_pred

    rc_mnr_pred = utide.reconstruct(
        jhr_pred, coef_epch, constit=mnr_cnst, verbose=False
    )
    rc_mnr_pred.h -= rc_mnr_pred.h.mean()

    # -----------------------------------------------------------------------
    # generate annual cycle variations for jhr_pred

    acyc_pred = acyc.acyc_pred(jhr_pred)
    acyc_pred -= acyc_pred.mean()

    # -----------------------------------------------------------------------
    # set up for obs and prjn scenarios

    out_name = out_path + "posterior_samples_" + mjr_cnst[0] + ".pickle"
    with open(out_name, "rb") as f:
        pstr = pickle.load(f)

    obs_jyr, prjn_amp, prjn_jyr = (
        pstr["jyr_obs"],
        pstr["amp_prjn"],
        pstr["jyr_prjn"],
    )

    # isolate scenarios
    iso_scn = ["obs", "low_od", "int_low_od", "int_od", "int_high_od", "high_od"]
    # iso_scn = None
    if iso_scn is not None:
        prjn_amp = {s: prjn_amp[s] for s in prjn_amp if s in iso_scn}

    obs_d1 = mdates.date2num(
        mdates.num2date(
            mdates.datestr2num(str(mdates.num2date(obs_jyr[0]).year) + "-01-01")
        )
    )
    obs_i1 = np.argmax(jhr_pred >= obs_d1)

    obs_d2 = mdates.date2num(
        mdates.num2date(
            mdates.datestr2num(str(mdates.num2date(obs_jyr[-1]).year + 1) + "-01-01")
        )
    )
    obs_i2 = np.argmin(jhr_pred < obs_d2)

    obs_jhr = jhr_pred[obs_i1:obs_i2]
    obs_dt = dt_pred[obs_i1:obs_i2]

    prjn_i1 = np.argmax(jhr_pred >= mdates.datestr2num("2020-01-01"))

    prjn_jhr = jhr_pred[prjn_i1:]
    prjn_dt = dt_pred[prjn_i1:]

    # -----------------------------------------------------------------------
    # initialize dataframe for probabilistic tide realizations
    # initialize with deterministic minor constituents
    # the time mean of reconstructed minor constituents is zero

    n_pstr_smpls = 200
    ptd = {"obs": pd.DataFrame(0, index=obs_dt, columns=range(n_pstr_smpls))}
    ptd["obs"] = ptd["obs"].add(rc_mnr_pred.h[obs_i1:obs_i2], axis=0)
    for scn in prjn_amp:
        ptd[scn] = pd.DataFrame(0, index=prjn_dt, columns=range(n_pstr_smpls))
        ptd[scn] = ptd[scn].add(rc_mnr_pred.h[prjn_i1:], axis=0)

    # -----------------------------------------------------------------------
    # loop over major constituents

    for cnst in mjr_cnst:

        print(cnst + ":", end="", file=lf)
        t0 = time.time()

        # -------------------------------------------------------------------

        ii = np.where(coef_epch.name == cnst)[0][0]
        lind = coef_epch["aux"]["lind"][ii]

        # -------------------------------------------------------------------

        out_name = out_path + "posterior_samples_" + cnst + ".pickle"
        with open(out_name, "rb") as f:
            pstr = pickle.load(f)

        obs_amp, obs_phs, obs_jyr = (
            pstr["amp_obs"],
            pstr["phs_obs"],
            pstr["jyr_obs"],
        )
        prjn_amp, prjn_phs, prjn_jyr = (
            pstr["amp_prjn"],
            pstr["phs_prjn"],
            pstr["jyr_prjn"],
        )

        if iso_scn is not None:
            prjn_amp = {s: prjn_amp[s] for s in prjn_amp if s in iso_scn}
            prjn_phs = {s: prjn_phs[s] for s in prjn_phs if s in iso_scn}

        # -------------------------------------------------------------------
        # arguments to constituent_realizations() that stay the same every loop

        # obs period
        F, U, V = utide.harmonics.FUV(
            np.atleast_1d(obs_jhr), tref, np.atleast_1d(lind), lat, ngflgs
        )
        obs_Ep = (F * np.exp(1j * (U + V) * 2 * np.pi)).flatten()
        obs_Em = np.conj(obs_Ep)
        base_args_obs = {"obs": [obs_jyr, obs_jhr, obs_Ep, obs_Em]}

        # projection period
        F, U, V = utide.harmonics.FUV(
            np.atleast_1d(prjn_jhr), tref, np.atleast_1d(lind), lat, ngflgs
        )
        prjn_Ep = (F * np.exp(1j * (U + V) * 2 * np.pi)).flatten()
        prjn_Em = np.conj(prjn_Ep)
        base_args_prjn = {
            scn: [prjn_jyr, prjn_jhr, prjn_Ep, prjn_Em] for scn in prjn_amp
        }
        base_args = {**base_args_obs, **base_args_prjn}

        # -------------------------------------------------------------------
        # arguments to constituent_realizations() that change with every loop

        loop_args_obs = {
            "obs": [[obs_amp[:, n], obs_phs[:, n]] for n in range(n_pstr_smpls)]
        }
        loop_args_prjn = {
            scn: [
                [prjn_amp[scn][:, n], prjn_phs[scn][:, n]] for n in range(n_pstr_smpls)
            ]
            for scn in prjn_amp
        }
        loop_args = {**loop_args_obs, **loop_args_prjn}

        # -------------------------------------------------------------------
        # parallelize

        agents = 8
        chunksize = 1

        # loop over each scenario
        for scn in loop_args:
            print(" " + scn, end="", file=lf)
            with Pool(
                processes=agents,
                initializer=pool_initializer,
                initargs=[base_args[scn]],
            ) as pool:
                result = pool.map(constituent_realizations, loop_args[scn], chunksize)
                pool.close()
                pool.join()
            ptd[scn] += np.vstack(result).T

            # cnst_h_rlzns[cnst] = pd.DataFrame(
            #     np.vstack(result).T, index=ptd[scn].index)
            # cnst99[cnst] = cnst_h_rlzns[cnst].groupby(
            #     pd.Grouper(freq='A')).quantile(0.99)#, axis=0)
            # cnst99[cnst].index = cnst99[cnst].index.year + 0.5
            # plt.figure(num=cnst)
            # plt.clf()
            # ax = plt.gca()
            # cnst99[cnst].plot(ax=ax)
            # plt.show()

        # -------------------------------------------------------------------
        # basic loop

        # for scn in loop_args:
        #     end = "\n" if scn == list(loop_args.keys())[-1] else ""
        #     print(" " + scn, end=end, file=lf)
        #     pool_initializer(base_args[scn])
        #     for n, args in enumerate(loop_args[scn]):
        #         mod_td_cnst = constituent_realizations(args)
        #         ptd[scn].iloc[:, n] += mod_td_cnst

        dt = (time.time() - t0) / 60  # minutes
        print(f" ({dt:.1f} min)", end="\n", file=lf)

    # -----------------------------------------------------------------------
    # give each tide prediction units of cm above MHHW by matching MSL
    #   of the tide gauge data (already in cm above MHWW) over the NTDE.

    print("Add constituent to total tide ... ", end="", file=lf)
    t0 = time.time()

    ptd_epch_mn = ptd["obs"].loc["1983":"2001", :].mean(axis=0)
    tg_epch_mn = tg.sl.loc["1983":"2001"].mean()
    for scn in ptd:
        ptd[scn] -= ptd_epch_mn
        ptd[scn] += tg_epch_mn

    dt = time.time() - t0  # seconds
    print(f"complete ({dt:.1f} sec).", end="\n", file=lf)

    # ----------------------------------------------------------------------
    # calculate and save mean and standard deviation of hourly tidal height

    print("Mean and standard deviation of hourly tidal height ... ", end="", file=lf)
    t0 = time.time()

    ptd_summary = {
        scn: pd.concat(
            [ptd[scn].mean(axis=1).rename("mn"), ptd[scn].std(axis=1).rename("sd"),],
            axis=1,
        )
        for scn in ptd
    }

    ptd_summary["acyc"] = {
        "obs": acyc_pred[obs_i1:obs_i2],
        "prjn": acyc_pred[prjn_i1:],
    }

    fname = out_path + "tide_prediction_mn_std.pickle"
    with open(fname, "wb") as f:
        pickle.dump(ptd_summary, f)

    dt = time.time() - t0  # seconds
    print(f"complete ({dt:.1f} sec).", end="\n", file=lf)

    # ----------------------------------------------------------------------
    # calculate and save annual standard deviations of high-pass filtered
    #   differences between probabilistic tidal predictions and observations

    print("Standard deviation of high-passed (pred - obs) ... ", end="", file=lf)
    t0 = time.time()

    pandarallel.initialize(nb_workers=agents, verbose=1)

    wts = lowpass_wts(48)[0]
    res = -ptd["obs"].iloc[:, :].reindex(tg.index).subtract(tg.sl, axis=0)
    res = res.parallel_apply(lambda x: x - np.convolve(x, wts, mode="same"), axis=0)
    std_res = res.groupby(pd.Grouper(freq="A")).apply(lambda x: x.std(axis=0))

    ptl = [2.5, 5, 17, 50, 83, 95, 97.5]
    qtl = [p / 100 for p in ptl]
    std_res_ptl = std_res.quantile(q=qtl, axis=1).T
    std_res_ptl.index = std_res_ptl.index.year
    std_res_ptl.index.name = "year"
    std_res_ptl.columns = ptl
    std_res_ptl.columns.name = "percentile"

    fname = out_path + "residuals_stdev_annual.pickle"
    with open(fname, "wb") as f:
        pickle.dump(std_res_ptl, f)
    dt = (time.time() - t0) / 60  # seconds
    print(f"complete ({dt:.1f} min).", end="\n", file=lf)

    # -----------------------------------------------------------------------
    # calculate and save annual 99th percentile of tidal height;
    # these values are relative to mhhw

    print(
        "Annual 99th percentile of tidal height ... ", end="", file=lf,
    )
    t0 = time.time()

    td99_yr = {
        scn: ptd[scn]
        .groupby(pd.Grouper(freq="A"))
        .parallel_apply(lambda x: x.quantile(0.99))
        .set_index(ptd[scn].index.year.unique())
        for scn in ptd
    }

    fname = out_path + "tide_99th_percentile_annual.pickle"
    with open(fname, "wb") as f:
        pickle.dump(td99_yr, f)

    dt = time.time() - t0  # seconds
    print(f"complete ({dt:.1f} sec).", end="\n", file=lf)

    # -----------------------------------------------------------------------
    # calculate and save monthly 99th percentile of tidal height;
    # these values are relative to mhhw

    print(
        "Monthly 99th percentile of tidal height ... ", end="", file=lf,
    )
    t0 = time.time()

    td99_mo = {
        scn: ptd[scn]
        .groupby(pd.Grouper(freq="MS"))
        .parallel_apply(lambda x: x.quantile(0.99))
        for scn in ptd
    }
    for scn in td99_mo:
        td99_mo[scn].index = td99_mo[scn].index + pd.Timedelta(days=14)

    fname = out_path + "tide_99th_percentile_monthly.pickle"
    with open(fname, "wb") as f:
        pickle.dump(td99_mo, f)

    dt = time.time() - t0  # seconds
    print(f"complete ({dt:.1f} sec).", end="\n", file=lf)

    # -----------------------------------------------------------------------

    if mode == "parallel":
        lf.close()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
