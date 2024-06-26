# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

import os
import sys

# below must be done before importing numpy
os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import shutil
from multiprocessing import Pool
import logging

import numpy as np
import pandas as pd
import xarray as xr
import time
import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import pymc3 as pm
import theano.tensor as tt

import pickle

from constituent_annual_fits import constituent_annual_fits

from data_functions import station_data, initial_tide_fit, plot_gp_dist

sys.path.insert(0, "../htf_projections/")
from support_functions import station_string, LoopInfo

# suppress some 'future warnings' related to numpy indexing in theano
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import theano

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def main():

    # mode = "parallel"
    mode = "series"

    # -----------------------------------------------------------------------

    fname = "../data/stations.pickle"
    stations = pd.read_pickle(fname)

    # -----------------------------------------------------------------------

    # remove stations to skip
    skip = []  # [8771013]
    keep = [n for n in stations.index if stations.loc[n, "id"] not in skip]
    stations = stations.loc[keep]

    # priortize certain stations in the order of completion
    first = ["1612340", "8443970", "8658120", "8545240", "9414290"]
    stations["sort"] = [
        first.index(stations.loc[n, "id"])
        if stations.loc[n, "id"] in first
        else n + len(first)
        for n in stations.index
    ]
    stations = stations.sort_values(by="sort")
    stations.index = range(stations.index.size)

    # only do select stations
    select = ["9410170"]
    keep = [n for n in stations.index if stations.loc[n, "id"] in select]
    stations = stations.loc[keep]

    # -----------------------------------------------------------------------
    # series loop

    if mode == "series":

        lp = LoopInfo(stations.index.size)
        for _, sta in stations.iterrows():
            lp.begin_iteration(sta)
            constituent_gp_regressions([sta, mode])
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
            result = pool.map(constituent_gp_regressions, pool_input, chunksize)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def constituent_gp_regressions(params):  # , iso_cnst=None):

    sta, mode = params[0], params[1]

    pbar = False if mode == "parallel" else True

    # -----------------------------------------------------------------------
    # set up for saving

    sta_str = station_string(sta)

    out_path = "./output/" + sta_str + "/"
    os.makedirs(out_path, exist_ok=True)

    fig_path = "./figures/" + sta_str + "/"
    os.makedirs(fig_path, exist_ok=True)

    # -----------------------------------------------------------------------
    # set up for logging

    log_dir = "./progress/" + sta_str + "/"
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir + "python_log.txt"
    lf = open(log_file, "w") if mode == "parallel" else None

    pm_logger = logging.getLogger("pymc3")
    pm_logger.removeHandler(pm_logger.handlers[0])
    pm_handler = logging.StreamHandler(lf)
    pm_logger.addHandler(pm_handler)

    th_log_file = log_dir + "theano_log.txt"
    thlf = open(th_log_file, "w")
    th_logger = logging.getLogger("theano")
    th_logger.removeHandler(th_logger.handlers[0])
    th_handler = logging.StreamHandler(thlf)
    th_logger.addHandler(th_handler)

    if mode == "parallel":
        print(sta["tool_name"] + " (NOAA ID: " + str(sta["id"]) + ")", file=lf)

    # -----------------------------------------------------------------------
    # load station data ('sl' in cm)
    # also includes stock NOAA tidal hindcast ('td')

    tg = station_data(sta)

    # -----------------------------------------------------------------------
    # initial tide fit to distnguish major/minor constituents
    # adds two tidal height time series to 'tg':
    #   'utd_epch' is tidal heights reconstructed from all tidal constituents
    #         with signal:noise > 2 in fit during 5 year epoch
    #   'utd_epch_mnr' is tidal heights reconstructed from minor tidal
    #         constituents only

    tg, mjr_cnst = initial_tide_fit(sta, tg, out_path, lf)

    # if iso_cnst is not None:
    #     mjr_cnst = [iso_cnst]

    # -----------------------------------------------------------------------
    # solve for major constituents individually in each year
    # adds one tidal height time series to 'tg':
    #   'td2' is tidal heights reconstructed from yearly fits for major
    #         constituents plus 'utd_epch_mnr' plus least squares annual cycle

    tg, annual_fits = constituent_annual_fits(sta, tg, mjr_cnst, out_path, fig_path, lf)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    # GP regressions

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # some constants

    chng_pnt = True  # whether to find a changepoint or not

    jdt_scale = 100.0 * 365.25  # 100 years; use to normalize time vectors
    p1861 = 18.61 * 365.25 / jdt_scale  # 18.61 years in normalized units
    p4425 = 4.425 * 365.25 / jdt_scale  # 4.425 ''
    p9305 = 9.305 * 365.25 / jdt_scale  # 9.305 ''
    p8850 = 8.85 * 365.25 / jdt_scale  # 8.85 ''
    L_trns_fixed = 0.03  # 3 year changepoint transition in normalized units

    N_pstr_smpls = 1000  # number of sample GPs to take from posterior

    # -----------------------------------------------------------------------
    # julian dates; normalize for GP regression

    jdt_ref = mdates.date2num(pd.Timestamp("1900-01-01"))  # reference time
    jdt_max = mdates.date2num(pd.Timestamp("2101-01-01"))  # max time

    jyr_obs = mdates.date2num(
        [
            pd.Timestamp(str(y) + "-07-01").to_pydatetime()
            for y in annual_fits["amp"].index
        ]
    )
    jyr_pred = mdates.date2num(
        [
            pd.Timestamp(str(y) + "-07-01").to_pydatetime()
            for y in range(annual_fits["amp"].index[0], 2101)
        ]
    )
    jyr_pred_obs = jyr_pred[jyr_pred <= jyr_obs[-1]]

    jyr_obs_nrm = (jyr_obs - jdt_ref) / jdt_scale
    jyr_pred_nrm = (jyr_pred - jdt_ref) / jdt_scale

    # -----------------------------------------------------------------------
    # normalize mean sea level

    msl_obs = annual_fits["msl"].values.astype(float)

    msl_mn = np.nanmean(msl_obs)
    msl_sd = np.nanstd(msl_obs)
    msl_obs_nrm = (msl_obs - msl_mn) / msl_sd

    # -----------------------------------------------------------------------
    # function for smoothing transition across changepoint

    def logistic(x, rt, x0):
        # rt is the slope of transition, x0 is the location
        return pm.math.invlogit(rt * (x - x0))

    # -----------------------------------------------------------------------
    # custom mean function to use MSL time series in GP regression

    class MslMean(pm.gp.mean.Mean):
        def __init__(self, msl):
            self.msl = msl

        def __call__(self, X):
            return self.msl

    # -----------------------------------------------------------------------
    # loop over constituents and perform GP regression

    for cnst in mjr_cnst:

        # try:

        # if analysis already completed, then skip
        fig_name_1 = fig_path + "gp_regression_" + cnst + ".pdf"
        if os.path.exists(fig_name_1):
            print("\n" + cnst + " already completed.", file=lf)
            continue

        print("\n *** " + cnst + " ***", file=lf)

        # -------------------------------------------------------------------
        # normalize constituent time series and remove missing values

        amp = annual_fits["amp"][cnst].values.astype(float)
        phs = annual_fits["phs"][cnst].values.astype(float)
        z = ~np.isnan(phs) & ~np.isnan(amp)
        amp = amp[z]
        phs = phs[z]

        amp_mn = np.mean(amp)
        amp_sd = np.std(amp)
        amp_nrm = (amp - amp_mn) / amp_sd

        phs_mn = np.mean(phs)
        phs_sd = np.std(phs)
        phs_nrm = (phs - phs_mn) / phs_sd

        jyr = jyr_obs[z]
        jyr_nrm = jyr_obs_nrm[z]

        msl_nrm = msl_obs_nrm[z]

        # -------------------------------------------------------------------
        # create GP regression model for tidal constituent

        with pm.Model() as model:

            # ---------------------------------------------------------------
            # CHANGEPOINT
            # ---------------------------------------------------------------

            # location of changepoint
            x0 = pm.Uniform("x0", lower=0.0, upper=1.0)

            # ---------------------------------------------------------------
            # AMPLITUDE
            # ---------------------------------------------------------------

            # msl contribution
            amp_K_msl = pm.Normal("amp_K_msl", mu=0.0, sd=0.5)
            amp_msl = amp_K_msl * msl_nrm
            amp_gp_msl = pm.gp.Marginal(
                mean_func=MslMean(amp_msl), cov_func=pm.gp.cov.Constant(0)
            )

            # ---------------------------------------------------------------

            # long-term trend before changepoint
            amp_K_trnd0 = pm.Gamma("amp_K_trnd0", alpha=2.0, beta=2.0)
            amp_c_trnd0 = pm.Normal("amp_c_trnd0", mu=0.33, sd=1.0)
            amp_cov_trnd0 = amp_K_trnd0 ** 2 * pm.gp.cov.Linear(1, amp_c_trnd0)

            # long-term trend after changepoint
            amp_K_trnd1 = pm.Gamma("amp_K_trnd1", alpha=2.0, beta=2.0)
            amp_c_trnd1 = pm.Normal("amp_c_trnd1", mu=0.67, sd=1.0)
            amp_cov_trnd1 = amp_K_trnd1 ** 2 * pm.gp.cov.Linear(1, amp_c_trnd1)

            # construct changepoint cov and gp:
            # L_trns is duration of transition in years/100
            # if not fitting transition length, use L_trns_fixed from above
            amp_sc_cov0 = pm.gp.cov.ScaledCov(
                1, amp_cov_trnd0, logistic, (-1 / L_trns_fixed, x0)
            )
            amp_sc_cov1 = pm.gp.cov.ScaledCov(
                1, amp_cov_trnd1, logistic, (1 / L_trns_fixed, x0)
            )
            amp_cov_trnd = amp_sc_cov0 + amp_sc_cov1
            amp_gp_trnd = pm.gp.Marginal(cov_func=amp_cov_trnd)

            # ---------------------------------------------------------------

            # short to medium variation:
            # number of up-crossings over unit length is related to
            # characteristic length scales by N_ux = 1/(2*pi*L)
            amp_K_mdm = pm.Gamma("amp_K_mdm", alpha=2.0, beta=5.0)
            amp_L_mdm = pm.Gamma("amp_L_mdm", alpha=2.0, beta=100.0)
            amp_cov_mdm = amp_K_mdm ** 2 * pm.gp.cov.Matern52(1, amp_L_mdm)
            amp_gp_mdm = pm.gp.Marginal(cov_func=amp_cov_mdm)

            # ---------------------------------------------------------------

            # 4.425 year periodic
            amp_K_p4425 = pm.Gamma("amp_K_p4425", alpha=2.0, beta=5.0)
            amp_cov_p4425 = amp_K_p4425 ** 2 * pm.gp.cov.Periodic(1, p4425, 1)
            amp_gp_p4425 = pm.gp.Marginal(cov_func=amp_cov_p4425)

            # always fit 18.61 periodicity even if not lunar, because
            #   non-linear interactions can produce 18.61 in solar.
            amp_K_p1861 = pm.Gamma("amp_K_p1861", alpha=2.0, beta=5.0)
            amp_cov_p1861 = amp_K_p1861 ** 2 * pm.gp.cov.Periodic(1, p1861, 1)
            amp_gp_p1861 = pm.gp.Marginal(cov_func=amp_cov_p1861)

            # 9.305 year periodic
            amp_K_p9305 = pm.Gamma("amp_K_p9305", alpha=2.0, beta=5.0)
            amp_cov_p9305 = amp_K_p9305 ** 2 * pm.gp.cov.Periodic(1, p9305, 1)
            amp_gp_p9305 = pm.gp.Marginal(cov_func=amp_cov_p9305)

            # 8.85 year periodic
            amp_K_p8850 = pm.Gamma("amp_K_p8850", alpha=2.0, beta=5.0)
            amp_cov_p8850 = amp_K_p8850 ** 2 * pm.gp.cov.Periodic(1, p8850, 1)
            amp_gp_p8850 = pm.gp.Marginal(cov_func=amp_cov_p8850)

            # ---------------------------------------------------------------

            # sum the component GPs
            amp_gp_notmsl = (
                amp_gp_trnd
                + amp_gp_mdm
                + amp_gp_p1861
                + amp_gp_p4425
                + amp_gp_p9305
                + amp_gp_p8850
            )
            amp_gp = amp_gp_msl + amp_gp_notmsl

            # white noise variance
            amp_whtns = pm.Gamma("amp_whtns", alpha=2.0, beta=5.0)
            amp_ = amp_gp.marginal_likelihood(
                "amp", X=jyr_nrm[:, None], y=amp_nrm, noise=amp_whtns
            )

            # ---------------------------------------------------------------
            # PHASE
            # ---------------------------------------------------------------

            # msl contribution
            phs_K_msl = pm.Normal("phs_K_msl", mu=0.0, sd=0.5)
            phs_msl = phs_K_msl * msl_nrm
            phs_gp_msl = pm.gp.Marginal(
                mean_func=MslMean(phs_msl), cov_func=pm.gp.cov.Constant(0)
            )

            # ---------------------------------------------------------------

            # long-term trend before changepoint
            phs_K_trnd0 = pm.Gamma("phs_K_trnd0", alpha=2.0, beta=2.0)
            phs_c_trnd0 = pm.Normal("phs_c_trnd0", mu=0.33, sd=1.0)
            phs_cov_trnd0 = phs_K_trnd0 ** 2 * pm.gp.cov.Linear(1, phs_c_trnd0)

            # long-term trend after changepoint
            phs_K_trnd1 = pm.Gamma("phs_K_trnd1", alpha=2.0, beta=2.0)
            phs_c_trnd1 = pm.Normal("phs_c_trnd1", mu=0.67, sd=1.0)
            phs_cov_trnd1 = phs_K_trnd1 ** 2 * pm.gp.cov.Linear(1, phs_c_trnd1)

            # construct changepoint cov and gp:
            # L_trns is duration of transition in years/100
            # if not fitting transition length, use L_trns_fixed from above
            phs_sc_cov0 = pm.gp.cov.ScaledCov(
                1, phs_cov_trnd0, logistic, (-1 / L_trns_fixed, x0)
            )
            phs_sc_cov1 = pm.gp.cov.ScaledCov(
                1, phs_cov_trnd1, logistic, (1 / L_trns_fixed, x0)
            )
            phs_cov_trnd = phs_sc_cov0 + phs_sc_cov1
            phs_gp_trnd = pm.gp.Marginal(cov_func=phs_cov_trnd)

            # ---------------------------------------------------------------

            # short to medium variation:
            # number of up-crossings over unit length is related to
            # characteristic length scales by N_ux = 1/(2*pi*L)
            phs_K_mdm = pm.Gamma("phs_K_mdm", alpha=2.0, beta=5.0)
            phs_L_mdm = pm.Gamma("phs_L_mdm", alpha=2.0, beta=100.0)
            phs_cov_mdm = phs_K_mdm ** 2 * pm.gp.cov.Matern52(1, phs_L_mdm)
            phs_gp_mdm = pm.gp.Marginal(cov_func=phs_cov_mdm)

            # ---------------------------------------------------------------

            # 4.425 year periodic
            phs_K_p4425 = pm.Gamma("phs_K_p4425", alpha=2.0, beta=5.0)
            phs_cov_p4425 = phs_K_p4425 ** 2 * pm.gp.cov.Periodic(1, p4425, 1)
            phs_gp_p4425 = pm.gp.Marginal(cov_func=phs_cov_p4425)

            # always fit 18.61 periodicity even if not lunar, because
            #   non-linear interactions can produce 18.61 in solar.
            phs_K_p1861 = pm.Gamma("phs_K_p1861", alpha=2.0, beta=5.0)
            phs_cov_p1861 = phs_K_p1861 ** 2 * pm.gp.cov.Periodic(1, p1861, 1)
            phs_gp_p1861 = pm.gp.Marginal(cov_func=phs_cov_p1861)

            # 9.305 year periodic
            phs_K_p9305 = pm.Gamma("phs_K_p9305", alpha=2.0, beta=5.0)
            phs_cov_p9305 = phs_K_p9305 ** 2 * pm.gp.cov.Periodic(1, p9305, 1)
            phs_gp_p9305 = pm.gp.Marginal(cov_func=phs_cov_p9305)

            # 8.85 year periodic
            phs_K_p8850 = pm.Gamma("phs_K_p8850", alpha=2.0, beta=5.0)
            phs_cov_p8850 = phs_K_p8850 ** 2 * pm.gp.cov.Periodic(1, p8850, 1)
            phs_gp_p8850 = pm.gp.Marginal(cov_func=phs_cov_p8850)

            # ---------------------------------------------------------------

            # sum the component GPs
            phs_gp_notmsl = (
                phs_gp_trnd
                + phs_gp_mdm
                + phs_gp_p1861
                + phs_gp_p4425
                + phs_gp_p9305
                + phs_gp_p8850
            )
            phs_gp = phs_gp_msl + phs_gp_notmsl

            # white noise variance
            phs_whtns = pm.Gamma("phs_whtns", alpha=2.0, beta=5.0)
            phs_ = phs_gp.marginal_likelihood(
                "phs", X=jyr_nrm[:, None], y=phs_nrm, noise=phs_whtns
            )

            # ---------------------------------------------------------------
            # SAMPLE

            trace = pm.sample(
                500,
                chains=2,
                cores=1 if mode == "parallel" else 2,
                tune=1000,
                target_accept=0.9,
                progressbar=pbar,
            )

        # -------------------------------------------------------------------
        # SAMPLE POSTERIOR

        with model:
            # note that pred_noise=True in the below adds the white noise
            # to the posterior samples, and thus it does not need to be
            # added in later
            amp_notmsl = amp_gp_notmsl.conditional(
                "amp_notmsl",
                jyr_pred_nrm[:, None],
                pred_noise=True,
                given={
                    "X": jyr_nrm[:, None],
                    "y": amp_nrm,
                    "noise": amp_whtns,
                    "gp": amp_gp,
                },
            )
            phs_notmsl = phs_gp_notmsl.conditional(
                "phs_notmsl",
                jyr_pred_nrm[:, None],
                pred_noise=True,
                given={
                    "X": jyr_nrm[:, None],
                    "y": phs_nrm,
                    "noise": phs_whtns,
                    "gp": phs_gp,
                },
            )
            amp_trnd = amp_gp_trnd.conditional(
                "amp_trnd",
                jyr_pred_nrm[:, None],
                pred_noise=False,
                given={
                    "X": jyr_nrm[:, None],
                    "y": amp_nrm,
                    "noise": amp_whtns,
                    "gp": amp_gp,
                },
            )
            phs_trnd = phs_gp_trnd.conditional(
                "phs_trnd",
                jyr_pred_nrm[:, None],
                pred_noise=False,
                given={
                    "X": jyr_nrm[:, None],
                    "y": phs_nrm,
                    "noise": phs_whtns,
                    "gp": phs_gp,
                },
            )
            pstr_vars = [
                amp_notmsl,
                phs_notmsl,
                amp_trnd,
                phs_trnd,
                amp_K_msl,
                phs_K_msl,
            ]
            pstr = pm.sample_posterior_predictive(
                trace,
                samples=N_pstr_smpls,
                vars=pstr_vars,
                progressbar=pbar,
            )

        # -------------------------------------------------------------------
        # return to physical units

        pstr_amp_msl_factor = pstr["amp_K_msl"]
        pstr_phs_msl_factor = pstr["phs_K_msl"]
        pstr_amp_notmsl = pstr["amp_notmsl"].T * amp_sd + amp_mn
        pstr_phs_notmsl = pstr["phs_notmsl"].T * phs_sd + phs_mn
        pstr_amp_trnd = pstr["amp_trnd"].T * amp_sd
        pstr_phs_trnd = pstr["phs_trnd"].T * phs_sd
        cpnt = trace["x0"] * jdt_scale + jdt_ref

        # -------------------------------------------------------------------
        # load GP annual means

        sta_path = "../gp_monthly/output/" + sta_str + "/"
        fname = sta_path + "gp_annual.pickle"
        with open(fname, "rb") as f:
            msl_gp = pickle.load(f)

        msl_gp_yr = np.arange(1890, 2110, 1)  # years used in gp_monthly_means.py
        msl_gp_jyr = mdates.date2num(
            [pd.Timestamp(str(int(y)) + "-07-01").to_pydatetime() for y in msl_gp_yr]
        )
        msl_gp = pd.DataFrame(msl_gp, index=msl_gp_jyr)
        msl_gp = msl_gp.iloc[:, :N_pstr_smpls]

        # -------------------------------------------------------------------
        # reconstruct contribution from MSL during observed record

        # get normalized annual means from observed linear trend
        tg.trnd -= tg.trnd.mean()
        tg.trnd += tg.sl.mean()
        msl_trnd = np.array(
            [tg.trnd.loc[str(y)].mean() for y in annual_fits["amp"].index]
        )[z]
        msl_trnd_pred_obs = np.interp(jyr_pred_obs, jyr, msl_trnd)
        msl_trnd_pred_obs -= msl_mn
        msl_trnd_pred_obs /= msl_sd

        # intitialize a dataframe with annual mean GPs plus observed trend
        msl_pred_obs = pd.DataFrame(
            msl_gp.loc[jyr_pred_obs].values / msl_sd + msl_trnd_pred_obs[:, None],
            index=jyr_pred_obs,
        )

        # replace years with sufficient observations with observed annual means
        # in all columns
        for y in jyr:
            msl_pred_obs.loc[y, :] = msl_nrm[jyr == y]

        # get estimated contributions of mean sea level to amp and phs
        pstr_amp_obs_msl = amp_sd * msl_pred_obs.values * pstr_amp_msl_factor[None, :]
        pstr_phs_obs_msl = phs_sd * msl_pred_obs.values * pstr_phs_msl_factor[None, :]

        # add contributions from msl to contributions from other GPs
        pstr_amp_obs = pstr_amp_obs_msl + pstr_amp_notmsl[: jyr_pred_obs.size]
        pstr_phs_obs = pstr_phs_obs_msl + pstr_phs_notmsl[: jyr_pred_obs.size]

        # -------------------------------------------------------------------
        # -------------------------------------------------------------------
        # project contribution from future MSL rise

        msl2000 = tg.sl.loc["1991":"2009"].mean()

        # -------------------------------------------------------------------
        # NOAA scenarios

        scn_nm = ["low", "int_low", "int", "int_high", "high"]

        scn_nm_in_file = [
            "rsl_total_" + "".join([c for c in nm.title() if c != "_"]) for nm in scn_nm
        ]
        scn_nm_in_file.extend(
            [
                "rsl_oceandynamics_" + "".join([c for c in nm.title() if c != "_"])
                for nm in scn_nm
            ]
        )

        scn_nm.extend([nm + "_od" for nm in scn_nm])

        fname = "../data/slr_scenarios/TR_local_projections.nc"
        p = xr.open_dataset(fname)

        prjn = (
            p.sel(tg=sta["scn_name"], percentiles=50)
            .to_pandas()[scn_nm_in_file]
            .loc[2020:2100]
        )
        prjn.columns = scn_nm
        prjn["noMSLR"] = 0
        prjn /= 10  # cm
        prjn += msl2000

        idx = prjn.index

        # -------------------------------------------------------------------

        msl_gp.index = msl_gp_yr.astype(int)

        pstr_amp_prjn = {}
        pstr_phs_prjn = {}
        for s in prjn.columns:

            msl_prjns = pd.DataFrame(msl_gp.loc[idx, :N_pstr_smpls], index=idx)
            msl_prjns = msl_prjns.add(prjn[s], axis=0)
            msl_prjns_nrm = (msl_prjns - msl_mn) / msl_sd

            pstr_amp_prjn_msl = (
                amp_sd * msl_prjns_nrm.values * pstr_amp_msl_factor[None, :]
            )
            pstr_phs_prjn_msl = (
                phs_sd * msl_prjns_nrm.values * pstr_phs_msl_factor[None, :]
            )

            pstr_amp_prjn[s] = pstr_amp_prjn_msl + pstr_amp_notmsl[-idx.size :]
            pstr_phs_prjn[s] = pstr_phs_prjn_msl + pstr_phs_notmsl[-idx.size :]

            amp_trnd_to_rmv = pstr_amp_trnd[-idx.size :] - pstr_amp_trnd[-idx.size]
            pstr_amp_prjn[s + "_noTrnd"] = pstr_amp_prjn[s] - amp_trnd_to_rmv

            phs_trnd_to_rmv = pstr_phs_trnd[-idx.size :] - pstr_phs_trnd[-idx.size]
            pstr_phs_prjn[s + "_noTrnd"] = pstr_phs_prjn[s] - phs_trnd_to_rmv

            # if s == "noMSLR_noTrnd":
            #     raise

        # -------------------------------------------------------------------
        # save results

        percentiles = [0.5, 2.5, 5.0, 17.0, 50.0, 83.0, 95.0, 97.5, 99.5]
        params = [p for p in trace.varnames if p[-1] != "_"]
        param_ci = {
            p: [np.percentile(trace[p], pctl) for pctl in percentiles] for p in params
        }

        posterior_samples = {
            "amp_obs": pstr_amp_obs,
            "phs_obs": pstr_phs_obs,
            "jyr_obs": jyr_pred_obs,
            "amp_prjn": pstr_amp_prjn,
            "phs_prjn": pstr_phs_prjn,
            "jyr_prjn": jyr_pred[-idx.size :],
            "amp_msl_factor": pstr_amp_msl_factor,
            "phs_msl_factor": pstr_phs_msl_factor,
            "param_ci": param_ci,
            "normalization": {
                "amp_mn": amp_mn,
                "amp_sd": amp_sd,
                "phs_mn": phs_mn,
                "phs_sd": phs_sd,
                "msl_mn": msl_mn,
                "msl_sd": msl_sd,
                "jdt_scale": jdt_scale,
                "jdt_ref": jdt_ref,
            },
        }

        out_name = out_path + "posterior_samples_" + cnst + ".pickle"
        with open(out_name, "wb") as f:
            pickle.dump(posterior_samples, f)

        # -------------------------------------------------------------------
        # posterior figure 1

        col = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        cpnt_col = col[9]

        fig = plt.figure(num="fit", figsize=[10, 8])
        plt.clf()

        scn = "int_od"
        scn_str = "Intermediate sterodynamic MSLR + non-climatic secular trend"

        ax = plt.subplot(211)
        if chng_pnt:
            mx_a = (
                max([np.percentile(pstr_amp_prjn[scn], 99.0), amp.max()])
                + 1 * amp.std()
            )
            mn_a = (
                min([np.percentile(pstr_amp_prjn[scn], 1.0), amp.min()]) - 1 * amp.std()
            )
            for ptl in range(15, 90, 10):
                ptl_lo = np.percentile(cpnt, ptl - 5)
                ptl_hi = np.percentile(cpnt, ptl + 5)
                alpha = 0.45 - np.abs(ptl - 50) / 100
                h = plt.fill_between(
                    [ptl_lo, ptl_hi],
                    y1=[mn_a, mn_a],
                    y2=[mx_a, mx_a],
                    color=cpnt_col,
                    alpha=alpha,
                    lw=0,
                )
                if ptl == 45:
                    h_cp = h
            cp50 = np.percentile(cpnt, 50)
            plt.plot([cp50, cp50], [mn_a, mx_a], lw=2, color=cpnt_col)
        ax_gp, h_pctl_prjn, h_smpl_prjn = plot_gp_dist(
            jyr_pred[-idx.size :], pstr_amp_prjn[scn], ax=ax, palette="Greys"
        )
        ax_gp, h_pctl, h_smpl = plot_gp_dist(jyr_pred_obs, pstr_amp_obs, ax=ax)
        h_amp = plt.plot(jyr, amp, ".k")[0]
        ax.xaxis_date()
        plt.xlim([datetime.date(1920, 1, 1), datetime.date(2100, 1, 1)])
        plt.ylim([mn_a, mx_a])
        plt.ylabel("cm")
        plt.title(sta["tool_name"] + ": " + cnst + " amplitude")
        handles = [h_amp, h_pctl, h_pctl_prjn, h_cp]
        labels = [
            "annual fitted " + cnst + " amplitudes",
            "predicted amplitudes (observed years)",
            "projected amplitudes (" + scn_str + ")",
            "changepoint estimation",
        ]
        if np.median(pstr_amp_prjn[scn][:, -10:]) > np.median(pstr_amp_obs[:, :10]):
            leg_loc = "upper left"
        else:
            leg_loc = "lower left"
        plt.legend(handles=handles, labels=labels, loc=leg_loc)

        ax = plt.subplot(212)
        if chng_pnt:
            mx_p = (
                max([np.percentile(pstr_phs_prjn[scn], 99.0), phs.max()])
                + 1 * phs.std()
            )
            mn_p = (
                min([np.percentile(pstr_phs_prjn[scn], 1.0), phs.min()]) - 1 * phs.std()
            )
            for ptl in range(15, 90, 10):
                ptl_lo = np.percentile(cpnt, ptl - 5)
                ptl_hi = np.percentile(cpnt, ptl + 5)
                alpha = 0.45 - np.abs(ptl - 50) / 100
                h = plt.fill_between(
                    [ptl_lo, ptl_hi],
                    y1=[mn_p, mn_p],
                    y2=[mx_p, mx_p],
                    color=cpnt_col,
                    alpha=alpha,
                    lw=0,
                )
                if ptl == 45:
                    h_cp = h
            cp50 = np.percentile(cpnt, 50)
            plt.plot([cp50, cp50], [mn_p, mx_p], lw=2, color=cpnt_col)
        ax_gp, h_pctl_prjn, h_smpl_prjn = plot_gp_dist(
            jyr_pred[-idx.size :], pstr_phs_prjn[scn], ax=ax, palette="Greys"
        )
        ax_gp, h_pctl, h_smpl = plot_gp_dist(jyr_pred_obs, pstr_phs_obs, ax=ax)
        h_phs = plt.plot(jyr, phs, ".k")[0]
        ax.xaxis_date()
        plt.xlim([datetime.date(1920, 1, 1), datetime.date(2100, 1, 1)])
        plt.ylim([mn_p, mx_p])
        plt.ylabel("degrees")
        plt.title(sta["tool_name"] + ": " + cnst + " phase (Greenwich lag)")
        handles = [h_phs, h_pctl, h_pctl_prjn, h_cp]
        labels = [
            "annual fitted " + cnst + " phases",
            "predicted phases (observed years)",
            "projected phases (" + scn_str + ")",
            "changepoint estimation",
        ]
        if np.median(pstr_phs_prjn[scn][:, -10:]) > np.median(pstr_phs_obs[:, :10]):
            leg_loc = "upper left"
        else:
            leg_loc = "lower left"
        plt.legend(handles=handles, labels=labels, loc=leg_loc)

        plt.tight_layout()
        # plt.show()

        fig.savefig(fig_name_1)

        # -------------------------------------------------------------------
        # posterior figure 2

        col = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        cpnt_col = col[9]

        fig = plt.figure(num="fit", figsize=[10, 8])
        plt.clf()

        scn = "noMSLR_noTrnd"

        ax = plt.subplot(211)
        if chng_pnt:
            for ptl in range(15, 90, 10):
                ptl_lo = np.percentile(cpnt, ptl - 5)
                ptl_hi = np.percentile(cpnt, ptl + 5)
                alpha = 0.45 - np.abs(ptl - 50) / 100
                h = plt.fill_between(
                    [ptl_lo, ptl_hi],
                    y1=[mn_a, mn_a],
                    y2=[mx_a, mx_a],
                    color=cpnt_col,
                    alpha=alpha,
                    lw=0,
                )
                if ptl == 45:
                    h_cp = h
            cp50 = np.percentile(cpnt, 50)
            plt.plot([cp50, cp50], [mn_a, mx_a], lw=2, color=cpnt_col)
        ax_gp, h_pctl_prjn, h_smpl_prjn = plot_gp_dist(
            jyr_pred[-idx.size :], pstr_amp_prjn[scn], ax=ax, palette="Greys"
        )
        ax_gp, h_pctl, h_smpl = plot_gp_dist(jyr_pred_obs, pstr_amp_obs, ax=ax)
        h_amp = plt.plot(jyr, amp, ".k")[0]
        ax.xaxis_date()
        plt.xlim([datetime.date(1920, 1, 1), datetime.date(2100, 1, 1)])
        plt.ylim([mn_a, mx_a])
        plt.ylabel("cm")
        plt.title(sta["tool_name"] + ": " + cnst + " amplitude")
        handles = [h_amp, h_pctl, h_pctl_prjn, h_cp]
        labels = [
            "annual fitted " + cnst + " amplitudes",
            "predicted amplitudes (observed years)",
            "projected amplitudes (no MSLR, no non-climatic secular trend)",
            "changepoint estimation",
        ]
        if np.median(pstr_amp_prjn[scn][:, -10:]) > np.median(pstr_amp_obs[:, :10]):
            leg_loc = "upper left"
        else:
            leg_loc = "lower left"
        plt.legend(handles=handles, labels=labels, loc=leg_loc)

        ax = plt.subplot(212)
        if chng_pnt:
            for ptl in range(15, 90, 10):
                ptl_lo = np.percentile(cpnt, ptl - 5)
                ptl_hi = np.percentile(cpnt, ptl + 5)
                alpha = 0.45 - np.abs(ptl - 50) / 100
                h = plt.fill_between(
                    [ptl_lo, ptl_hi],
                    y1=[mn_p, mn_p],
                    y2=[mx_p, mx_p],
                    color=cpnt_col,
                    alpha=alpha,
                    lw=0,
                )
                if ptl == 45:
                    h_cp = h
            cp50 = np.percentile(cpnt, 50)
            plt.plot([cp50, cp50], [mn_p, mx_p], lw=2, color=cpnt_col)
        ax_gp, h_pctl_prjn, h_smpl_prjn = plot_gp_dist(
            jyr_pred[-idx.size :], pstr_phs_prjn[scn], ax=ax, palette="Greys"
        )
        ax_gp, h_pctl, h_smpl = plot_gp_dist(jyr_pred_obs, pstr_phs_obs, ax=ax)
        h_phs = plt.plot(jyr, phs, ".k")[0]
        ax.xaxis_date()
        plt.xlim([datetime.date(1920, 1, 1), datetime.date(2100, 1, 1)])
        plt.ylim([mn_p, mx_p])
        plt.ylabel("degrees")
        plt.title(sta["tool_name"] + ": " + cnst + " phase (Greenwich lag)")
        handles = [h_phs, h_pctl, h_pctl_prjn, h_cp]
        labels = [
            "annual fitted " + cnst + " phases",
            "predicted phases (observed years)",
            "projected phases (no MSLR, no non-climatic secular trend)",
            "changepoint estimation",
        ]
        if np.median(pstr_phs_prjn[scn][:, -10:]) > np.median(pstr_phs_obs[:, :10]):
            leg_loc = "upper left"
        else:
            leg_loc = "lower left"
        plt.legend(handles=handles, labels=labels, loc=leg_loc)

        plt.tight_layout()
        # plt.show()

        fig_name_2 = fig_path + "gp_regression_" + cnst + "_noMSLR_noTrnd.pdf"
        fig.savefig(fig_name_2)

        # import sys; sys.exit()

        if mode == "parallel":
            print("\n   Complete.", file=lf)

        # -----------------------------------------------------------------------

        # except:

        # print("\n   An error occurred.", file=lf)

        # error_file = log_dir + "error_occurred_" + cnst + ".txt"
        # ef = open(error_file, "w")
        # print("An error occurred.", file=ef)
        # ef.close()

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    # close log file
    if lf is not None:
        lf.close()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
