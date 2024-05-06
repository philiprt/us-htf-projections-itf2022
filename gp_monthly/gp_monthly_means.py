# ---------------------------------------------------------------------------

import os

# below must be done before importing numpy
os.environ["OMP_NUM_THREADS"] = "5"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import shutil
from multiprocessing import Pool
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

import pymc3 as pm
import theano.tensor as tt
import theano

import sys

sys.path.insert(0, "../htf_projections/")
from support_functions import station_string
from tide_gauge_data import tide_gauge_data

from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
logging.getLogger().setLevel(logging.CRITICAL)

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
    skip = []
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
    select = ["1617433"]
    keep = [n for n in stations.index if stations.loc[n, "id"] in select]
    stations = stations.loc[keep]

    # -----------------------------------------------------------------------
    # series loop

    if mode == "series":

        Nsta = stations.index.size
        for idx, sta in stations.iterrows():
            gp_monthly_means([sta, mode, idx + 1, Nsta])

    # -----------------------------------------------------------------------
    # parallel processing

    elif mode == "parallel":

        try:
            shutil.rmtree("./progress/")
        except:
            pass
        os.makedirs("./progress/")

        pool_input = [[sta, mode, idx, None] for idx, sta in stations.iterrows()]

        agents = 23
        chunksize = 1
        with Pool(processes=agents) as pool:
            result = pool.map(gp_monthly_means, pool_input, chunksize)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def gp_monthly_means(params):

    sta, mode, idx, Nsta = params[0], params[1], params[2], params[3]

    # -----------------------------------------------------------------------

    # station ouput directory
    sta_str = station_string(sta)
    sta_path = "./output/" + sta_str + "/"
    os.makedirs(sta_path, exist_ok=True)

    # figure directory and figure name
    fig_path = "./figures/"
    fig_file = fig_path + "gpmo_" + sta_str + ".png"

    # if analysis already completed, then skip
    if os.path.exists(fig_file):
        return

    # -----------------------------------------------------------------------
    # setup for logging

    log_dir = "./progress/" + sta_str + "/"
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir + "python_log.txt"

    th_log_file = log_dir + "theano_log.txt"
    thlf = open(th_log_file, "w")
    th_logger = logging.getLogger("theano")
    th_logger.removeHandler(th_logger.handlers[0])
    th_handler = logging.StreamHandler(thlf)
    th_logger.addHandler(th_handler)

    if mode == "parallel":

        lf = open(log_file, "w")

        pm_logger = logging.getLogger("pymc3")
        pm_logger.removeHandler(pm_logger.handlers[0])
        pm_handler = logging.StreamHandler(lf)
        pm_logger.addHandler(pm_handler)

        print(sta["tool_name"] + " (NOAA ID: " + str(sta["id"]) + ")", file=lf)

    elif mode == "series":
        if os.path.exists(log_file):
            return
        prnt = (
            "\nStation "
            + str(sta["id"])
            + ": "
            + sta["tool_name"]
            + " ("
            + str(idx)
            + " of "
            + str(Nsta)
            + ")"
        )
        print(prnt)
        lf = open(log_file, "w")
        print(prnt, file=lf)
        lf.close()

    # -----------------------------------------------------------------------

    try:

        # -----------------------------------------------------------------------

        tg = tide_gauge_data(sta)
        if tg is None:
            return

        # remove trend and annual cycle
        tg.res -= tg.trnd + tg.acyc

        # -----------------------------------------------------------------------

        # drop years which have more than 75% hours missing data
        tg = tg.groupby(pd.Grouper(freq="A")).apply(
            lambda x: None if x.res.dropna().count() < 0.75 * 24 * 365 else x
        )
        try:
            tg.index = tg.index.droplevel(0)
        except:
            pass

        # drop months which have more than 10 days missing data
        tg = tg.groupby(pd.Grouper(freq="M")).apply(
            lambda x: None if x.res.dropna().count() < 24 * 10 else x
        )
        try:
            tg.index = tg.index.droplevel(0)
        except:
            pass

        # -----------------------------------------------------------------------

        ha = tg.res.dropna()

        # -----------------------------------------------------------------------

        amn = ha.groupby(pd.Grouper(freq="A")).mean()
        amn.dropna(inplace=True)
        amn_t = amn.index.year + 0.5

        amn = amn.values
        amn_t = amn_t.values

        amn_mn = amn.mean()
        amn_sd = amn.std()
        amn_nrm = (amn - amn_mn) / amn_sd

        amn_t_pred = np.arange(1890, 2110, 1) + 0.5

        amn_t_nrm = (amn_t - 1900) / 100
        amn_t_pred_nrm = ((amn_t_pred - 1900) / 100)[:, None]

        # -----------------------------------------------------------------------

        momn = ha.groupby(pd.Grouper(freq="M")).mean()
        momn.dropna(inplace=True)
        momn_t = momn.index.year + momn.index.month / 12 - 1 / 24

        momn = momn.values
        momn_t = momn_t.values

        amni = np.interp(momn_t, amn_t, amn)
        momn -= amni

        # fit monthly gp to lastN months
        lastN = 240

        momn_mn = momn[-lastN:].mean()
        momn_sd = momn[-lastN:].std()
        momn_nrm = (momn - momn_mn) / momn_sd

        momn_t_pred = np.arange(2000, 2101, 1 / 12) + 1 / 24

        momn_t_nrm = (momn_t - 1900) / 100
        momn_t_pred_nrm = ((momn_t_pred - 1900) / 100)[:, None]

        momn_nrm = momn_nrm[-lastN:]
        momn_t_nrm = momn_t_nrm[-lastN:]

        # -----------------------------------------------------------------------

        draws = 500
        chains = 2
        cores = 2 if mode == "series" else 1
        tune = 2000
        target_accept = 0.95
        pbar = True if mode == "series" else False

        posterior_samples = 1000

        # -----------------------------------------------------------------------
        # -----------------------------------------------------------------------

        with pm.Model() as a_model:

            # three random numbers that sum to one; used for variances of gp components;
            # they sum to one because data normalized to have std=1
            A2 = pm.Dirichlet("A2", a=np.array([1, 1, 1]))

            L0 = pm.Gamma("L0", alpha=5, beta=500)
            cov0 = A2[0] * pm.gp.cov.ExpQuad(1, L0)
            gp0 = pm.gp.Marginal(cov_func=cov0)

            L1 = pm.Gamma("L1", alpha=5, beta=200)
            cov1 = A2[1] * pm.gp.cov.ExpQuad(1, L1)
            gp1 = pm.gp.Marginal(cov_func=cov1)

            gp = gp0 + gp1

            amn_ = gp.marginal_likelihood(
                "amn_", X=amn_t_nrm[:, None], y=amn_nrm, noise=tt.sqrt(A2[2])
            )

            a_trace = pm.sample(
                draws,
                chains=chains,
                cores=cores,
                tune=tune,
                target_accept=target_accept,
                progressbar=pbar,
            )

            amn_pred = gp.conditional("amn_pred", amn_t_pred_nrm, pred_noise=True)

            amn_smpls = pm.sample_posterior_predictive(
                a_trace, samples=posterior_samples, vars=[amn_pred], progressbar=pbar,
            )

            amn_smpls = amn_smpls["amn_pred"].T

        # -----------------------------------------------------------------------
        # -----------------------------------------------------------------------

        with pm.Model() as mo_model:

            # two random numbers that sum to one; used for variances of gp components;
            # they sum to one because data normalized to have std=1
            A2 = pm.Dirichlet("A2", a=np.array([1, 1]))

            L0 = pm.Gamma("L0", alpha=2, beta=2000)
            cov0 = A2[0] * pm.gp.cov.ExpQuad(1, L0)
            gp = pm.gp.Marginal(cov_func=cov0)

            momn_ = gp.marginal_likelihood(
                "momn_", X=momn_t_nrm[:, None], y=momn_nrm, noise=tt.sqrt(A2[1])
            )

            mo_trace = pm.sample(
                draws,
                chains=chains,
                cores=cores,
                tune=tune,
                target_accept=target_accept,
                progressbar=pbar,
            )

            momn_pred = gp.conditional("momn_pred", momn_t_pred_nrm, pred_noise=True)

            pbar = True if mode == "series" else False
            momn_smpls = pm.sample_posterior_predictive(
                mo_trace, samples=posterior_samples, vars=[momn_pred], progressbar=pbar,
            )

            momn_smpls = momn_smpls["momn_pred"].T

        # -----------------------------------------------------------------------
        # -----------------------------------------------------------------------

        amn_smpls *= amn_sd

        for k in range(posterior_samples):

            msmpl = momn_smpls[:, k] * momn_sd
            asmpl = np.interp(
                momn_t_pred_nrm.flatten(), amn_t_pred_nrm.flatten(), amn_smpls[:, k],
            )

            momn_smpls[:, k] = msmpl + asmpl

        fname = sta_path + "gp_annual.pickle"
        with open(fname, "wb") as f:
            pickle.dump(amn_smpls, f)

        fname = sta_path + "gp_monthly.pickle"
        with open(fname, "wb") as f:
            pickle.dump(momn_smpls, f)

        # -----------------------------------------------------------------------
        # -----------------------------------------------------------------------

        # np.random.seed(10)

        momn_smpls -= momn_smpls.mean()

        fig = plt.figure(figsize=(7.2, 4.5), num="pred1")
        fig.clf()
        col = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        obs = momn + amni
        obs_sd = np.std(obs)

        ax = plt.subplot(221)
        plt.plot(momn_t, obs, "k", lw=1, label="observed")
        lp = np.convolve(obs, np.ones(36) / 36, "same")
        plt.plot(momn_t[2:-2], lp[2:-2], color=col[1], lw=2, label="3-year means")
        ax.set_xlim([1920, 2030])
        ax.set_ylim([-3 * obs_sd, 3.5 * obs_sd])
        ax.set_ylabel("cm")
        ax.legend(ncol=2, loc=1, fontsize=9)
        ax.set_title(str(sta["id"]) + ": " + sta["name"])
        # pt.panel_label(ax, 'a')

        for k in range(3):
            ax = plt.subplot(2, 2, k + 2)
            n = np.random.randint(momn_smpls.shape[1])
            y = momn_smpls[:, n]
            lp = np.convolve(y, np.ones(36) / 36, "same")
            plt.plot(momn_t_pred, y, "k", lw=1, label="simulated")
            plt.plot(
                momn_t_pred[2:-2], lp[2:-2], color=col[1], lw=2, label="3-year means",
            )
            if k == 1:
                ax.set_ylabel("cm")
            ax.set_xlim([2000, 2110])
            ax.set_ylim([-3 * obs_sd, 3.5 * obs_sd])
            ax.legend(ncol=2, loc=1, fontsize=9)
            ax.set_title("Posterior sample " + str(k + 1))
            # pt.panel_label(ax, chr(97+1+k))

        fig.tight_layout()

        # fig.show()

        os.makedirs(fig_path, exist_ok=True)
        fig.savefig(fig_file)

    # -----------------------------------------------------------------------

    except:

        print("An error occurred.", file=lf)

        error_file = log_dir + "error_occurred.txt"
        ef = open(error_file, "w")
        print("An error occurred.", file=ef)
        ef.close()

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    # close log file
    if mode == "parallel":
        lf.close()
        thlf.close()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
