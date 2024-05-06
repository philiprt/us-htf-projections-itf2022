# ---------------------------------------------------------------------------

import glob
import os

import numpy as np
import pandas as pd

import pickle
import xarray as xr

import matplotlib.dates as mdates
from timezonefinder import TimezoneFinder

from support_functions import station_string

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def tide_gauge_data(sta):

    # -----------------------------------------------------------------------
    # load station data

    def load_data(sta_id):

        fname = "../data/tide_gauge/" + str(sta_id) + ".nc"

        fname = glob.glob(fname)
        if len(fname) == 0:
            tg = None
            return tg
        else:
            fname = fname[0]

        tg = xr.open_dataset(fname).to_dataframe()
        tg.columns = ["sl", "td"]

        beg_end = tg.sl.loc[~tg.sl.isna()].iloc[[0, -1]].index
        tg = tg.loc[beg_end[0] : beg_end[1]]

        tg *= 100  # cm above MHHW

        return tg

    tg = load_data(sta["id"])

    # -----------------------------------------------------------------------

    if sta["id"] == "8413320":
        tg = tg.drop(tg.loc["1975-08":"1977"].index)

    if sta["id"] == "8774770":
        tg = tg.loc["1960":]

    if sta["id"] == "1770000":
        tg = tg.loc[:"2009-09-28"]

    if sta["id"] == "8729108":
        tg = tg.loc["1979":]

    if sta["id"] == "8447930":
        tg.loc[:"1970", :] = None

    if sta["id"] == "8720030":
        tg.loc[:"1930", :] = None

    if sta["id"] == "8573364":
        tg.loc[:"1990", :] = None

    if sta["id"] == "8518750":
        tg.loc[:"1926", :] = None

    if sta["id"] == "8531680":
        tg.loc[:"1930", :] = None

    if sta["id"] == "8635750":
        tg.loc[:"1978", :] = None

    if sta["id"] == "1617433":  # Kawaihae: fix offset due to earthquake in 2006

        tg_comp = load_data("1617760")  # Hilo
        diff = tg_comp.sl - tg.sl  # Hilo minus Kawaihae
        offset = diff.loc[:"2006-10-14"].mean() - diff.loc["2006-10-15":].mean()

        tg.sl.loc[:"2006-10-14"] += offset

    # -----------------------------------------------------------------------

    sta_str = station_string(sta)
    fname = "../tides/output/" + sta_str + "/tide_prediction_mn_std.pickle"

    if os.path.exists(fname):

        with open(fname, "rb") as f:
            ptd = pickle.load(f)

        # if sta["id"] == "1612340":  # honolulu
        #     ptd["obs"].index = [
        #         d - pd.Timedelta("30 m") if d < pd.Timestamp("1947-06-13") else d
        #         for d in ptd["obs"].index
        #     ]

        tg = tg.loc[: ptd["obs"].index[-1]]
        tg.td = ptd["obs"].mn.reindex(tg.index)
        tg.td -= tg.td.loc["1983":"2001"].mean()
        tg.td += tg.sl.loc["1983":"2001"].mean()

    # -----------------------------------------------------------------------
    # calculate residuals

    tg["res"] = tg.sl - tg.td

    # -----------------------------------------------------------------------
    # calculate annual cycle and trend in residuals

    jdt = tg.index.to_julian_date()
    jdt -= jdt[0]

    # calculate annual cycle and trend in residuals
    phs = jdt * (2 * np.pi) / 365.25
    harm = np.vstack([np.sin(phs), np.cos(phs), np.sin(2 * phs), np.cos(2 * phs)])

    z = ~tg.res.isna()
    A = np.vstack([np.ones(jdt.size), jdt, harm]).T
    c = np.linalg.lstsq(A[z, :], tg.res[z], rcond=None)[0]

    tg["acyc"] = np.sum(A[:, 2:] * c[2:], axis=1)
    tg["trnd"] = np.sum(A[:, 0:2] * c[0:2], axis=1)

    # --------------------------------------------------------------------

    return tg

    # -----------------------------------------------------------------------
