# ---------------------------------------------------------------------------

import os

# below must be done before importing numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from multiprocessing import Pool

import pickle

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import splev, splrep

from support_functions import station_string
from tide_gauge_data import tide_gauge_data

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def main():

    mode = "serial"
    # mode = "multiprocess"

    # -----------------------------------------------------------------------

    fname = "../data/stations.pickle"
    stations = pd.read_pickle(fname)

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

    # # only do select stations
    # select = ["1612340"]
    # keep = [n for n in stations.index if stations.loc[n, "id"] in select]
    # stations = stations.loc[keep]

    # -----------------------------------------------------------------------

    if mode == "serial":

        Nsta = stations.index.size
        for idx, sta in stations.iterrows():
            print(f"Station {sta['id']}: {sta['name']} ({idx+1} of {Nsta})")
            eta99_ensembles(sta)

    # ----------------------------------------------------------------------

    elif mode == "multiprocess":

        sta_list = [stations.loc[n] for n in stations.index]

        agents = 6
        chunksize = 1
        with Pool(processes=agents) as pool:
            pool.map(eta99_ensembles, sta_list, chunksize)


# ---------------------------------------------------------------------------


def eta99_ensembles(sta):

    sta_str = station_string(sta)
    n_smpls = 1000

    # -----------------------------------------------------------------------
    # Directories for I/O
    # -----------------------------------------------------------------------

    # station output directory
    sta_path = "./output/" + sta_str + "/"
    os.makedirs(sta_path, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load resources and setup
    # -----------------------------------------------------------------------

    tg = tide_gauge_data(sta)
    if tg is None:
        return

    msl2000 = tg.res.loc["1991":"2009"].mean()

    # -----------------------------------------------------------------------
    # calculate mean annual cycle

    # drop months which have more than 10 days missing data
    tg = tg.groupby(pd.Grouper(freq="M")).apply(
        lambda x: None if x.res.isna().sum() > 24 * 10 else x
    )
    try:
        tg.index = tg.index.droplevel(0)
    except:
        pass

    mac = tg.sl.groupby(tg.index.month).mean()
    mac.sort_index(inplace=True)
    mac -= mac.mean()
    mac = mac.values

    # -----------------------------------------------------------------------
    # load scenarios

    scn_nm = ["low", "int_low", "int", "int_high", "high"]
    scn_nm_in_file = [
        "rsl_total_" + "".join([c for c in nm.title() if c != "_"]) for nm in scn_nm
    ]

    filename = "../data/slr_scenarios/TR_local_projections.nc"
    p = xr.open_dataset(filename)

    scn = (
        p.sel(tg=sta["scn_name"], percentiles=50)
        .to_pandas()[scn_nm_in_file]
        .loc[2020:2100]
    )
    scn.columns = scn_nm
    scn.index += 0.5
    scn /= 10  # cm
    scn += msl2000

    # -----------------------------------------------------------------------
    # interpolate to monthly

    idx = np.arange(2020, 2101, 1 / 12) + 1 / 24
    msl_prjn = pd.DataFrame(index=idx, columns=scn.columns)

    for s in scn_nm:
        spl = splrep(scn.index, scn[s].values, k=3, s=2)
        msl_prjn[s] = splev(msl_prjn.index, spl)

    # -----------------------------------------------------------------------
    # trajectory ensemble

    filename = "../data/slr_scenarios/TR_local_projections_ensemble.nc"
    e = xr.open_dataset(filename)

    traj_ens = e.rsl_trajectory_ensemble.sel(tg=sta["scn_name"]).to_pandas().T
    traj_ens = traj_ens.sample(n=n_smpls, axis="columns")
    traj_ens.columns = range(n_smpls)
    traj_ens.index += 0.5
    traj_ens /= 10  # cm
    traj_ens += msl2000

    traj_ens_mo = pd.DataFrame(index=idx, columns=traj_ens.columns)

    for c in traj_ens.columns:
        spl = splrep(traj_ens.index, traj_ens.loc[:, c].values, k=3, s=2)
        traj_ens_mo.loc[:, c] = splev(traj_ens_mo.index, spl)

    traj_ens_mo.loc[idx > 2051, :] = None

    # -----------------------------------------------------------------------

    fname = f"../gp_monthly/output/{sta_str}/gp_monthly.pickle"
    with open(fname, "rb") as f:
        gp_smpls = pickle.load(f)

    gp_idx = np.arange(2000, 2101, 1 / 12) + 1 / 24
    gp = pd.DataFrame(gp_smpls, index=gp_idx)
    gp = gp.loc[gp.index > 2020]
    gp = np.tile(gp.values, [1, int(n_smpls / gp.values.shape[1])])

    # -----------------------------------------------------------------------

    td_path = f"../tides/output/{station_string(sta)}/"
    fname = td_path + "tide_99th_percentile_monthly.pickle"

    with open(fname, "rb") as f:
        zeta99 = pickle.load(f)

    for k in zeta99.keys():
        if k != "obs":
            zeta99[k] = zeta99[k].loc["2020":"2100"]
            n_tiles = int(n_smpls / zeta99[k].shape[1])
            zeta99[k] = pd.DataFrame(np.tile(zeta99[k].values, [1, n_tiles]), index=idx)

    # -----------------------------------------------------------------------

    mac = np.tile(mac, int(gp.shape[0] / 12))

    # -----------------------------------------------------------------------

    e99 = {
        s: zeta99[s + "_od"].add(gp, axis=0).add(msl_prjn[s], axis=0).add(mac, axis=0)
        for s in msl_prjn
    }
    e99["traj"] = (
        zeta99["int_low_od"].add(gp, axis=0).add(traj_ens_mo, axis=0).add(mac, axis=0)
    )
    e99 = xr.Dataset(e99)
    e99 = e99.rename({"dim_0": "time", "dim_1": "realization"})

    filename = "./output/" + sta_str + "/eta99_ensemble.nc"
    if os.path.exists(filename):
        os.remove(filename)
    e99.to_netcdf(filename)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
