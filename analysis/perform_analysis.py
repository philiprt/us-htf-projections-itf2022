import os

# below must be done before importing numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import xarray as xr
import pickle
import json

from multiprocessing import Pool
from functools import partial

import analysis_module as anlyz


def main():

    # get meta for list of NOS stations to analyze
    # stations = anlyz.station_list(exclude=["8638901", "8720218", "8725520", "9463502"])
    stations = anlyz.station_list(select=["1617433"])

    Nsta = stations.index.size
    for idx, sta in stations.iterrows():

        print(f"\nStation {sta['id']}: {sta['name']} ({idx+1} of {Nsta})")

        sta_str = anlyz.station_string(sta)
        sta_path = f"../htf_projections/output/{sta_str}/"

        # load eta99 ensemble
        e99 = xr.open_dataset(f"{sta_path}eta99_ensemble.nc")
        years = np.floor(e99.time).astype(int).values
        months = ((e99.time - years + 1 / 24) * 12).round().astype(int).values
        e99_midx = pd.MultiIndex.from_arrays([years, months], names=("year", "month"))

        # loop over each SLR scenario
        for scn in ["traj", "low", "int_low", "int", "int_high", "high"]:

            end = "\n" if scn == "high" else ""
            print(f"{scn} ", end=end)

            # isolate e99 for this scenario
            e99_scn = e99[scn].to_pandas()
            e99_scn.index = e99_midx

            # load monthly statistical models
            bbmodels = dict()
            time_of_day = [0, 23]
            for m in range(1, 13):

                mo_str = "mo" + "{:0>2}".format(m)
                tod_str = (
                    "h"
                    + "{:0>2}".format(time_of_day[0])
                    + "{:0>2}".format(time_of_day[1])
                )
                exp_str = mo_str + "_" + tod_str

                filename = sta_path + exp_str + "/bbmodel.pickle"
                with open(filename, "rb") as f:
                    bbmodels[m] = pickle.load(f)

            # define thresholds
            thresholds = [x for x in range(306)]

            # perform analysis in parallel
            agents = 5
            chunksize = 1
            with Pool(processes=agents) as pool:
                sta_anlys = partial(
                    station_analysis,
                    e99_scn=e99_scn,
                    bbmodels=bbmodels,
                    station=sta,
                    scenario=scn,
                )
                _ = pool.map(sta_anlys, thresholds, chunksize)


def station_analysis(threshold, e99_scn, bbmodels, station, scenario):

    e99_abv_thrsh = e99_scn - threshold

    if scenario == "traj":
        e99_abv_thrsh = e99_abv_thrsh.loc[(slice(None, 2050), slice(None)), :]

    xd_ens = anlyz.xdys_monthly_ensemble(e99_abv_thrsh, bbmodels)

    analysis = dict()

    (
        analysis["monthly_percentiles"],
        analysis["monthly_probabilities"],
    ) = anlyz.monthly_calculations(xd_ens, station, scenario, threshold)

    (
        analysis["annual_percentiles"],
        analysis["yoi"],
        analysis["annual_probabilities"],
    ) = anlyz.annual_calculations(xd_ens, station, scenario, threshold)

    (
        analysis["pentad_mean_month_percentiles"],
        analysis["pentad_max_month_percentiles"],
        analysis["pentad_max_season_percentiles"],
    ) = anlyz.pentad_calculations(xd_ens, station, scenario, threshold)

    path = f"./ensemble_stats/{station.id}/{scenario}/"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}{threshold:03d}.json", "w") as f:
        json.dump(analysis, f)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
