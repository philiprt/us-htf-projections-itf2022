#%%
# ---------------------------------------------------------------------------

import glob
import json
import pickle

import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------

scn = xr.open_dataset("./slr_scenarios/TR_local_projections_ensemble.nc")
scn_names = scn.tg.data.tolist()

exclude = ["8638901"]

# special cases for scenario names
special_cases = {
    "9461380": "Adak Island, AK ",
    "1630000": "Apra Harbor, Guam",
    "8747437": "Bay Waveland, MS",
    "8519483": "Bergen Point, NY",
    "8638901": "Chesapeake Bay Bridge, VA",
    "8726724": "Clearwater, FL",
    "8775870": "Corpus Christi, TX",
    "9418767": "Humboldt Bay, CA",
    "1820000": "Kwajalein Island",
    "8720218": "Mayport, FL",
    "1619910": "Midway Island",
    "8652587": "Oregon Inlet, NC ",
    "1770000": "Pago Pago, American Samoa ",
    "9431647": "Port Orford, CA",
    "8770570": "Sabine Pass, TX",
    "9440910": "Toke Point,WA",
    "1890000": "Wake Island",
}

rename = {
    "9461380": "Adak Island, AK",
    "1820000": "Kwajalein, RMI",
    "1619910": "Sand Island (Midway)",
    "8652587": "Oregon Inlet, NC",
    "1770000": "Pago Pago, AS",
    "9440910": "Toke Point, WA",
    "1890000": "Wake Island",
}

data_files = glob.glob("./tide_gauge/*.nc")

meta = []
for f in data_files:
    d = xr.open_dataset(f)
    name = d.name[: d.name.find(",")] if "," in d.name else d.name
    state = d.state[: d.state.find(",")] if "," in d.state else d.state
    scn_name = special_cases[d.id] if d.id in special_cases else name + ", " + state
    tool_name = rename[d.id] if d.id in rename else scn_name
    meta.append(
        {
            "id": d.id,
            "name": d.name,
            "state": d.state,
            "scn_name": scn_name,
            "tool_name": tool_name,
            "lat": d.latitude,
            "lon": d.longitude,
        }
    )
    d.close()

meta_df = pd.DataFrame(meta)

missing_names = [nm for nm in scn_names if nm not in meta_df.scn_name.tolist()]
print(missing_names)

meta_df = meta_df.loc[meta_df.scn_name.isin(scn_names)].reset_index()
meta_df = meta_df.drop(columns="index")

with open("./stations.pickle", "wb") as f:
    pickle.dump(meta_df, f)

# ---------------------------------------------------------------------------

meta_json = {}
for _, sta in meta_df.iterrows():
    if sta.id not in exclude:
        meta_json[sta.id] = {
            "id": sta.id,
            "name": sta.tool_name,
            "scn_name": sta.scn_name,
            "lat": sta.lat,
            "lon": sta.lon,
        }

filename = "stations.json"
with open("./stations.json", "w") as f:
    json.dump(meta_json, f, indent=4)

# ---------------------------------------------------------------------------
