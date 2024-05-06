import ipdb
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import analysis_module as anlyz


def hover_template(is_minor):
    return (
        "<b>%{customdata[0]}</b><br>"
        + "Region: %{customdata[1]}<br>"
        + "Nodal cycle range: %{y:0.1f} cm<br>"
        + "10 years of SLR: %{x:0.1f} cm"
        + "<extra></extra>"
    )


def load_ncyc_vs_slr(station):

    sta = pd.Series({"name": station.name, "noaa id": station["NOAA ID"]})
    sta_str = anlyz.station_string(sta)

    fname = "../Output/pickle/" + sta_str + "/nodal_cycle_vs_slr.pickle"

    with open(fname, "rb") as f:
        d = pickle.load(f)

    station["ncyc"] = d["ncyc"]
    try:
        station["slr"] = d["slr_prjn"]
    except:
        station["slr"] = None

    return station


def nodal_cycle_figure(stations, combine_regions=None):

    stations = stations.copy()

    if combine_regions is not None:
        for new_reg in combine_regions:
            for old_reg in combine_regions[new_reg]:
                z = stations["Region"] == old_reg
                stations.loc[z, "Region"] = new_reg

    z = stations["Region"] == "Caribbean"
    stations = pd.concat([stations.loc[~z], stations.loc[z]])
    
    stations["ncyc"] = None
    stations["slr"] = None
    
    stations.apply(lambda x: load_ncyc_vs_slr(x), axis=1)
    stations["Name"] = stations.index
    
    stations["msize"] = 14
    stations.loc[stations.slr < -0.5, "msize"] = 10
    print((stations.ncyc > 2).sum()/stations.ncyc.size)

    fig = px.scatter(
        stations,
        y="ncyc",
        x="slr",
        size="msize",
        color="Region",
        color_discrete_sequence=[
            px.colors.qualitative.Bold[n] for n in [0, 1, 2, 3, 10, 4, 6]
#             px.colors.qualitative.Bold[n] for n in [0, 1, 6, 3, 2, 4, 10]
        ],
        hover_data=["Name", "Region", "ncyc", "slr"],
    )
    fig.update_traces(
        marker=dict(size=14, opacity=0.7, line=dict(width=0.5, color="white"))
    )
    fig.for_each_trace(
        lambda trace: trace.update(
            hovertemplate=hover_template(trace.showlegend),
        )
    )
    fig.add_scatter(
        x=[-1, 20],
        y=[-0.5, 10],
        showlegend=False,
        mode="lines",
        line=dict(width=1),
    )

    fig.update_layout(
        template="none",
        width=550,
        height=450,
        margin=dict(l=70, r=15, b=45, t=55, pad=0,),
        xaxis=dict(
            domain=[0, 1],  # 0.73],
            zeroline=False,
            title_text="10 years observed SLR (cm)",
            range=[-0.5, 15],
            tickvals=np.arange(0, 15, 2),
        ),
        yaxis=dict(
            title_text="Nodal cycle modulation range (cm)",
            range=[-0.5, 11.5],
            tickvals=np.arange(0, 11, 2),
            zeroline=False,
        ),
        hovermode="closest",
        legend_title_text="Region",
        legend=dict(x=0.05, y=0.22),
        title=dict(
            text="Relevance of the 18.6-year nodal cycle",
            x=0.13,
            y=0.96,
            font=dict(size=20),
        ),
        #         annotations=annotations,
        #         shapes=shapes,
    )

    config = {"displayModeBar": False, "responsive": False}
    fig.show(config=config)

    return fig, stations
