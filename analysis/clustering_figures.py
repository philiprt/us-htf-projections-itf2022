# import ipdb
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as psb

import analysis_module as anlyz


def clustering_figures(prjn_subset, yr_lims):

    quantity = "xdys_pent_top10pct_ptl"

    stn_meta = [
        {**loc["station"], **loc["experiment"], **{"steps": loc["steps"]},}
        for reg in prjn_subset
        for loc in prjn_subset[reg]
    ]

    pct_top10 = [
        100 * loc[quantity].loc[yr_lims[0] : yr_lims[1]].dropna()
        for reg in prjn_subset
        for loc in prjn_subset[reg]
    ]
    pct_top10 = pct_top10[:4]

    eff_dens = [
        (loc[quantity] * 5 * loc["xdys_pent_mn_ptl"] / 6)
        .loc[yr_lims[0] : yr_lims[1]]
        .dropna()
        for reg in prjn_subset
        for loc in prjn_subset[reg]
    ]
    eff_dens = eff_dens[:4]

    vspace = 0.15
    fig = psb.make_subplots(
        rows=2,
        cols=2,
        #     subplot_titles=[s["name"] for s in stn_meta],
        #         shared_xaxes=True,
        vertical_spacing=vspace,
    )

    annotations = []
    for n, (pct, effd) in enumerate(zip(pct_top10, eff_dens)):
        r = int(n / 2) + 1
        c = (n % 2) + 1
        #         ipdb.set_trace()
        fig.add_trace(
            go.Scatter(
                x=effd[50].values,
                error_x=dict(
                    array=(effd[90] - effd[50]).values,
                    arrayminus=(effd[50] - effd[10]).values,
                ),               
                y=pct[50].values,
                error_y=dict(
                    array=(pct[90] - pct[50]).values,
                    arrayminus=(pct[50] - pct[10]).values,
                ),
                mode="markers"
            ),
            row=r,
            col=c,
        )

    #         leg = True if n == 0 else False
    #         r = int(n / 2) + 1
    #         c = (n % 2) + 1
    #         traces = projection_traces(prjn, stn_meta[n], col, leg)
    #         annotations.extend(
    #             projection_annotations(prjn, stn_meta[n], r, c, vspace)
    #         )
    #         for trc in traces:
    #             fig.add_trace(trc, row=r, col=c)

    #         fig.update_yaxes(title_text="days", row=r, col=c, side="right")
    #     #     fig.update_xaxes(range=yr_lims, row=r, col=c)

    fig.update_layout(
        width=800,
        height=525,
        template="none",
        #         margin=dict(l=25, r=60, b=20, t=100, pad=0,),
        #         xaxis=dict(layer="below traces",),
        #         yaxis=dict(layer="below traces",),
        #         hovermode="x",
        #         #     hoverlabel_align="left",
        #         legend=dict(orientation="h", x=0.675, y=1.17),
        #         title=dict(
        #             text="Projected High-Tide Flooding Days",
        #             x=0.035,
        #             y=0.975,
        #             font=dict(size=24),
        #         ),
        #         annotations=annotations,
    )

    #     config = {"displayModeBar": False, "responsive": False}
    fig.show()  # config=config)


def projection_traces(prjn, meta, col, leg):

    fcol = [fill_color(c, 0.5) for c in col if c[0] == "#"]

    prjn_traces = [
        {
            "x": prjn.index.values,
            "y": prjn[10].values,
            "type": "scatter",
            "fill": "none",
            "showlegend": False,
            "line": {"color": fcol[0], "width": 0},
            "hoverinfo": "y",
        },
        {
            "x": prjn.index.values,
            "y": prjn[90].values,
            "type": "scatter",
            "fill": "tonexty",
            "fillcolor": fcol[0],
            "showlegend": False,
            "mode": "none",
            "hoverinfo": "none",
        },
        {
            "x": prjn.index.values,
            "y": prjn[50].values,
            "type": "scatter",
            "name": "50th percentile" if leg else None,
            "showlegend": True if leg else False,
            "line": {"color": col[0], "width": 3},
            "hoverinfo": "y",
        },
        {
            "x": prjn.index.values,
            "y": prjn[90].values,
            "type": "scatter",
            "name": "90th percentile" if leg else None,
            "showlegend": True if leg else False,
            "line": {"color": col[2], "width": 2},
            "hoverinfo": "y",
        },
    ]

    annotation_traces = []
    yrs = [y for y in meta["steps"]]
    cmmn = {
        "type": "scatter",
        "mode": "lines",
        "showlegend": False,
        "hoverinfo": "none",
    }
    annotation_traces.append(
        {
            **cmmn,
            **{
                "x": yrs,
                "y": prjn[50].loc[yrs],
                "type": "scatter",
                "mode": "markers",
                "showlegend": False,
                "hoverinfo": "none",
                "line": dict(color="black"),
            },
        }
    )

    for n, stp in enumerate(zip(meta["steps"][:-1], meta["steps"][1:])):
        lower = n < (len(meta["steps"]) - 2)
        dash = "dot" if lower else "solid"
        p12 = [stp[1], stp[0]] if lower else stp

        annotation_traces.append(
            {
                **{
                    "x": [stp[0], p12[0]],
                    "y": prjn[50].loc[[stp[0], p12[1]]],
                    "line": dict(color="black", width=1, dash=dash),
                },
                **cmmn,
            }
        )
        annotation_traces.append(
            {
                **{
                    "x": [p12[0], stp[1]],
                    "y": prjn[50].loc[[p12[1], stp[1]]],
                    "line": dict(color="black", width=1, dash=dash),
                },
                **cmmn,
            }
        )

    prjn_traces.extend(annotation_traces)

    return prjn_traces


def projection_annotations(prjn, meta, r, c, vspace):

    x = 2020

    dy = 0.05
    y = 1 + dy if r == 1 else (1 - vspace) / 2 + dy

    cmmn = {
        "x": x,
        "xref": "x" + str(c),
        "yref": "paper",
        "xanchor": "left",
        "yanchor": "top",
        "bgcolor": "white",
        "borderpad": 2,
        "showarrow": False,
    }
    annotations = [
        {
            **{
                "y": y - 0.055,
                "text": "Threshold: NOAA " + meta["threshold"].capitalize(),
            },
            **cmmn,
        },
        {
            **{
                "y": y,
                "text": "<b>" + meta["name"] + "<b>",  # ,
                "font": dict(size=16),
            },
            **cmmn,
        },
    ]
    for n, stp in enumerate(zip(meta["steps"][:-1], meta["steps"][1:])):

        dlt = prjn[50].loc[stp[1]] - prjn[50].loc[stp[0]]
        day_or_days = " day/year" if dlt == 1 else " days/year"
        inc_str = u"Δ = " + str(int(dlt)) + day_or_days

        if prjn[50].loc[stp[1]] <= 1:
            step_str = str(stp[0]) + u" → " + str(stp[1]) + ": " + "Few events"
        else:
            step_str = str(stp[0]) + u" → " + str(stp[1]) + ": " + inc_str
        annotations.append(
            {**{"y": y - 0.01 - (n + 2) * 0.045, "text": step_str}, **cmmn,},
        )
    if (r == 1) & (c == 1):
        scenario_strings = {
            "int_low": "NOAA Intermediate Low",
            "int": "NOAA Intermediate",
            "int_high": "NOAA Intermediate High",
            "kopp": "Kopp et al. (2014)",
        }
        annotations.append(
            {
                **cmmn,
                **{
                    "y": y + 0.105,
                    "text": scenario_strings[meta["scenario"]]
                    + " SLR Scenario",
                    "font": dict(size=14),
                    "bgcolor": None,
                },
            },
        )

    return annotations
