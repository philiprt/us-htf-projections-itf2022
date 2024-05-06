# import ipdb
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def color_palette():

    # col = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # col = [col[k] for k in [0, 3, 1]]

    # col = ["#00AFBB", "#E7B802", "#FC4F08"] + col
    # col = ["#003459", "#FB5607", "#D7263D"] + col

    # col = ["#e69f00", "#56b4e9", "#009e74", "#cc79a7", "#0071b2", "#d55e00", "#f0e442"]
    # col = ["#0077bb", "#33bbee", "#009988", "#bbbbbb", "#cc3311", "#ee3377", "#ee7733"]
    # col = ["#228833", "#66ccee", "#4477aa", "#bbbbbb", "#ee6677", "#aa3377", "#ccbb44"]

    # return col

    # Wong (2011), Nature
    return [
        "#0072B2",
        "#56B4E9",
        "#CC79A7",
        "#009E73",
        "#F0E442",
        "#D55E00",
        "#E69F00",
    ]




def multistation_annual_compare(
    analysis, years, quantity, title=None, max_count=None, save=False
):

    title = "Projected exceedance days" if title is None else title

    station_names, thresholds, scenarios, sta_max_count = [], [], [], []
    for reg in analysis:
        for sta in analysis[reg]:
            station_names.append(sta["station"]["name"])
            thresholds.append(sta["experiment"]["threshold"])
            scenarios.append(sta["experiment"]["scenario"])
            if max_count is None:
                sta_max_count.append(sta[quantity].loc[years[-1], 83])

    max_count = max_count if max_count is not None else max(sta_max_count)

    uniq_thrsh = list(set(thresholds))
    if len(uniq_thrsh) > 1:
        sym_thrsh = ["*", "^", "~", "+"]
        station_names = [
            sym_thrsh[uniq_thrsh.index(thrsh)] + nm
            for nm, thrsh in zip(station_names, thresholds)
        ]
    else:
        sym_thrsh = [""]

    uniq_thrsh = [
        sym_thrsh[uniq_thrsh.index(thrsh)] + "NOAA " + thrsh
        if thrsh in ["minor", "moderate"]
        else sym_thrsh[uniq_thrsh.index(thrsh)] + thrsh
        for thrsh in uniq_thrsh
    ]

    uniq_scn = list(set(scenarios))
    if len(uniq_scn) == 1:
        scenario = uniq_scn[0]
    else:
        raise ("Function not constructed to handle multiple scenarios.")

    fig = go.Figure()
    col = color_palette()

    y = 0
    for m, reg in enumerate(analysis):
        for n, sta in enumerate(analysis[reg]):
            annual_compare_single_station(fig, sta[quantity], years, y, col)
            y += 1

    # layout
    annual_compare_layout(fig, station_names, max_count)

    # legend
    annual_compare_legend(fig, years, col)

    # title
    annual_compare_title(fig, title, uniq_thrsh, scenario)

    # config
    config = {"displayModeBar": False, "responsive": False}
    fig.show(config=config)

    return fig


def annual_compare_single_station(fig, quantity, years, y, col):

    wbar = 0.1
    wtot = 0.5 * wbar * len(years)
    yprm = 0.5 * wbar * np.arange(len(years)) - 0.5 * wtot

    for k, yr in enumerate(years):

        zorder = (k + 10) * 10 if yr == years[-1] else (k + 1) * 10

        p17 = int(quantity[17].loc[yr])
        p50 = int(quantity[50].loc[yr])
        p83 = int(quantity[83].loc[yr])

        print

        fig.add_trace(
            go.Scatter(
                x=[p17, p83],
                y=[y, y],
                mode="lines",
                line=dict(color=col[k], width=4),
                hoverinfo="none",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[p50],
                y=[y],
                mode="markers+text",
                marker=dict(color=col[k], size=28),
                text=[str(p50)],
                textposition="middle center",
                textfont=dict(size=12, color="white"),
                hoverlabel=dict(
                    bgcolor=col[k],
                    bordercolor=col[k],
                    font=dict(color="white"),
                ),
                hovertemplate="<b>Year: "
                + str(yr)
                + "</b><br>"
                + "Most likely: "
                + str(p50)
                + "<br>"
                + "Likely range: ["
                + str(p17)
                + ", "
                + str(p83)
                + "] <br>"
                + "<extra></extra>",
            )
        )
        if yr == years[-1]:
            fig.add_trace(
                go.Scatter(
                    x=[p17],
                    y=[y + 0.08],
                    mode="text",
                    text=[str(p17)],
                    textposition="top right",
                    textfont=dict(size=9, color=col[k]),
                    hoverinfo="none",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[p83],
                    y=[y + 0.08],
                    mode="text",
                    text=[str(p83)],
                    textposition="top left",
                    textfont=dict(size=9, color=col[k]),
                    hoverinfo="none",
                )
            )


def annual_compare_layout(fig, station_names, max_count):

    fig_width = 500

    Nsta = len(station_names)
    ytck = np.arange(Nsta)
    yrng = [-0.5, Nsta + 1.2]
    axH = yrng[1] - yrng[0]

    xrng = [-0.08 * max_count, max_count]

    left_margin = max([len(lb) for lb in station_names]) * 8
    fig.update_layout(
        width=fig_width,
        height=axH * 50,
        showlegend=False,
        margin=dict(
            l=left_margin,
            r=10,
            b=0,
            t=0,
            pad=0,
        ),
        template="none",
        xaxis=dict(
            range=xrng,
            tickmode="array",
            tickvals=[],
            showgrid=False,
            zeroline=False,
            layer="below traces",
        ),
        yaxis=dict(
            range=yrng,
            tickmode="array",
            tickvals=ytck,
            ticktext=station_names,
            tickfont=dict(size=14),
            showgrid=True,
            zeroline=False,
            layer="below traces",
        ),
        hovermode="closest",
        hoverlabel_align="left",
    )


def annual_compare_legend(fig, years, col):

    pix_to_nxd = np.diff(fig.layout.xaxis.range)[0] / (
        fig.layout.width - (fig.layout.margin["l"] + fig.layout.margin["r"])
    )

    x_spc_pix = 50
    x_spc_nxd = x_spc_pix * pix_to_nxd

    x0_pix = 35
    x0_nxd = fig.layout.xaxis.range[1] - (x0_pix - fig.layout.margin["r"]) * pix_to_nxd

    legx = [x0_nxd - n * x_spc_nxd for n in range(len(years) - 1, -1, -1)]

    # legend
    for k, y in enumerate(years):
        fig.add_trace(
            go.Scatter(
                x=[legx[k]],
                y=[fig.layout.yaxis.range[1] - 0.95],
                mode="markers+text",
                marker=dict(color=col[k], size=42),
                text="<b>" + str(y) + "</b>",
                textposition="middle center",
                textfont=dict(size=12, color="white"),
                name="none",
                hoverinfo="none",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[np.mean(legx)],
            y=[fig.layout.yaxis.range[1] - 0.2],
            mode="text",
            text="Exceedance days in:",
            textposition="bottom center",
            textfont=dict(size=13),
            name="none",
            hoverinfo="none",
        )
    )


def annual_compare_title(fig, title, uniq_thrsh, scenario):

    scenario_strings = {
        "int_low": "NOAA Intermediate Low",
        "int": "NOAA Intermediate",
        "int_high": "NOAA Intermediate High",
        "kopp": "Kopp et al. (2014)",
    }
    subtitle_strings = [
        "Thresholds: " + ", ".join(uniq_thrsh),
        "SLR Scenario: " + scenario_strings[scenario],
    ]

    xpix_to_ppr = 1 / (
        fig.layout.width - (fig.layout.margin["l"] + fig.layout.margin["r"])
    )
    x0_pix = -fig.layout.margin["l"] + fig.layout.margin["r"]
    x0_ppr = 0 + x0_pix * xpix_to_ppr

    ypix_to_ppr = 1 / fig.layout.height
    y0_pix = fig.layout.height - 3
    y0_ppr = y0_pix * ypix_to_ppr
    dy0_ppr = 31 * ypix_to_ppr
    dy1_ppr = 19 * ypix_to_ppr

    cmmn = dict(
        x=x0_ppr,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
    )
    annotations = [
        {
            **dict(
                y=y0_ppr,
                text="<b>" + title + "</b>",
                font=dict(size=24),
            ),
            **cmmn,
        }
    ]
    for n, sbttl in enumerate(subtitle_strings):
        annotations.append(
            {
                **dict(
                    y=y0_ppr - dy0_ppr - dy1_ppr * n,
                    text=sbttl,
                    font=dict(size=13),
                ),
                **cmmn,
            }
        )
    fig.update_layout(annotations=annotations)
