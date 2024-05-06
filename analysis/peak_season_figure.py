import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def hover_template(is_minor):
    threshold = "Minor" if is_minor else "Moderate"
    threshold_string = "Threshold: NOAA " + threshold + "<br>"
    return (
        "<b>%{customdata[0]}</b><br>"
        + "Region: %{customdata[1]}<br>"
        + threshold_string
        + "Multiplier: %{y:0.2f}"
    )


def peak_season_figure(
    pks_df, slr_scenario, threshold=None, threshold_by_region=None, min_mod_switch=None,
):

    pks_df = pks_df.loc[
        (pks_df.Scenario == slr_scenario)
        & (pks_df.Year >= 2030)
        & (pks_df.Year <= 2050)
    ]

    if threshold_by_region is not None:
        z_thrsh = np.zeros(pks_df.shape[0]).astype(bool)
        for reg in pks_df.Region.unique():
            kp_thrsh = (
                threshold_by_region[reg] if reg in threshold_by_region else "minor"
            )
            z_thrsh[(pks_df.Region == reg) & (pks_df.Threshold == kp_thrsh)] = True
    elif min_mod_switch is not None:
        z_thrsh = np.zeros(pks_df.shape[0]).astype(bool)
        for nm in pks_df.Name.unique():
            for yr in pks_df.Year.unique():
                z_nmyr = (pks_df.Name == nm) & (pks_df.Year == yr)
                z_nmyr_mnr = z_nmyr & (pks_df.Threshold == "minor")
                if pks_df.Quantity.loc[z_nmyr_mnr].values[0] >= min_mod_switch:
                    z_thrsh[z_nmyr_mnr] = True
                else:
                    z_nmyr_mdt = z_nmyr & (pks_df.Threshold == "moderate")
                    z_thrsh[z_nmyr_mdt] = True
    elif threshold is not None:
        z_thrsh = pks_df.Threshold == threshold
    else:
        raise (
            "One of 'threshold_by_region', 'min_mod_switch', or 'threshold' "
            + "needs to be set."
        )
    pks_df = pks_df.loc[z_thrsh, :]

    regions = pks_df["Region"].unique()
    offsets = 0.5 * np.arange(regions.size)
    offsets -= offsets.mean()
    offsets += 2.5
    for n, r in enumerate(regions):
        z = pks_df["Region"] == r
        pks_df.loc[z, "Year"] += offsets[n]
    pks_df.loc[:, "Year"] += 0.15 * np.random.random_sample((pks_df.shape[0],)) - 0.075

    fig = px.scatter(
        pks_df,
        y="Quantity",
        x="Year",
        color="Region",
        color_discrete_sequence=[
            px.colors.qualitative.Bold[n] for n in [0, 1, 2, 3, 10, 4, 6]
        ],
        symbol="Threshold",
        hover_data=["Name", "Region", "Quantity"],
    )
    fig.update_traces(
        marker=dict(size=10, opacity=0.7, line=dict(width=0.5, color="white"))
    )
    fig.update_traces(
        patch=dict(showlegend=False),
        selector=dict(marker_symbol="diamond"),
        overwrite=True,
    )
    fig.for_each_trace(
        lambda trace: trace.update(
            legendgroup=trace.name[:-7] if trace.showlegend else trace.name[:-10],
            name=trace.name[:-7] if trace.showlegend else trace.name[:-10],
            hovertemplate=hover_template(trace.showlegend),
        )
    )

    shapes = [pentad_shading(y5) for y5 in [2030, 2040, 2050]]

    scenario_strings = {
        "int_low": "NOAA Intermediate Low",
        "int": "NOAA Intermediate",
        "int_high": "NOAA Intermediate High",
        "kopp": "Kopp et al. (2014)",
    }
    cmmn = {
        "xref": "paper",
        "yref": "paper",
        "xanchor": "left",
        "yanchor": "top",
        "bgcolor": "white",
        "borderpad": 2,
        "showarrow": False,
    }
    annotations = [
        {
            **cmmn,
            **{
                "x": 0.01,
                "y": 1.08,
                "text": scenario_strings[slr_scenario] + " SLR Scenario",
                "font": dict(size=14),
                "bgcolor": None,
            },
        },
    ]

    fig.update_layout(
        template="none",
        width=800,
        height=500,
        margin=dict(l=70, r=15, b=35, t=75, pad=0,),
        xaxis=dict(
            domain=[0, 0.73],
            title_text=None,
            range=[2029.5, 2055],
            showgrid=False,
            tickmode="array",
            tickvals=[2032.5 + 5 * v for v in range(5)],
            ticktext=[
                str(2030 + 5 * v) + "–" + str(2030 + 5 * v + 4) for v in range(5)
            ],
        ),
        yaxis=dict(
            title_text="Peak season multiplier",
            range=[0.75, 10.5],
            # showgrid=False,
            zeroline=False,
        ),
        hovermode="closest",
        legend_title_text="Region",
        legend=dict(x=0.75, y=1),
        title=dict(
            text="Relative severity of peak HTF seasons",
            x=0.098,
            y=0.96,
            font=dict(size=24),
        ),
        annotations=annotations,
        shapes=shapes,
    )

    # create new axes for threshold legend
    fig.update_layout(
        xaxis2=dict(
            domain=[0.75, 1],
            range=[0, 1],
            fixedrange=True,
            tickvals=[],
            zeroline=False,
        ),
        yaxis2=dict(
            domain=[0, 1], range=[0, 1], fixedrange=True, tickvals=[], zeroline=False,
        ),
    )

    thrsh_leg_dy = 0.047
    threshold_legend = pd.DataFrame(
        {
            "name": ["   NOAA Moderate", "   NOAA Minor"],
            "symbol": ["diamond", "circle"],
            "x": [0.115 for k in range(2)],
            "y": [0.43 + thrsh_leg_dy * k for k in [0, 1]],
        }
    )
    fig.add_trace(
        go.Scatter(
            x=[0.015],
            y=[threshold_legend["y"].iloc[-1] + thrsh_leg_dy],
            xaxis="x2",
            yaxis="y2",
            text=["HTF Threshold"],
            textposition="middle right",
            mode="text",
            showlegend=False,
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=threshold_legend["x"],
            y=threshold_legend["y"],
            xaxis="x2",
            yaxis="y2",
            text=threshold_legend["name"],
            textposition="middle right",
            mode="markers+text",
            marker=dict(
                size=9,
                symbol=threshold_legend["symbol"],
                color="rgba(0.3, 0.3, 0.3, 0.75)"  # "white",
                #                 line=dict(color="black", width=1)
            ),
            showlegend=False,
            hoverinfo="none",
        )
    )

    config = {"displayModeBar": False, "responsive": False}
    fig.show(config=config)

    return fig


def pentad_shading(y5):
    return dict(
        type="rect",
        x0=y5,
        x1=y5 + 5,
        y0=-10,
        y1=100,
        layer="below",
        fillcolor="#f5f5f5",
        line=dict(color="black", width=0),
    )
