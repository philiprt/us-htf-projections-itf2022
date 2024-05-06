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
        + "5-yr Total: %{customdata[3]}<br>"
        + "% in top 6: %{y:0.0f}%"
        + "<extra></extra>"
    )


def top10_percent_figure(
    t10_df, slr_scenario, pentad_yr1,
):

    t10_df = t10_df.copy()

    t10_df = t10_df.loc[(t10_df.Scenario == slr_scenario) & (t10_df.Year == pentad_yr1)]

    t10_df.loc[:, "Quantity"] *= 100

    fig = px.scatter(
        t10_df,
        y="Quantity",
        x="Total",
        color="Region",
        color_discrete_sequence=[
            # px.colors.qualitative.Bold[n] for n in [0, 1, 2, 3, 10, 4, 6]
            "#228833",
            "#66ccee",
            "#4477aa",
            "#bbbbbb",
            "#ee6677",
            "#aa3377",
            "#ccbb44",
        ],
        symbol="Threshold",
        hover_data=["Name", "Region", "Quantity", "Total"],
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

    #     shapes = [pentad_shading(y5) for y5 in [2030, 2040, 2050]]

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
                "x": -0.037,
                "y": 1.1,
                "text": scenario_strings[slr_scenario] + " SLR Scenario",
                "font": dict(size=14),
                "bgcolor": None,
            },
        },
    ]

    fig.update_layout(
        template="none",
        width=550,
        height=450,
        margin=dict(l=70, r=15, b=45, t=75, pad=0,),
        xaxis=dict(
            domain=[0, 1],  # 0.73],
            zeroline=False,
            title_text="5-year total HTF days",
            range=[-10, 900],
        ),
        yaxis=dict(
            title_text="% in most severe six months",
            range=[10, 105],
            tickvals=np.arange(20, 110, 20),
            zeroline=False,
        ),
        hovermode="closest",
        legend_title_text="Region",
        legend=dict(x=0.65, y=1),
        title=dict(
            text="Clustering of HTF days, "
            + str(pentad_yr1)
            + "–"
            + str(pentad_yr1 + 4),
            x=0.098,
            y=0.96,
            font=dict(size=24),
        ),
        annotations=annotations,
        #         shapes=shapes,
    )

    # create new axes for threshold legend
    fig.update_layout(
        xaxis2=dict(
            domain=[0.6, 1], range=[0, 1], fixedrange=True, tickvals=[], zeroline=False,
        ),
        yaxis2=dict(
            domain=[0.3, 1], range=[0, 1], fixedrange=True, tickvals=[], zeroline=False,
        ),
    )

    thrsh_leg_dy = 0.082
    threshold_legend = pd.DataFrame(
        {
            "name": ["   NOAA Moderate", "   NOAA Minor"],
            "symbol": ["diamond", "circle"],
            "x": [0.235 for k in range(2)],
            "y": [0.08 + thrsh_leg_dy * k for k in [0, 1]],
        }
    )
    fig.add_trace(
        go.Scatter(
            x=[0.14],
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
