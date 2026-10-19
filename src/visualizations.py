"""
Climate Trend Analyzer - Interactive Plotly Visualization Module.

Responsible for:
- Generating publication-grade, interactive Plotly charts.
- Enforcing consistent typography, color schemes, and informative hover tooltips.
- Implementing zero-anomaly baselines, warming/cooling color maps, and confidence bands.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Curated climate theme color palette
COLOR_PRIMARY_BLUE = "#38bdf8"     # Light Sky Blue
COLOR_WARM_RED = "#f87171"         # Coral Red
COLOR_GOLD = "#fbbf24"             # Amber Gold
COLOR_PURPLE = "#c084fc"           # Soft Violet
COLOR_EMERALD = "#34d399"          # Emerald Green
COLOR_DARK_BG = "#0f172a"          # Slate 900
COLOR_CARD_BG = "#1e293b"          # Slate 800
COLOR_GRID = "#334155"             # Slate 700
COLOR_TEXT = "#f8fafc"             # Slate 50


def apply_theme_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 480,
) -> go.Figure:
    """Apply uniform, professional dark-slate climate styling to any Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 18, "color": COLOR_TEXT, "family": "Inter, sans-serif"},
            "x": 0.02,
            "y": 0.95,
        },
        paper_bgcolor=COLOR_CARD_BG,
        plot_bgcolor=COLOR_DARK_BG,
        font={"color": COLOR_TEXT, "family": "Inter, sans-serif"},
        xaxis={
            "title": xaxis_title,
            "gridcolor": COLOR_GRID,
            "zeroline": True,
            "zerolinecolor": "#64748b",
            "showgrid": True,
        },
        yaxis={
            "title": yaxis_title,
            "gridcolor": COLOR_GRID,
            "zeroline": True,
            "zerolinecolor": "#ef4444",
            "zerolinewidth": 1.5,
            "showgrid": True,
        },
        legend={
            "bgcolor": "rgba(30, 41, 59, 0.8)",
            "bordercolor": COLOR_GRID,
            "borderwidth": 1,
            "font": {"size": 12},
        },
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        height=height,
        hovermode="x unified",
    )
    return fig


def plot_historical_series(
    df: pd.DataFrame,
    show_rolling: bool = True,
    title: str = "Historical Surface Temperature Anomalies",
) -> go.Figure:
    """
    Plot historical monthly observations with trailing 12-month and 60-month moving averages.
    """
    fig = go.Figure()

    # Raw monthly observations (subtle line with markers)
    fig.add_trace(
        go.Scatter(
            x=df["Decimal_Year"],
            y=df["Anomaly"],
            mode="lines",
            name="Monthly Anomaly",
            line={"color": "rgba(148, 163, 184, 0.45)", "width": 1.0},
            hovertemplate="Date: %{customdata}<br>Anomaly: %{y:+.2f} °C<extra></extra>",
            customdata=df["Date"],
        )
    )

    if show_rolling:
        if "Rolling_12m" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Decimal_Year"],
                    y=df["Rolling_12m"],
                    mode="lines",
                    name="12-Month Moving Avg",
                    line={"color": COLOR_PRIMARY_BLUE, "width": 2.2},
                    hovertemplate="12m Avg: %{y:+.2f} °C<extra></extra>",
                )
            )

        if "Rolling_60m" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Decimal_Year"],
                    y=df["Rolling_60m"],
                    mode="lines",
                    name="60-Month (5-Yr) Moving Avg",
                    line={"color": COLOR_GOLD, "width": 2.8},
                    hovertemplate="60m Avg: %{y:+.2f} °C<extra></extra>",
                )
            )

    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text="1901-2000 Baseline (0.0°C)",
        annotation_position="bottom right",
    )

    return apply_theme_layout(
        fig,
        title=title,
        xaxis_title="Year",
        yaxis_title="Temperature Departure (°C)",
    )


def plot_annual_bars(
    annual_df: pd.DataFrame,
    title: str = "Annual Average Temperature Departures",
) -> go.Figure:
    """
    Render annual mean temperature anomalies as diverging colored bars (blue = cool, red = warm).
    """
    fig = go.Figure()

    colors = np.where(
        annual_df["Annual_Mean"] >= 0,
        "#ef4444",  # Red for positive anomaly
        "#3b82f6",  # Blue for negative anomaly
    )

    fig.add_trace(
        go.Bar(
            x=annual_df["Year"],
            y=annual_df["Annual_Mean"],
            marker={"color": colors, "line": {"width": 0.5, "color": "#1e293b"}},
            name="Annual Mean",
            hovertemplate="Year: %{x}<br>Annual Mean: %{y:+.3f} °C<extra></extra>",
        )
    )

    fig.add_hline(y=0.0, line_dash="solid", line_color="#cbd5e1", line_width=1)

    return apply_theme_layout(
        fig,
        title=title,
        xaxis_title="Calendar Year",
        yaxis_title="Departure from Climatological Mean (°C)",
    )


def plot_trend_regression(
    trend_dict: Dict[str, Any],
    title: str = "Long-Term Temperature Trend (OLS Regression)",
) -> go.Figure:
    """
    Display empirical observations against the fitted OLS linear regression line.
    """
    fig = go.Figure()

    x = trend_dict["x_values"]
    y = trend_dict["y_values"]
    y_pred = trend_dict["trend_predictions"]
    rate = trend_dict["rate_per_decade"]
    r2 = trend_dict["r_squared"]
    p_val = trend_dict["p_value"]
    ci_lo = trend_dict["ci_decade_lower"]
    ci_hi = trend_dict["ci_decade_upper"]

    # Scatter points of observations
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Observations",
            marker={"color": "#64748b", "size": 3.5, "opacity": 0.5},
            hovertemplate="Year: %{x:.1f}<br>Observed: %{y:+.2f} °C<extra></extra>",
        )
    )

    # Fitted Trend Line
    trend_label = f"OLS Trend: +{rate:.3f} °C/decade (R²={r2:.3f})"
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred,
            mode="lines",
            name=trend_label,
            line={"color": "#ef4444", "width": 3.2},
            hovertemplate="Trend Estimate: %{y:+.2f} °C<extra></extra>",
        )
    )

    subtitle_note = (
        f"Warming Rate: +{rate:.3f} °C/decade (95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}] °C/decade, p < 0.0001)"
        if p_val < 0.0001
        else f"Warming Rate: +{rate:.3f} °C/decade (95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}], p={p_val:.4f})"
    )

    fig = apply_theme_layout(
        fig,
        title=f"{title}<br><sup>{subtitle_note}</sup>",
        xaxis_title="Decimal Year",
        yaxis_title="Anomaly (°C)",
    )
    return fig


def plot_monthly_climatology(
    monthly_df: pd.DataFrame,
    title: str = "Climatological Monthly Anomaly Distribution",
) -> go.Figure:
    """
    Render mean monthly temperature departures with standard deviation error bars.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=monthly_df["Month_Name"],
            y=monthly_df["Mean"],
            error_y={
                "type": "data",
                "array": monthly_df["Std"],
                "visible": True,
                "color": "#94a3b8",
            },
            marker={"color": COLOR_PRIMARY_BLUE},
            name="Monthly Mean ± 1σ",
            hovertemplate="Month: %{x}<br>Mean: %{y:+.3f} °C<br>Std Dev: ±%{error_y.array:.3f} °C<extra></extra>",
        )
    )

    return apply_theme_layout(
        fig,
        title=title,
        xaxis_title="Month",
        yaxis_title="Mean Anomaly (°C)",
    )


def plot_seasonal_comparison(
    seasonal_trend_df: pd.DataFrame,
    title: str = "Warming Rate by Meteorological Season",
) -> go.Figure:
    """
    Compare decadal warming rates across meteorological seasons (DJF, MAM, JJA, SON).
    """
    fig = go.Figure()

    colors = ["#38bdf8", "#34d399", "#f87171", "#fbbf24"]

    fig.add_trace(
        go.Bar(
            x=seasonal_trend_df["Season"],
            y=seasonal_trend_df["Warming Rate (°C/Decade)"],
            marker={"color": colors[: len(seasonal_trend_df)]},
            name="Warming Rate",
            hovertemplate="Season: %{x}<br>Rate: +%{y:.3f} °C/decade<extra></extra>",
        )
    )

    return apply_theme_layout(
        fig,
        title=title,
        xaxis_title="Meteorological Season",
        yaxis_title="Decadal Warming Rate (°C / Decade)",
    )


def plot_decomposition(
    decomp_df: pd.DataFrame,
    title: str = "Classical Additive Time Series Decomposition (Y = T + S + R)",
) -> go.Figure:
    """
    4-panel stacked subplot visualizing Observed, Trend, Seasonal, and Residual components.
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "1. Observed Series (Y)",
            "2. Long-Term Trend (T)",
            "3. Seasonal Cycle (S)",
            "4. Residual Irregular Component (R)",
        ],
    )

    x = decomp_df["Decimal_Year"]

    # 1. Observed
    fig.add_trace(
        go.Scatter(x=x, y=decomp_df["Observed"], line={"color": "#94a3b8", "width": 1.2}, name="Observed"),
        row=1, col=1,
    )
    # 2. Trend
    fig.add_trace(
        go.Scatter(x=x, y=decomp_df["Trend"], line={"color": "#ef4444", "width": 2.2}, name="Trend"),
        row=2, col=1,
    )
    # 3. Seasonal
    fig.add_trace(
        go.Scatter(x=x, y=decomp_df["Seasonal"], line={"color": COLOR_PRIMARY_BLUE, "width": 1.5}, name="Seasonal"),
        row=3, col=1,
    )
    # 4. Residual
    fig.add_trace(
        go.Scatter(x=x, y=decomp_df["Residual"], line={"color": COLOR_GOLD, "width": 1.0}, name="Residual"),
        row=4, col=1,
    )

    fig.update_layout(
        paper_bgcolor=COLOR_CARD_BG,
        plot_bgcolor=COLOR_DARK_BG,
        font={"color": COLOR_TEXT, "family": "Inter, sans-serif"},
        height=720,
        showlegend=False,
        margin={"l": 50, "r": 30, "t": 60, "b": 40},
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=COLOR_GRID, row=i, col=1)
        fig.update_yaxes(gridcolor=COLOR_GRID, row=i, col=1)

    return fig


def plot_anomaly_detection(
    annotated_df: pd.DataFrame,
    method_name: str,
    title: str = "Flagged Statistical Climate Anomalies",
) -> go.Figure:
    """
    Plot time series with flagged unusual observations highlighted with red markers.
    """
    fig = go.Figure()

    # Normal line
    fig.add_trace(
        go.Scatter(
            x=annotated_df["Decimal_Year"],
            y=annotated_df["Anomaly"],
            mode="lines",
            name="Normal Observations",
            line={"color": "#64748b", "width": 1.2},
            hovertemplate="Date: %{customdata}<br>Anomaly: %{y:+.2f} °C<extra></extra>",
            customdata=annotated_df["Date"],
        )
    )

    # Flagged anomalies
    anomalies = annotated_df[annotated_df["Is_Anomaly"] == True]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["Decimal_Year"],
                y=anomalies["Anomaly"],
                mode="markers",
                name="Flagged Anomalies",
                marker={
                    "color": "#ef4444",
                    "size": 7.0,
                    "symbol": "circle",
                    "line": {"color": "#fecaca", "width": 1.2},
                },
                hovertemplate="<b>Flagged Anomaly</b><br>Date: %{customdata[0]}<br>Value: %{y:+.2f} °C<br>Severity: %{customdata[1]}<extra></extra>",
                customdata=list(zip(anomalies["Date"], anomalies.get("Severity", ["Flagged"] * len(anomalies)))),
            )
        )

    fig.add_hline(y=0.0, line_dash="dash", line_color="#94a3b8")

    return apply_theme_layout(
        fig,
        title=f"{title} ({method_name})",
        xaxis_title="Year",
        yaxis_title="Temperature Departure (°C)",
    )


def plot_prediction_comparison(
    eval_results: Dict[str, Any],
    title: str = "Chronological Model Holdout Evaluation",
) -> go.Figure:
    """
    Contrast actual observations against out-of-sample model predictions.
    """
    fig = go.Figure()

    dates = eval_results["test_decimal_years"]
    actual = eval_results["actual_test_values"]

    # Actual test observations
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=actual,
            mode="lines+markers",
            name="Actual Observations",
            line={"color": "#f8fafc", "width": 2.2},
            marker={"size": 4},
            hovertemplate="Actual: %{y:+.2f} °C<extra></extra>",
        )
    )

    colors = {
        "Baseline Mean": "#94a3b8",
        "Linear Trend": "#38bdf8",
        "Harmonic Trend": "#f87171",
    }

    for model_name, preds in eval_results["predictions"].items():
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=preds,
                mode="lines",
                name=model_name,
                line={"color": colors.get(model_name, COLOR_GOLD), "width": 2.0, "dash": "dot" if "Baseline" in model_name else "solid"},
                hovertemplate=f"{model_name}: %{{y:+.2f}} °C<extra></extra>",
            )
        )

    return apply_theme_layout(
        fig,
        title=title,
        xaxis_title="Year",
        yaxis_title="Temperature Departure (°C)",
    )
