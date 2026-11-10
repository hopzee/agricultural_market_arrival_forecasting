"""Plotly charts used by the Streamlit dashboard."""
import plotly.express as px
import pandas as pd


def arrivals_chart(series: pd.Series, commodity: str, state: str):
    frame = series.rename("arrivals_t").rename_axis("month").reset_index()
    fig = px.line(frame, x="month", y="arrivals_t", title=f"Monthly market arrivals: {commodity} — {state}")
    fig.update_layout(xaxis_title="Month", yaxis_title="Arrivals (tonnes)", hovermode="x unified")
    return fig


def forecast_chart(history: pd.Series, forecast: pd.Series, commodity: str, state: str):
    hist = history.rename("Historical arrivals").rename_axis("month").reset_index()
    fut = forecast.rename("Forecast arrivals").rename_axis("month").reset_index()
    fig = px.line(hist, x="month", y="Historical arrivals", title=f"Historical and forecast arrivals: {commodity} — {state}")
    fig.add_scatter(x=fut["month"], y=fut["Forecast arrivals"], mode="lines+markers", name="Forecast")
    fig.update_layout(xaxis_title="Month", yaxis_title="Arrivals (tonnes)", hovermode="x unified")
    return fig


def comparison_chart(metrics: pd.DataFrame, x="Model", y="Test MAE"):
    usable = metrics.dropna(subset=[y])
    fig = px.bar(usable, x=x, y=y, title=f"Model comparison — {y}")
    fig.update_layout(xaxis_title="Model", yaxis_title=y)
    return fig
