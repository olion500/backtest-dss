"""Shared helpers for building equity vs. price charts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import altair as alt
import pandas as pd


@dataclass
class ExtraLineConfig:
    series: pd.Series
    column_name: str
    title: str
    color: str = 'green'
    stroke_dash: Sequence[int] | None = None
    include_in_tooltip: bool = True

default_height = 400


@dataclass
class EquityPriceChartConfig:
    target_label: str
    price_axis_label: str | None = None
    log_scale: bool = True
    height: int = default_height


def _coerce_series(series: pd.Series | pd.DataFrame | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    if isinstance(series, pd.DataFrame):
        if 'Close' in series.columns:
            return series['Close'].copy()
        return series.squeeze("columns").copy()
    return series.copy()


def prepare_equity_price_frames(
    equity: pd.Series,
    price_series: pd.Series | pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if equity is None or equity.empty:
        return pd.DataFrame(columns=['Date', 'Equity']), pd.DataFrame()

    price = _coerce_series(price_series)
    if price.empty:
        eq_df = equity.reset_index()
        eq_df.columns = ['Date', 'Equity']
        return eq_df, pd.DataFrame()

    eq_df = equity.reset_index()
    eq_df.columns = ['Date', 'Equity']
    price = price.dropna()
    price_df = price.reset_index()
    price_df.columns = ['Date', 'Price']
    combined = pd.merge(eq_df, price_df, on='Date', how='inner')
    return eq_df, combined


def build_equity_price_chart(
    eq_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    config: EquityPriceChartConfig,
    extra_lines: Iterable[ExtraLineConfig] | None = None,
    mode_backgrounds: pd.DataFrame | None = None,
) -> alt.Chart | None:
    if eq_df.empty:
        return None

    scale_type = 'log' if config.log_scale else 'linear'
    price_axis_title = config.price_axis_label or f"{config.target_label} Price ($)"

    if combined_df.empty:
        return (
            alt.Chart(eq_df)
            .mark_line(color='steelblue', strokeWidth=2)
            .encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Equity:Q', title='Equity ($)', scale=alt.Scale(type=scale_type)),
                tooltip=[
                    alt.Tooltip('Date:T', format='%Y-%m-%d'),
                    alt.Tooltip('Equity:Q', format='$,.0f'),
                ],
            )
            .properties(height=config.height)
            .interactive()
        )

    combined = combined_df.copy()
    combined['Date_Next'] = combined['Date'].shift(-1)
    combined['Date_Next'] = combined['Date_Next'].fillna(combined['Date'] + pd.Timedelta(days=1))

    extra_lines = list(extra_lines or [])
    extra_tooltips: list[alt.Tooltip] = []
    for line in extra_lines:
        aligned = line.series.reindex(combined['Date']).astype(float)
        combined[line.column_name] = aligned
        if line.include_in_tooltip:
            extra_tooltips.append(
                alt.Tooltip(
                    f'{line.column_name}:Q',
                    format='$,.2f',
                    title=line.title,
                )
            )

    tooltip_fields = [
        alt.Tooltip('Date:T', format='%Y-%m-%d'),
        alt.Tooltip('Equity:Q', format='$,.0f', title='Equity'),
        alt.Tooltip('Price:Q', format='$,.2f', title=price_axis_title),
        *extra_tooltips,
    ]

    base = alt.Chart(combined).encode(x=alt.X('Date:T', title='Date'))

    equity_line = base.mark_line(color='steelblue', strokeWidth=2).encode(
        y=alt.Y(
            'Equity:Q',
            title='Strategy Equity ($)',
            scale=alt.Scale(type=scale_type),
            axis=alt.Axis(titleColor='steelblue', format='$,.0f'),
        ),
        tooltip=tooltip_fields,
    )

    price_line = base.mark_line(color='orange', strokeWidth=2).encode(
        y=alt.Y(
            'Price:Q',
            title=price_axis_title,
            scale=alt.Scale(type=scale_type),
            axis=alt.Axis(titleColor='orange', orient='right', format='$,.2f'),
        ),
        tooltip=tooltip_fields,
    )

    extra_layers = []
    for line in extra_lines:
        if line.column_name not in combined.columns:
            continue
        extra_layers.append(
            base.mark_line(color=line.color, strokeWidth=1.2, strokeDash=line.stroke_dash).encode(
                y=alt.Y(
                    f'{line.column_name}:Q',
                    title=line.title,
                    scale=alt.Scale(type=scale_type),
                    axis=alt.Axis(titleColor=line.color, orient='right', format='$,.2f'),
                ),
                tooltip=tooltip_fields,
            )
        )

    hover_overlay = base.mark_rect(opacity=0).encode(
        x='Date:T',
        x2='Date_Next:T',
        tooltip=tooltip_fields,
    )

    mode_layer = []
    if mode_backgrounds is not None and not mode_backgrounds.empty:
        mode_rects = (
            alt.Chart(mode_backgrounds)
            .mark_rect(opacity=0.08)
            .encode(
                x='start:T',
                x2='end:T',
                color=alt.Color(
                    'mode:N',
                    scale=alt.Scale(
                        domain=['공세', '안전'],
                        range=['#ff6b6b', '#4dabf7'],
                    ),
                    legend=alt.Legend(title='모드'),
                ),
            )
        )
        mode_layer = [mode_rects]

    layers = [*mode_layer, equity_line, price_line, *extra_layers, hover_overlay]
    return alt.layer(*layers).resolve_scale(y='independent').properties(height=config.height).interactive()
