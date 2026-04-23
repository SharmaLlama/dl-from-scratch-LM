"""
Head-level attention dashboard.
One layer at a time; toggle individual heads on/off.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


def run_head_dashboard(
    attention_weights: np.ndarray,
    tokens: list[str],
    port: int = 8050,
    colormap: str = "Magma",
) -> Optional[object]:
    """
    Launch a Dash app visualising per-head attention for a single sequence.

    Args:
        attention_weights: (n_layers, n_heads, N, N)
        tokens: list of N token strings
        port: localhost port to serve on
        colormap: plotly colorscale name

    Returns:
        IPython IFrame when running in Jupyter, else None.
    """
    n_layers, n_heads, N, _ = attention_weights.shape

    app = Dash(__name__)
    app.layout = html.Div(
        style={"fontFamily": "monospace", "backgroundColor": "#1a1a2e", "color": "#eee", "padding": "20px"},
        children=[
            html.H3("Head View — Causal Self-Attention", style={"color": "#e94560"}),
            html.Div(
                style={"display": "flex", "gap": "20px", "marginBottom": "16px"},
                children=[
                    html.Div([
                        html.Label("Layer", style={"marginBottom": "4px"}),
                        dcc.Dropdown(
                            id="layer-select",
                            options=[{"label": f"Layer {i}", "value": i} for i in range(n_layers)],
                            value=0,
                            clearable=False,
                            style={"width": "160px", "color": "#333"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Heads", style={"marginBottom": "4px"}),
                        dcc.Checklist(
                            id="head-select",
                            options=[{"label": f" H{h}", "value": h} for h in range(n_heads)],
                            value=list(range(min(n_heads, 4))),
                            inline=True,
                            style={"display": "flex", "flexWrap": "wrap", "gap": "8px"},
                        ),
                    ]),
                ],
            ),
            html.Div(id="heatmaps"),
        ],
    )

    @app.callback(
        Output("heatmaps", "children"),
        Input("layer-select", "value"),
        Input("head-select", "value"),
    )
    def update_heatmaps(layer_idx: int, selected_heads: list[int]):
        if not selected_heads:
            return html.P("Select at least one head.")

        selected_heads = sorted(selected_heads)
        cols = min(len(selected_heads), 4)
        rows = (len(selected_heads) + cols - 1) // cols

        figs = []
        for head_idx in selected_heads:
            attn = attention_weights[layer_idx, head_idx]  # (N, N)
            fig = go.Figure(
                go.Heatmap(
                    z=attn,
                    x=tokens,
                    y=tokens,
                    colorscale=colormap,
                    showscale=False,
                    hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.3f}<extra></extra>",
                )
            )
            fig.update_layout(
                title=dict(text=f"Head {head_idx}", font=dict(size=12)),
                margin=dict(l=60, r=10, t=30, b=60),
                height=280,
                paper_bgcolor="#16213e",
                plot_bgcolor="#16213e",
                font=dict(color="#eee", size=9),
                xaxis=dict(tickangle=45, showgrid=False),
                yaxis=dict(autorange="reversed", showgrid=False),
            )
            figs.append(
                html.Div(
                    dcc.Graph(figure=fig, config={"displayModeBar": False}),
                    style={"flex": "1 1 200px", "minWidth": "200px"},
                )
            )

        return html.Div(figs, style={"display": "flex", "flexWrap": "wrap", "gap": "8px"})

    _launch_server(app, port)
    return _maybe_iframe(port)


def _launch_server(app: Dash, port: int) -> None:
    thread = threading.Thread(
        target=lambda: app.run(port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    print(f"Head dashboard running at http://localhost:{port}")


def _maybe_iframe(port: int):
    try:
        from IPython.display import IFrame
        return IFrame(f"http://localhost:{port}", width="100%", height="600px")
    except ImportError:
        return None
