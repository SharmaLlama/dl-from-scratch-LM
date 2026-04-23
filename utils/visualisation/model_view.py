"""
Model-level attention dashboard.
Compact layers × heads grid; click a cell to expand the full heatmap.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update
from plotly.subplots import make_subplots


def run_model_dashboard(
    attention_weights: np.ndarray,
    tokens: list[str],
    port: int = 8051,
    colormap: str = "Viridis",
) -> Optional[object]:
    """
    Launch a Dash app showing a compact grid of all layers and heads.
    Click any cell to expand it to a full-size heatmap.

    Args:
        attention_weights: (n_layers, n_heads, N, N)
        tokens: list of N token strings
        port: localhost port
        colormap: plotly colorscale name

    Returns:
        IPython IFrame or None.
    """
    n_layers, n_heads, N, _ = attention_weights.shape

    # Pre-compute the compact grid figure
    grid_fig = make_subplots(
        rows=n_layers,
        cols=n_heads,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
        subplot_titles=[
            f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)
        ],
    )

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            attn = attention_weights[layer_idx, head_idx]
            grid_fig.add_trace(
                go.Heatmap(
                    z=attn,
                    colorscale=colormap,
                    showscale=False,
                    name=f"L{layer_idx}H{head_idx}",
                    hovertemplate=f"L{layer_idx} H{head_idx}<br>Q: %{{y}}<br>K: %{{x}}<br>%{{z:.3f}}<extra></extra>",
                ),
                row=layer_idx + 1,
                col=head_idx + 1,
            )

    grid_height = max(400, n_layers * 80)
    grid_fig.update_layout(
        height=grid_height,
        paper_bgcolor="#16213e",
        plot_bgcolor="#16213e",
        font=dict(color="#eee", size=8),
        margin=dict(l=40, r=10, t=30, b=10),
    )
    grid_fig.update_xaxes(showticklabels=False, showgrid=False)
    grid_fig.update_yaxes(showticklabels=False, showgrid=False, autorange="reversed")

    app = Dash(__name__)
    app.layout = html.Div(
        style={"fontFamily": "monospace", "backgroundColor": "#1a1a2e", "color": "#eee", "padding": "20px"},
        children=[
            html.H3("Model View — All Layers & Heads", style={"color": "#e94560"}),
            html.P("Click any subplot to expand.", style={"color": "#aaa", "fontSize": "12px"}),
            dcc.Graph(
                id="grid-graph",
                figure=grid_fig,
                config={"displayModeBar": False},
                style={"marginBottom": "20px"},
            ),
            html.Div(id="expanded-view"),
        ],
    )

    @app.callback(
        Output("expanded-view", "children"),
        Input("grid-graph", "clickData"),
        prevent_initial_call=True,
    )
    def expand_cell(click_data):
        if click_data is None:
            return no_update

        trace_name: str = click_data["points"][0].get("curveNumber", None)
        # Extract layer/head from the trace index
        trace_idx = click_data["points"][0]["curveNumber"]
        layer_idx = trace_idx // n_heads
        head_idx = trace_idx % n_heads

        attn = attention_weights[layer_idx, head_idx]
        fig = go.Figure(
            go.Heatmap(
                z=attn,
                x=tokens,
                y=tokens,
                colorscale=colormap,
                hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(text=f"Layer {layer_idx} · Head {head_idx}", font=dict(color="#eee")),
            height=500,
            paper_bgcolor="#16213e",
            plot_bgcolor="#16213e",
            font=dict(color="#eee", size=10),
            xaxis=dict(tickangle=45, showgrid=False),
            yaxis=dict(autorange="reversed", showgrid=False),
            margin=dict(l=80, r=20, t=50, b=80),
        )
        return dcc.Graph(figure=fig, config={"displayModeBar": True})

    _launch_server(app, port)
    return _maybe_iframe(port)


def _launch_server(app: Dash, port: int) -> None:
    thread = threading.Thread(
        target=lambda: app.run(port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    print(f"Model dashboard running at http://localhost:{port}")


def _maybe_iframe(port: int):
    try:
        from IPython.display import IFrame
        return IFrame(f"http://localhost:{port}", width="100%", height="700px")
    except ImportError:
        return None
