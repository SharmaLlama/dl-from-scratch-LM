"""
Neuron-level attention dashboard.
Select a query token to see which key dimensions it fires on,
and how the Q·K dot product decomposes across dimensions.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots


def run_neuron_dashboard(
    attention_weights: np.ndarray,
    queries: np.ndarray,
    keys: np.ndarray,
    tokens: list[str],
    port: int = 8052,
) -> Optional[object]:
    """
    Launch a Dash app showing per-dimension Q/K analysis.

    For a selected (layer, head, query token), shows:
    - Which key tokens receive the most attention
    - The per-dimension contribution Q[q] * K[k] for the top attended key

    Args:
        attention_weights: (n_layers, n_heads, N, N) — post-softmax
        queries: (n_layers, n_heads, N, dk)
        keys: (n_layers, n_heads, N, dk)
        tokens: list of N token strings
        port: localhost port

    Returns:
        IPython IFrame or None.
    """
    n_layers, n_heads, N, dk = queries.shape

    app = Dash(__name__)
    app.layout = html.Div(
        style={"fontFamily": "monospace", "backgroundColor": "#1a1a2e", "color": "#eee", "padding": "20px"},
        children=[
            html.H3("Neuron View — Q·K Dimension Analysis", style={"color": "#e94560"}),
            html.Div(
                style={"display": "flex", "gap": "20px", "marginBottom": "16px"},
                children=[
                    html.Div([
                        html.Label("Layer"),
                        dcc.Dropdown(
                            id="nv-layer",
                            options=[{"label": f"Layer {i}", "value": i} for i in range(n_layers)],
                            value=0,
                            clearable=False,
                            style={"width": "140px", "color": "#333"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Head"),
                        dcc.Dropdown(
                            id="nv-head",
                            options=[{"label": f"Head {h}", "value": h} for h in range(n_heads)],
                            value=0,
                            clearable=False,
                            style={"width": "120px", "color": "#333"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Query token"),
                        dcc.Dropdown(
                            id="nv-token",
                            options=[{"label": f"[{i}] {t}", "value": i} for i, t in enumerate(tokens)],
                            value=min(N - 1, 4),
                            clearable=False,
                            style={"width": "220px", "color": "#333"},
                        ),
                    ]),
                ],
            ),
            html.Div(id="nv-plots"),
        ],
    )

    @app.callback(
        Output("nv-plots", "children"),
        Input("nv-layer", "value"),
        Input("nv-head", "value"),
        Input("nv-token", "value"),
    )
    def update_plots(layer_idx: int, head_idx: int, query_token: int):
        attn_row = attention_weights[layer_idx, head_idx, query_token]   # (N,) — attention weights for this query
        q_vec = queries[layer_idx, head_idx, query_token]                # (dk,)

        # Top-5 attended key tokens
        top_k_indices = np.argsort(attn_row)[::-1][:5]
        top_k_labels = [f"[{i}] {tokens[i]}" for i in top_k_indices]
        top_k_weights = attn_row[top_k_indices]

        # Per-dimension Q·K contribution for the most attended key
        top_key_idx = top_k_indices[0]
        k_vec = keys[layer_idx, head_idx, top_key_idx]   # (dk,)
        dim_contributions = q_vec * k_vec                 # element-wise product

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f"Attention weights from '{tokens[query_token]}'",
                f"Q·K per dimension (query→'{tokens[top_key_idx]}')",
            ],
        )

        # Left: attention weight bar chart
        fig.add_trace(
            go.Bar(
                x=top_k_weights,
                y=top_k_labels,
                orientation="h",
                marker_color="#e94560",
                name="Attention weight",
            ),
            row=1, col=1,
        )

        # Right: per-dimension Q·K contribution
        dim_labels = [f"d{i}" for i in range(dk)]
        colors = ["#4ecca3" if v >= 0 else "#e94560" for v in dim_contributions]
        fig.add_trace(
            go.Bar(
                x=list(range(dk)),
                y=dim_contributions,
                marker_color=colors,
                name="Q·K contribution",
            ),
            row=1, col=2,
        )

        fig.update_layout(
            height=380,
            paper_bgcolor="#16213e",
            plot_bgcolor="#16213e",
            font=dict(color="#eee", size=10),
            margin=dict(l=120, r=20, t=50, b=40),
            showlegend=False,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#333", row=1, col=2)

        # Attention heatmap for this head
        heat_fig = go.Figure(
            go.Heatmap(
                z=attention_weights[layer_idx, head_idx],
                x=tokens,
                y=tokens,
                colorscale="Magma",
                hovertemplate="Query: %{y}<br>Key: %{x}<br>%{z:.3f}<extra></extra>",
            )
        )
        # Highlight selected query row
        heat_fig.add_shape(
            type="rect",
            x0=-0.5, x1=N - 0.5,
            y0=query_token - 0.5, y1=query_token + 0.5,
            line=dict(color="#4ecca3", width=2),
        )
        heat_fig.update_layout(
            title=dict(text=f"Full attention matrix — Layer {layer_idx} Head {head_idx}", font=dict(color="#eee")),
            height=400,
            paper_bgcolor="#16213e",
            plot_bgcolor="#16213e",
            font=dict(color="#eee", size=9),
            xaxis=dict(tickangle=45, showgrid=False),
            yaxis=dict(autorange="reversed", showgrid=False),
            margin=dict(l=80, r=20, t=50, b=80),
        )

        return [
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            dcc.Graph(figure=heat_fig, config={"displayModeBar": False}),
        ]

    _launch_server(app, port)
    return _maybe_iframe(port)


def _launch_server(app: Dash, port: int) -> None:
    thread = threading.Thread(
        target=lambda: app.run(port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    print(f"Neuron dashboard running at http://localhost:{port}")


def _maybe_iframe(port: int):
    try:
        from IPython.display import IFrame
        return IFrame(f"http://localhost:{port}", width="100%", height="900px")
    except ImportError:
        return None
