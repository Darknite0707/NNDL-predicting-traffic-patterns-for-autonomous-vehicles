import math

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


st.set_page_config(
    page_title="Predictive Traffic Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


SEED = 42

N_NODES = 12
HISTORY = 8
STEPS_PER_DAY = 24
N_STEPS = STEPS_PER_DAY * 60
TRAIN_FRACTION = 0.8

FREE_FLOW_SPEED = 60.0
MIN_SPEED = 8.0

NODE_POSITIONS = {
    0: (0.0, 2.0), 1: (1.0, 3.0), 2: (2.0, 3.0), 3: (3.0, 2.0),
    4: (0.0, 1.0), 5: (1.0, 2.0), 6: (2.0, 2.0), 7: (3.0, 1.0),
    8: (1.0, 1.0), 9: (2.0, 1.0), 10: (0.0, 0.0), 11: (3.0, 0.0),
}

ROADS = [
    (0, 1), (1, 2), (2, 3),
    (0, 5), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7),
    (4, 5), (5, 6), (6, 7),
    (4, 8), (5, 8), (6, 9), (7, 9),
    (8, 9), (8, 10), (9, 11),
    (10, 11),
]

ROAD_DETOUR_KM = 0.35

NODE_OFFSET = np.array([0, 1, -1, 2, -2, -7, -6, 1, 2, -2, 3, 0], dtype=float)

SOURCES = {
    "GPS": (0.95, 1.3),
    "IoT": (0.90, 1.8),
    "CCTV": (0.82, 2.3),
    "V2I": (0.93, 1.0),
}
SOURCE_COLS = {name: f"{name.lower()}_speed" for name in SOURCES}
MISSING_RATE = 0.025

INCIDENT_RATE = 0.12
INCIDENT_IMPACT = 18.0
NEIGHBOR_SPILLOVER = 0.35
RAIN_IMPACT = 6.0

CONGESTION_WEIGHT = 0.5
INCIDENT_RISK_MIN = 4.0

FEATURES = ["speed", "incident", "rain", "tod_sin", "tod_cos"]
SPEED_F = FEATURES.index("speed")
INCIDENT_F = FEATURES.index("incident")

EPOCHS = 40
BATCH_SIZE = 64


def distance(u, v):
    (x1, y1), (x2, y2) = NODE_POSITIONS[u], NODE_POSITIONS[v]
    return math.hypot(x2 - x1, y2 - y1)


def congestion_from_speed(speed):
    return np.clip(1.0 - np.asarray(speed) / FREE_FLOW_SPEED, 0.0, 1.0)


def congestion_label(value):
    if value < 0.35:
        return "LOW"
    if value < 0.65:
        return "MEDIUM"
    return "HIGH"


def route_pairs(route):
    return list(zip(route[:-1], route[1:]))


def route_text(route):
    return " → ".join(map(str, route))


def error_metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "MAE (km/h)": float(np.mean(np.abs(err))),
        "RMSE (km/h)": float(np.sqrt(np.mean(err ** 2))),
        "MAPE (%)": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-4)) * 100),
    }


def build_network():
    g = nx.Graph()
    for node, pos in NODE_POSITIONS.items():
        g.add_node(node, pos=pos)
    for u, v in ROADS:
        g.add_edge(u, v, length=distance(u, v) + ROAD_DETOUR_KM)
    return g


G = build_network()


def incident_field(node, severity):
    field = np.zeros(N_NODES)
    field[node] = severity
    for nb in G.neighbors(node):
        field[nb] = max(field[nb], NEIGHBOR_SPILLOVER * severity)
    return field


def normalized_adjacency(g):
    a = nx.to_numpy_array(g, nodelist=range(N_NODES)) + np.eye(N_NODES)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(a.sum(axis=1)))
    return torch.tensor(d_inv_sqrt @ a @ d_inv_sqrt, dtype=torch.float32)


A_HAT = normalized_adjacency(G)


def simulate_ground_truth(rng):
    t = np.arange(N_STEPS)
    nodes = np.arange(N_NODES)

    daily = 7.0 * np.sin(2 * np.pi * t / STEPS_PER_DAY - 1.2)
    local = 3.0 * np.sin(0.45 * t[:, None] + 0.7 * nodes[None, :])
    base = 42.0 + daily[:, None] + local + NODE_OFFSET

    rain = np.zeros(N_STEPS)
    level = 0.0
    for i in range(N_STEPS):
        level = 0.9 * level + rng.normal(0, 0.12)
        rain[i] = np.clip(level, 0.0, 1.0)

    incident = np.zeros((N_STEPS, N_NODES))
    for i in range(N_STEPS):
        if rng.random() < INCIDENT_RATE:
            node = int(rng.integers(N_NODES))
            severity = rng.uniform(0.3, 1.0)
            duration = int(rng.integers(2, 6))
            incident[i:i + duration] = np.maximum(
                incident[i:i + duration], incident_field(node, severity)
            )

    incident[-HISTORY:] = 0.0

    slowdown = np.vstack([np.zeros((1, N_NODES)), incident[:-1]])

    speed = (
        base
        - RAIN_IMPACT * rain[:, None]
        - INCIDENT_IMPACT * slowdown
        + rng.normal(0, 1.0, (N_STEPS, N_NODES))
    )
    return np.clip(speed, MIN_SPEED, FREE_FLOW_SPEED), incident, rain


def fuse_sources(df):
    obs = df[list(SOURCE_COLS.values())].to_numpy()
    weights = np.array([w for w, _ in SOURCES.values()])

    weight_sum = (~np.isnan(obs) * weights).sum(axis=1)
    fused = np.nansum(obs * weights, axis=1) / np.where(weight_sum > 0, weight_sum, np.nan)

    fused = (
        pd.Series(fused, index=df.index)
        .groupby(df["node"])
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )
    return fused.clip(MIN_SPEED, FREE_FLOW_SPEED)


@st.cache_data
def build_dataset(seed=SEED):
    rng = np.random.default_rng(seed)
    speed, incident, rain = simulate_ground_truth(rng)

    time_idx, node_idx = np.meshgrid(np.arange(N_STEPS), np.arange(N_NODES), indexing="ij")
    df = pd.DataFrame({
        "time": time_idx.ravel(),
        "node": node_idx.ravel(),
        "ground_truth_speed": speed.ravel(),
    })

    for name, (_, noise) in SOURCES.items():
        obs = speed + rng.normal(0, noise, speed.shape)
        obs[rng.random(speed.shape) < MISSING_RATE] = np.nan
        df[SOURCE_COLS[name]] = obs.ravel()

    df["rain_intensity"] = np.repeat(np.clip(rain + rng.normal(0, 0.03, N_STEPS), 0, 1), N_NODES)
    df["incident_report"] = incident.ravel()
    df["fused_speed"] = fuse_sources(df)
    df["congestion"] = congestion_from_speed(df["fused_speed"])
    return df


def to_grid(df, col):
    return df.pivot(index="time", columns="node", values=col).to_numpy()


def feature_array(df):
    tod = 2 * np.pi * (np.arange(N_STEPS) % STEPS_PER_DAY) / STEPS_PER_DAY
    per_node = lambda x: np.repeat(x[:, None], N_NODES, axis=1)
    return np.stack(
        [
            to_grid(df, "fused_speed") / FREE_FLOW_SPEED,
            to_grid(df, "incident_report"),
            to_grid(df, "rain_intensity"),
            per_node(np.sin(tod)),
            per_node(np.cos(tod)),
        ],
        axis=-1,
    ).astype(np.float32)


DATA = build_dataset()
FEATURE_GRID = feature_array(DATA)
NOW = N_STEPS - 1


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, a_hat):
        return self.linear(torch.einsum("ij,btjf->btif", a_hat, x))


class STGNN(nn.Module):
    def __init__(self, in_features, hidden=32):
        super().__init__()
        self.gc1 = GraphConv(in_features, hidden)
        self.gc2 = GraphConv(hidden, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, a_hat):
        h = F.relu(self.gc1(x, a_hat))
        h = F.relu(self.gc2(h, a_hat))

        b, t, n, f = h.shape
        h = h.permute(0, 2, 1, 3).reshape(b * n, t, f)
        _, last = self.gru(h)

        delta = self.head(last[0]).reshape(b, n)
        return x[:, -1, :, SPEED_F] + delta


def make_windows():
    n = N_STEPS - HISTORY
    x = np.stack([FEATURE_GRID[i:i + HISTORY] for i in range(n)])
    y = FEATURE_GRID[HISTORY:, :, SPEED_F]
    truth = to_grid(DATA, "ground_truth_speed")[HISTORY:]
    return torch.from_numpy(x), torch.from_numpy(y), truth


@st.cache_resource
def train_model(seed=SEED):
    torch.manual_seed(seed)
    x, y, truth = make_windows()
    split = int(TRAIN_FRACTION * len(x))

    model = STGNN(len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(EPOCHS):
        for idx in torch.randperm(split, generator=generator).split(BATCH_SIZE):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x[idx], A_HAT), y[idx])
            loss.backward()
            optimizer.step()
    model.eval()

    with torch.no_grad():
        stgnn_pred = model(x[split:], A_HAT).numpy() * FREE_FLOW_SPEED
    persistence_pred = x[split:, -1, :, SPEED_F].numpy() * FREE_FLOW_SPEED
    test_truth = truth[split:]

    metrics = pd.DataFrame(
        [
            {"Model": "ST-GNN", **error_metrics(test_truth, stgnn_pred)},
            {"Model": "Persistence (last value)", **error_metrics(test_truth, persistence_pred)},
        ]
    )
    return model, metrics, len(x) - split


MODEL, TEST_METRICS, N_TEST = train_model()


def current_state():
    window = FEATURE_GRID[NOW]
    speed = window[:, SPEED_F] * FREE_FLOW_SPEED
    return pd.DataFrame({
        "node": np.arange(N_NODES),
        "speed": speed,
        "congestion": congestion_from_speed(speed),
        "incident": window[:, INCIDENT_F],
    })


def predict_next(incident=None):
    window = FEATURE_GRID[-HISTORY:].copy()
    if incident is not None:
        window[-1, :, INCIDENT_F] = np.maximum(window[-1, :, INCIDENT_F], incident_field(*incident))

    with torch.no_grad():
        speed = MODEL(torch.from_numpy(window)[None], A_HAT)[0].numpy() * FREE_FLOW_SPEED
    speed = np.clip(speed, MIN_SPEED, FREE_FLOW_SPEED)

    return pd.DataFrame({
        "node": np.arange(N_NODES),
        "speed": speed,
        "congestion": congestion_from_speed(speed),
        "incident": window[-1, :, INCIDENT_F],
    })


CURRENT_STATE = current_state()


def traffic_graph(state):
    s = state.set_index("node")
    g = nx.Graph()

    for u, v, road in G.edges(data=True):
        speed = max((s.at[u, "speed"] + s.at[v, "speed"]) / 2.0, MIN_SPEED)
        congestion = (s.at[u, "congestion"] + s.at[v, "congestion"]) / 2.0
        exposure = max(s.at[u, "incident"], s.at[v, "incident"])

        travel_time = road["length"] / speed * 60.0
        congestion_penalty = CONGESTION_WEIGHT * congestion * travel_time
        incident_risk = INCIDENT_RISK_MIN * exposure

        g.add_edge(
            u, v,
            length=road["length"],
            speed=speed,
            congestion=congestion,
            travel_time=travel_time,
            congestion_penalty=congestion_penalty,
            incident_risk=incident_risk,
            cost=travel_time + congestion_penalty + incident_risk,
        )
    return g


def heuristic(u, v):
    return distance(u, v) / FREE_FLOW_SPEED * 60.0


def plan_route(g, start, destination):
    return nx.astar_path(g, start, destination, heuristic=heuristic, weight="cost")


def route_breakdown(g, route):
    rows = []
    for u, v in route_pairs(route):
        e = g[u][v]
        rows.append({
            "Road": f"{u} → {v}",
            "Length (km)": e["length"],
            "Speed (km/h)": e["speed"],
            "Congestion": e["congestion"],
            "Travel time (min)": e["travel_time"],
            "Congestion penalty": e["congestion_penalty"],
            "Incident risk": e["incident_risk"],
            "A* cost": e["cost"],
        })
    return pd.DataFrame(rows)


def total_time(breakdown):
    return float(breakdown["Travel time (min)"].sum())


def total_cost(breakdown):
    return float(breakdown["A* cost"].sum())


def compare_routes(start, destination, incident=None):
    future_state = predict_next(incident)
    now_graph = traffic_graph(CURRENT_STATE)
    future_graph = traffic_graph(future_state)

    current_route = plan_route(now_graph, start, destination)
    predicted_route = plan_route(future_graph, start, destination)

    return {
        "future_state": future_state,
        "current_route": current_route,
        "predicted_route": predicted_route,
        "current_now": route_breakdown(now_graph, current_route),
        "current_future": route_breakdown(future_graph, current_route),
        "predicted_future": route_breakdown(future_graph, predicted_route),
    }


@st.cache_data
def find_demo_incident(start, destination):
    best = None
    for node in G.nodes():
        if node in (start, destination):
            continue
        for severity in (0.6, 0.8, 1.0):
            result = compare_routes(start, destination, (node, severity))
            saved = total_time(result["current_future"]) - total_time(result["predicted_future"])
            rerouted = result["predicted_route"] != result["current_route"]
            if rerouted and saved > 0 and (best is None or saved > best[2]):
                best = (node, severity, saved)
    return best


def network_figure(state, current_route, predicted_route, incident=None):
    fig = go.Figure()

    def add_lines(pairs, name, color, width, dash=None, showlegend=True):
        xs, ys = [], []
        for u, v in pairs:
            xs += [NODE_POSITIONS[u][0], NODE_POSITIONS[v][0], None]
            ys += [NODE_POSITIONS[u][1], NODE_POSITIONS[v][1], None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=name,
            line=dict(color=color, width=width, dash=dash),
            hoverinfo="skip", showlegend=showlegend,
        ))

    add_lines(G.edges(), "Road", "rgba(140, 150, 165, 0.45)", 4, showlegend=False)
    add_lines(route_pairs(current_route), "Current route", "#3b82f6", 10)
    add_lines(route_pairs(predicted_route), "Predicted route", "#ff8c42", 5, dash="dash")

    s = state.set_index("node")
    hover = [
        f"Node {n}<br>"
        f"Predicted speed: {s.at[n, 'speed']:.1f} km/h<br>"
        f"Predicted congestion: {s.at[n, 'congestion']:.2f} ({congestion_label(s.at[n, 'congestion'])})"
        for n in G.nodes()
    ]
    fig.add_trace(go.Scatter(
        x=[NODE_POSITIONS[n][0] for n in G.nodes()],
        y=[NODE_POSITIONS[n][1] for n in G.nodes()],
        mode="markers+text",
        text=[str(n) for n in G.nodes()],
        textfont=dict(color="#111", size=13),
        marker=dict(
            size=34,
            color=[s.at[n, "congestion"] for n in G.nodes()],
            colorscale="RdYlGn_r", cmin=0, cmax=1,
            line=dict(color="white", width=1.5),
            colorbar=dict(title="Predicted<br>congestion"),
        ),
        hovertext=hover, hoverinfo="text", showlegend=False,
    ))

    if incident is not None:
        x, y = NODE_POSITIONS[incident[0]]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", name="What-if incident",
            marker=dict(size=54, color="rgba(255, 80, 80, 0.2)", line=dict(color="#ff5555", width=3)),
            hoverinfo="skip",
        ))

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False, range=[-0.6, 3.6]),
        yaxis=dict(visible=False, range=[-0.6, 3.6], scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", y=1.04, x=0),
    )
    return fig


def node_prediction_chart(node, baseline, scenario=None, lookback=24):
    history = DATA[(DATA["node"] == node) & (DATA["time"] > NOW - lookback)]
    last_speed = float(history["fused_speed"].iloc[-1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["time"], y=history["fused_speed"],
        mode="lines+markers", name="Observed (fused)",
        line=dict(color="#3b82f6", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[NOW, NOW + 1], y=[last_speed, baseline.at[node, "speed"]],
        mode="lines+markers", name="Prediction (no incident)",
        line=dict(color="#8ea0b8", width=3, dash="dot"),
    ))
    if scenario is not None:
        fig.add_trace(go.Scatter(
            x=[NOW, NOW + 1], y=[last_speed, scenario.at[node, "speed"]],
            mode="lines+markers", name="Prediction (with what-if incident)",
            line=dict(color="#ff8c42", width=3, dash="dash"),
        ))

    fig.update_layout(
        title=f"Node {node}: recent speed and next-step prediction",
        xaxis_title="Time step (hour)",
        yaxis_title="Speed (km/h)",
        height=360,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def fusion_chart(node, lookback=48):
    window = DATA[(DATA["node"] == node) & (DATA["time"] > NOW - lookback)]

    fig = go.Figure()
    for name, col in SOURCE_COLS.items():
        fig.add_trace(go.Scatter(
            x=window["time"], y=window[col], mode="lines", name=name,
            line=dict(width=1), opacity=0.6,
        ))
    fig.add_trace(go.Scatter(
        x=window["time"], y=window["fused_speed"], mode="lines", name="Fused",
        line=dict(width=3),
    ))
    fig.add_trace(go.Scatter(
        x=window["time"], y=window["ground_truth_speed"], mode="lines", name="Ground truth",
        line=dict(width=2, dash="dot", color="#888"),
    ))
    fig.update_layout(
        title=f"Node {node}: raw sources vs fused speed",
        xaxis_title="Time step (hour)",
        yaxis_title="Speed (km/h)",
        height=400,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


node_options = list(G.nodes())
node_name = lambda n: f"Node {n}"

st.sidebar.title("⚙️ Simulation Controls")
st.sidebar.caption("Simulated data. Decision support only.")

st.sidebar.subheader("Route")
start_node = st.sidebar.selectbox("Start", node_options, index=0, format_func=node_name)
destination_node = st.sidebar.selectbox("Destination", node_options, index=7, format_func=node_name)

if start_node == destination_node:
    st.sidebar.error("Start and destination must be different.")
    st.error("Start and destination must be different. Choose another node in the sidebar.")
    st.stop()

st.sidebar.subheader("What-if incident")
incident_mode = st.sidebar.radio(
    "Scenario",
    ["Auto (best demo)", "Manual", "None"],
    help=(
        "An incident report is injected into the latest observation and the "
        "ST-GNN predicts how it affects traffic in the next step."
    ),
)

incident = None
if incident_mode == "Auto (best demo)":
    demo = find_demo_incident(start_node, destination_node)
    if demo is None:
        st.sidebar.info("No single incident changes this route. Try Manual mode.")
    else:
        incident = (demo[0], demo[1])
        st.sidebar.caption(
            f"Selected: incident at **Node {demo[0]}**, severity **{demo[1]:.0%}**."
        )
elif incident_mode == "Manual":
    incident_node = st.sidebar.selectbox(
        "Incident node", node_options, index=6, format_func=node_name
    )
    incident_severity = st.sidebar.slider(
        "Severity", min_value=0.1, max_value=1.0, value=0.8, step=0.05,
        help="0.1 = minor slowdown, 1.0 = severe incident.",
    )
    incident = (incident_node, incident_severity)


result = compare_routes(start_node, destination_node, incident)
baseline_state = predict_next() if incident is not None else result["future_state"]

current_route = result["current_route"]
predicted_route = result["predicted_route"]
rerouted = current_route != predicted_route

time_now = total_time(result["current_now"])
time_keep = total_time(result["current_future"])
time_predicted = total_time(result["predicted_future"])
time_saved = time_keep - time_predicted


st.title("🚗 Predictive Traffic Intelligence")
st.caption(
    "Multi-source fusion → ST-GNN traffic prediction → traffic-aware A* routing. "
    "The system recommends routes; it does not control a vehicle."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Current route — now", f"{time_now:.2f} min")
k2.metric(
    "Current route — predicted",
    f"{time_keep:.2f} min",
    delta=f"{time_keep - time_now:+.2f} min",
    delta_color="inverse",
)
k3.metric(
    "Predicted route — predicted",
    f"{time_predicted:.2f} min",
    delta=f"{-time_saved:+.2f} min",
    delta_color="inverse",
)
k4.metric("Rerouted", "YES" if rerouted else "NO")

if rerouted and time_saved > 0:
    st.success(
        f"**Proactive reroute.** The ST-GNN predicts a slowdown on "
        f"{route_text(current_route)}, so A* switches to "
        f"**{route_text(predicted_route)}**, saving **{time_saved:.2f} min** "
        f"({time_saved / time_keep:.0%}) compared with staying on the current route."
    )
elif rerouted:
    st.warning(
        f"**Rerouted for risk, not time.** {route_text(predicted_route)} avoids "
        f"incident and congestion exposure but isn't faster "
        f"({time_predicted:.2f} vs {time_keep:.2f} min)."
    )
elif incident is not None:
    st.info(
        f"**No reroute.** Even with the what-if incident, "
        f"{route_text(current_route)} is still the best route under predicted traffic."
    )
elif incident_mode == "Auto (best demo)":
    st.info(
        f"**No reroute.** {route_text(current_route)} is the best route, and no single "
        f"what-if incident makes another route faster for this trip."
    )
else:
    st.info(
        f"**No reroute.** {route_text(current_route)} is the best route now and "
        f"under predicted traffic. Add a what-if incident to see proactive rerouting."
    )


tab_network, tab_prediction, tab_fusion, tab_routing, tab_data = st.tabs([
    "🗺️ Network & Routes",
    "🧠 ST-GNN Prediction",
    "🔗 Multi-Source Fusion",
    "🧭 A* Route Analysis",
    "📊 Data",
])

with tab_network:
    st.caption(
        "Nodes are intersections and edges are roads. Node colour shows predicted congestion. "
        "Blue = route chosen on current traffic; orange dashed = route chosen on predicted traffic."
    )
    st.plotly_chart(
        network_figure(result["future_state"], current_route, predicted_route, incident),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.dataframe(
        pd.DataFrame({
            "Scenario": [
                "Current route, current traffic",
                "Current route, predicted traffic",
                "Predicted route, predicted traffic",
            ],
            "Path": [route_text(current_route), route_text(current_route), route_text(predicted_route)],
            "Travel time (min)": [time_now, time_keep, time_predicted],
            "A* cost": [
                total_cost(result["current_now"]),
                total_cost(result["current_future"]),
                total_cost(result["predicted_future"]),
            ],
        }).round(2),
        use_container_width=True,
        hide_index=True,
    )

    if incident is not None:
        st.caption(
            f"🚨 What-if scenario: incident reported at Node {incident[0]} with severity "
            f"{incident[1]:.0%}. This is a controlled demonstration, not a real incident feed."
        )

with tab_prediction:
    st.markdown(
        "**Model:** GraphConv → GraphConv → per-node GRU → next-step speed  \n"
        "**Inputs (last 8 hours, per node):** fused speed, incident reports, rain intensity, time of day  \n"
        "**Output:** next-hour speed, converted to congestion for routing"
    )

    st.markdown(f"**Held-out test set** (last {1 - TRAIN_FRACTION:.0%} of the timeline, {N_TEST} windows)")
    st.dataframe(TEST_METRICS.round(2), use_container_width=True, hide_index=True)
    stgnn_mae, persistence_mae = TEST_METRICS["MAE (km/h)"]
    st.caption(
        f"The ST-GNN's MAE is {1 - stgnn_mae / persistence_mae:.0%} lower than simply "
        f"repeating the last observed speed. Errors are measured against simulated "
        f"ground truth and are not real-world accuracy."
    )

    selected_node = st.selectbox(
        "Inspect node", node_options,
        index=incident[0] if incident is not None else 5,
        format_func=node_name,
    )
    baseline_by_node = baseline_state.set_index("node")
    scenario_by_node = result["future_state"].set_index("node") if incident is not None else None
    st.plotly_chart(
        node_prediction_chart(selected_node, baseline_by_node, scenario_by_node),
        use_container_width=True,
    )

    prediction_table = pd.DataFrame({
        "Node": CURRENT_STATE["node"],
        "Current speed (km/h)": CURRENT_STATE["speed"],
        "Predicted speed (km/h)": result["future_state"]["speed"],
        "Change (km/h)": result["future_state"]["speed"] - CURRENT_STATE["speed"],
        "Predicted congestion": result["future_state"]["congestion"],
        "Level": result["future_state"]["congestion"].apply(congestion_label),
    })
    st.dataframe(prediction_table.round(2), use_container_width=True, hide_index=True)

with tab_fusion:
    st.markdown(
        "Each source reports noisy speeds and occasionally drops readings. The fused speed is a "
        "reliability-weighted average of whichever sources reported at that time and node."
    )

    source_rows = []
    for name, (weight, _) in SOURCES.items():
        col = SOURCE_COLS[name]
        source_rows.append({
            "Source": name,
            "Reliability weight": weight,
            "Missing (%)": DATA[col].isna().mean() * 100,
            "MAE vs ground truth (km/h)": (DATA[col] - DATA["ground_truth_speed"]).abs().mean(),
        })
    source_rows.append({
        "Source": "Fused",
        "Reliability weight": np.nan,
        "Missing (%)": DATA["fused_speed"].isna().mean() * 100,
        "MAE vs ground truth (km/h)": (DATA["fused_speed"] - DATA["ground_truth_speed"]).abs().mean(),
    })
    st.dataframe(pd.DataFrame(source_rows).round(3), use_container_width=True, hide_index=True)

    fusion_node = st.selectbox("Node", node_options, index=5, format_func=node_name, key="fusion_node")
    st.plotly_chart(fusion_chart(fusion_node), use_container_width=True)

with tab_routing:
    st.markdown(
        "**A* edge cost** = travel time "
        f"+ {CONGESTION_WEIGHT} × congestion × travel time "
        f"+ {INCIDENT_RISK_MIN:g} min × incident exposure  \n"
        "Travel time is reported separately in minutes; A* cost is an optimisation score."
    )

    for title, route, key in [
        ("Current route under current traffic", current_route, "current_now"),
        ("Current route under predicted traffic", current_route, "current_future"),
        ("Predicted route under predicted traffic", predicted_route, "predicted_future"),
    ]:
        st.markdown(f"#### {title}: {route_text(route)}")
        st.dataframe(result[key].round(3), use_container_width=True, hide_index=True)

with tab_data:
    st.write("Most recent rows of the unified multi-source dataset used by the prediction pipeline.")
    st.dataframe(DATA.tail(N_NODES * 6).round(3), use_container_width=True, hide_index=True)


st.markdown("---")
st.caption(
    "Academic prototype — Predicting Traffic Patterns for Autonomous Navigation. "
    "Decision-support only; no direct autonomous vehicle control."
)
