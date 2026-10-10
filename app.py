"""
SenaGistics - Live dashboard (Streamlit)

Ties together every component of the project:
  - Depot consumption data & charts
  - Forecasting model (LSTM) predictions
  - Anomaly detection
  - Micro-LLM intent classifier (type a message, see it classified live)
  - Autonomous agent recommendation
  - Tamper-evident, encrypted audit trail viewer

Run with:  streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import streamlit as st

from data_sim import simulate_all_depots, DEPOTS
from models import WINDOW, load_forecast_model, load_intent_model, train_and_save_all
from audit import AuditLog, ROLES, is_authorized
from agent import SenaGisticsAgent

st.set_page_config(page_title="SenaGistics", page_icon="\U0001F6E1️", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #eef4f8;
        --muted: #96a8b8;
        --surface: #111e2b;
        --surface-raised: #172737;
        --line: rgba(168, 190, 207, .14);
        --teal: #61e2c0;
        --teal-soft: rgba(97, 226, 192, .12);
    }
    .stApp {
        color: var(--ink);
        background:
            radial-gradient(ellipse at 12% 0%, rgba(20, 105, 111, .20), transparent 36%),
            linear-gradient(145deg, #08121d 0%, #0b1723 55%, #0a1420 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1500px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101e2b 0%, #0b1621 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    h1, h2, h3 { color: var(--ink); letter-spacing: -.025em; }
    h2 { font-size: 1.3rem !important; }
    [data-testid="stTabs"] [role="tablist"] {
        gap: .45rem;
        border-bottom: 1px solid var(--line);
    }
    [data-testid="stTabs"] button[role="tab"] {
        color: var(--muted);
        border-radius: 10px 10px 0 0;
        padding: .7rem 1rem;
        transition: color .18s ease, background .18s ease;
    }
    [data-testid="stTabs"] button[role="tab"]:hover {
        color: var(--ink);
        background: rgba(255, 255, 255, .04);
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--teal);
        background: var(--teal-soft);
    }
    [data-testid="stMetric"] {
        height: 100%;
        padding: 1.05rem 1.15rem;
        background: linear-gradient(145deg, rgba(23, 39, 55, .96), rgba(17, 30, 43, .96));
        border: 1px solid var(--line);
        border-radius: 14px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, .12);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stMetricDelta"] { font-size: .82rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(17, 30, 43, .66);
        border-color: var(--line) !important;
        border-radius: 16px;
    }
    div[data-testid="stAlert"] { border-radius: 12px; }
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background: #0e1a26;
        border-color: var(--line);
        border-radius: 10px;
    }
    div[data-testid="stButton"] button {
        border-radius: 10px;
        font-weight: 650;
        transition: transform .18s ease, box-shadow .18s ease;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        color: #071711;
        background: linear-gradient(120deg, #72edc9, #4ac9b1);
        border: 0;
        box-shadow: 0 8px 20px rgba(73, 208, 177, .18);
    }
    div[data-testid="stButton"] button:hover { transform: translateY(-1px); }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    .sg-hero {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1.5rem;
        padding: 1.8rem 2rem;
        margin: 0 0 1.5rem;
        background:
            radial-gradient(ellipse at 85% 20%, rgba(97, 226, 192, .16), transparent 32%),
            linear-gradient(120deg, rgba(20, 39, 54, .98), rgba(13, 29, 42, .96));
        border: 1px solid rgba(97, 226, 192, .17);
        border-radius: 20px;
        box-shadow: 0 18px 48px rgba(0, 0, 0, .2);
    }
    .sg-kicker {
        color: var(--teal);
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }
    .sg-hero h1 {
        color: var(--ink);
        font-size: clamp(1.8rem, 3vw, 2.55rem);
        line-height: 1.08;
        letter-spacing: -.045em;
        margin: 0;
    }
    .sg-hero p {
        color: #a8bac7;
        font-size: .98rem;
        margin: .65rem 0 0;
    }
    .sg-status {
        flex: 0 0 auto;
        color: #aef5dc;
        background: rgba(74, 201, 177, .11);
        border: 1px solid rgba(97, 226, 192, .23);
        border-radius: 999px;
        padding: .55rem .8rem;
        font-size: .78rem;
        font-weight: 650;
        white-space: nowrap;
    }
    .sg-section-note {
        color: var(--muted);
        margin-top: -.55rem;
        margin-bottom: 1rem;
        font-size: .91rem;
    }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 2rem; }
        .sg-hero { align-items: flex-start; flex-direction: column; padding: 1.35rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_PATHS_EXIST = (
    os.path.exists("saved_models/forecast.keras") and
    os.path.exists("saved_models/intent.keras")
)


@st.cache_resource
def get_models():
    if not MODEL_PATHS_EXIST:
        train_and_save_all(verbose=False)
    fmodel, fstats = load_forecast_model()
    imodel, tok = load_intent_model()
    return fmodel, fstats, imodel, tok


@st.cache_data
def get_data():
    if os.path.exists("depot_consumption.csv"):
        return pd.read_csv("depot_consumption.csv")
    return simulate_all_depots()


@st.cache_resource
def get_audit_log():
    return AuditLog()


fmodel, fstats, imodel, tok = get_models()
data = get_data()
audit_log = get_audit_log()
agent = SenaGisticsAgent(fmodel, fstats, imodel, tok, audit_log)

depot_names = [d[0] for d in DEPOTS]

# ---------------------------------------------------------------------------
# Sidebar - role selection (RBAC) + depot/scenario controls
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🛡️ SenaGistics")
st.sidebar.caption("AUTONOMOUS LOGISTICS CONTROL")
st.sidebar.caption("Group 27 · Operational prototype")

st.sidebar.divider()
st.sidebar.subheader("Session identity")
role = st.sidebar.selectbox("Your role", ROLES, index=1)
user = st.sidebar.text_input("User ID", value="demo_user")

st.sidebar.divider()
st.sidebar.subheader("Scenario controls")
selected_depot = st.sidebar.selectbox("Depot", depot_names)
current_stock = st.sidebar.slider("Current fuel stock (litres)", 200, 10000, 6000, step=100)
force_anomaly = st.sidebar.checkbox("Force an anomaly (demo)", value=False)

st.sidebar.divider()
st.sidebar.caption("MODEL HEALTH")
st.sidebar.caption("Forecast model · ready")
st.sidebar.caption("Validation MAE: %.1f L/day" % fstats["mae"])
st.sidebar.caption("Data source: synthetic")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <section class="sg-hero">
        <div>
            <div class="sg-kicker">SenaGistics · Operations intelligence</div>
            <h1>Consumables command center</h1>
            <p>Secure, real-time fuel and ammunition monitoring across your depots.</p>
        </div>
        <div class="sg-status">● &nbsp;Monitoring active</div>
    </section>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_forecast, tab_agent, tab_audit = st.tabs(
    ["Overview", "Forecast & anomaly", "Agent assistant", "Secure audit trail"]
)

# ---------------------------------------------------------------------------
# Tab 1: Depot overview
# ---------------------------------------------------------------------------

with tab_overview:
    depot_df = data[data.depot == selected_depot]
    avg_fuel = depot_df["fuel_litres"].mean()
    avg_ammo = depot_df["ammo_rounds"].mean()
    n_anomalies = int(depot_df["is_anomaly"].sum())

    st.subheader(f"{selected_depot} overview")
    st.markdown(
        '<p class="sg-section-note">Consumption and anomaly indicators for this depot.</p>',
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average fuel use", f"{avg_fuel:,.0f} L / day")
    col2.metric("Average ammunition use", f"{avg_ammo:,.0f} rounds / day")
    col3.metric("Historical anomalies", n_anomalies, help="Recorded anomaly days in the selected depot's dataset.")
    col4.metric("Days monitored", f"{depot_df['day'].nunique():,}")

    fuel_col, ammo_col = st.columns(2)
    with fuel_col:
        with st.container(border=True):
            st.markdown("#### Fuel consumption")
            st.caption("Daily usage · litres")
            st.line_chart(depot_df.set_index("day")["fuel_litres"], height=280, color="#61e2c0")
    with ammo_col:
        with st.container(border=True):
            st.markdown("#### Ammunition consumption")
            st.caption("Daily usage · rounds")
            st.line_chart(depot_df.set_index("day")["ammo_rounds"], height=280, color="#83a8ff")

    with st.container(border=True):
        st.markdown("#### Depot comparison")
        st.caption("Daily fuel consumption · litres")
        pivot = data.pivot(index="day", columns="depot", values="fuel_litres")
        st.line_chart(pivot, height=300)

# ---------------------------------------------------------------------------
# Tab 2: Forecast & anomaly
# ---------------------------------------------------------------------------

with tab_forecast:
    st.subheader(f"Forecast & anomaly detection")
    st.markdown(
        f'<p class="sg-section-note">Tomorrow\'s fuel estimate for {selected_depot}, based on the most recent {WINDOW}-day window.</p>',
        unsafe_allow_html=True,
    )

    depot_series = data[data.depot == selected_depot]["fuel_litres"].values.astype("float32")
    recent_window = depot_series[-WINDOW:]

    from models import predict_next_day, is_anomalous
    predicted = predict_next_day(fmodel, fstats, recent_window)
    last_actual = recent_window[-1] if not force_anomaly else predicted * 2.8
    anomaly = is_anomalous(last_actual, predicted, fstats)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted tomorrow's usage", f"{predicted:.1f} L")
    c2.metric("Last actual reading", f"{last_actual:.1f} L", delta=f"{last_actual - predicted:+.1f} L vs prediction")
    c3.metric("Anomaly status", "FLAGGED" if anomaly else "Normal")

    days_to_depletion = current_stock / recent_window.mean()
    if anomaly:
        st.warning(
            f"Unexpected usage detected. At the current rate, **{selected_depot}** has an estimated "
            f"**{days_to_depletion:.1f} days** of fuel remaining ({current_stock:,} L in stock)."
        )
    else:
        st.info(
            f"At the current rate, **{selected_depot}** has an estimated **{days_to_depletion:.1f} days** "
            f"of fuel remaining ({current_stock:,} L in stock)."
        )

    with st.container(border=True):
        st.markdown("#### Recent fuel readings")
        st.caption(f"Last {WINDOW} days · actual daily usage in litres")
        chart_df = pd.DataFrame(
            {"Fuel usage (L)": recent_window},
            index=pd.Index(range(1, len(recent_window) + 1), name="Recent day"),
        )
        st.bar_chart(chart_df, height=300, color="#61e2c0")
    st.caption(
        f"Model: LSTM · anomaly threshold: residual greater than 2× validation MAE "
        f"({2 * fstats['mae']:.1f} L/day)."
    )

# ---------------------------------------------------------------------------
# Tab 3: Agent + intent classifier
# ---------------------------------------------------------------------------

with tab_agent:
    st.subheader("Agent assistant")
    st.markdown(
        '<p class="sg-section-note">Send an operator message through the intent classifier and autonomous decision loop.</p>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("#### Analyze an incoming message")
        message = st.text_input(
            "Operator message",
            value="fuel stock will run out in three days at this rate",
            help="Try a status question, procurement request, anomaly report, or resupply alert.",
        )

        run_clicked = st.button("Run agent cycle", type="primary", width="stretch")

    if run_clicked:
        if not is_authorized(role, "log_consumption"):
            st.error(f"Role '{role}' is not authorized to log consumption / trigger an agent cycle.")
        else:
            depot_series = data[data.depot == selected_depot]["fuel_litres"].values.astype("float32")
            recent_window = depot_series[-WINDOW:]

            decision = agent.run_cycle(
                depot_name=selected_depot,
                recent_series=recent_window,
                current_stock=current_stock,
                incoming_message=message,
                user=user,
                role=role,
                force_anomaly=force_anomaly,
            )

            st.success("Agent cycle complete and recorded to the secure audit trail.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Detected intent", decision.intent, delta=f"{decision.intent_confidence:.1%} confidence")
            m2.metric("Days to depletion", f"{decision.days_to_depletion}")
            m3.metric("Anomaly", "Yes" if decision.anomaly_flagged else "No")

            action_colors = {
                "RAISE_PROCUREMENT": "\U0001F7E0",
                "ESCALATE": "\U0001F534",
                "FORWARD": "\U0001F7E1",
                "NO_ACTION": "\U0001F7E2",
            }
            with st.container(border=True):
                st.markdown(
                    f"#### {action_colors.get(decision.recommendation.split()[0], '')} Recommendation"
                )
                st.markdown(f"### {decision.recommendation}")
                st.write(decision.rationale)

            with st.expander("Full decision object"):
                st.json(decision.__dict__)
    else:
        st.caption("The agent checks authorization before running, then records its decision for audit.")

    st.divider()
    st.caption(
        "Intent categories: status check · procurement request · anomaly report · resupply alert. "
        "Use the sidebar demo toggle to test the anomaly escalation path."
    )

# ---------------------------------------------------------------------------
# Tab 4: Audit trail
# ---------------------------------------------------------------------------

with tab_audit:
    st.subheader("Secure audit trail")
    st.markdown(
        '<p class="sg-section-note">Encrypted agent activity with a hash chain for tamper-evident integrity verification.</p>',
        unsafe_allow_html=True,
    )

    valid, broken_at = audit_log.verify_chain()
    if valid:
        st.success("Audit chain verified · no tampering detected.")
    else:
        st.error(f"Chain integrity broken at entry index {broken_at}.")

    if not is_authorized(role, "view_audit_log") and role != "duty_officer":
        st.warning(f"Role '{role}' has limited audit visibility. Switch to 'logistics_command' for full access.")

    entries = audit_log.all_entries_decrypted()
    audit_col1, audit_col2 = st.columns(2)
    audit_col1.metric("Recorded events", len(entries))
    audit_col2.metric("Chain integrity", "Verified" if valid else "Broken")
    if entries:
        rows = []
        for e in entries[::-1]:
            rows.append({
                "index": e["index"],
                "time": pd.to_datetime(e["timestamp"], unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                "role": e["role"],
                "user": e["user"],
                "action": e["action"],
                "depot": e["payload"].get("depot", "-"),
                "recommendation": e["payload"].get("recommendation", "-"),
                "entry_hash": e["entry_hash"][:12] + "...",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No audit entries yet — run an agent cycle from the 'Agent assistant' tab.")

    with st.expander("\U0001F50D What's encrypted under the hood"):
        if entries:
            st.caption("Raw encrypted payload (as stored on disk) vs. decrypted content, for the latest entry:")
            latest = entries[-1]
            st.code(latest["encrypted_payload"][:120] + "...", language="text")
            st.json(latest["payload"])
