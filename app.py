import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="StemShield | Cardamom Borer Risk",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        :root {
            --forest: #123c2a;
            --leaf: #2f7d4a;
            --mint: #dff3e4;
            --ink: #18312a;
            --muted: #6c7e76;
            --line: #dce9e1;
        }
        [data-testid="stAppViewContainer"] { background: #f7faf8; }
        [data-testid="stSidebar"] { background: #102f24; }
        [data-testid="stSidebar"] * { color: #e9f5ed !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #1b4936;
            border-color: #477c60;
        }
        .block-container {
            max-width: 1440px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        .hero {
            background: linear-gradient(120deg, #123c2a 0%, #23633e 62%, #4a9860 100%);
            border-radius: 22px;
            color: #ffffff;
            padding: 2rem 2.25rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 14px 30px rgba(18, 60, 42, 0.16);
        }
        .hero-kicker {
            color: #bde6c6;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }
        .hero h1 {
            color: #ffffff;
            font-size: clamp(2rem, 4vw, 3.4rem);
            letter-spacing: -0.04em;
            line-height: 1;
            margin: 0 0 0.8rem;
        }
        .hero p {
            color: #dbf1df;
            font-size: 1rem;
            margin: 0;
            max-width: 680px;
        }
        .section-label {
            color: var(--leaf);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin: 1.3rem 0 0.45rem;
        }
        .risk-panel {
            border-radius: 18px;
            padding: 1.2rem 1.35rem;
            min-height: 150px;
            border: 1px solid var(--line);
            background: #ffffff;
        }
        .risk-panel.critical {
            background: #fff5f3;
            border-color: #f2b8ad;
        }
        .risk-panel.moderate {
            background: #fff9ea;
            border-color: #efd38c;
        }
        .risk-panel.safe {
            background: #f0faf3;
            border-color: #b8dfc1;
        }
        .risk-label {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .risk-score {
            color: #18312a;
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.06em;
            line-height: 1.05;
            margin: 0.2rem 0;
        }
        .risk-status {
            color: var(--leaf);
            font-weight: 750;
        }
        .critical .risk-status {
            color: #bf3e2c;
        }
        .moderate .risk-status {
            color: #a66a00;
        }
        .info-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.15rem 1.25rem;
            height: 100%;
        }
        .info-card h3 {
            color: #18312a;
            font-size: 1rem;
            margin: 0 0 0.55rem;
        }
        .info-card p {
            color: var(--muted);
            font-size: 0.91rem;
            line-height: 1.55;
            margin: 0;
        }
        .action-card {
            background: #ffffff;
            border-left: 4px solid var(--leaf);
            border-radius: 0 14px 14px 0;
            box-shadow: 0 4px 14px rgba(18, 60, 42, 0.06);
            margin: 0.6rem 0;
            padding: 0.95rem 1.1rem;
        }
        .action-card strong {
            color: #18312a;
        }
        .action-card span {
            color: var(--muted);
        }
        .small-note {
            color: var(--muted);
            font-size: 0.78rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }
        div[data-testid="stTabs"] button {
            color: var(--muted);
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--leaf);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


CARDAMOM_LOCATIONS = {
    "Udumbanchola · Idukki": {"lat": 9.88, "lon": 77.18},
    "Nedumkandam · Idukki": {"lat": 9.85, "lon": 77.16},
    "Kattappana · Idukki": {"lat": 9.77, "lon": 77.08},
    "Vandanmedu · Idukki": {"lat": 9.80, "lon": 77.13},
    "Munnar · High Range": {"lat": 10.08, "lon": 77.06},
    "Rajakumari · Idukki": {"lat": 9.94, "lon": 77.10},
}


def location_seed(location: str) -> int:
    """Create a stable seed so the same demo location stays reproducible."""
    return int(hashlib.sha256(location.encode("utf-8")).hexdigest()[:8], 16)


@st.cache_data(ttl=1800, show_spinner=False)
def generate_simulation_data(location: str) -> pd.DataFrame:
    """Generate realistic high-range microclimate data for offline demos."""
    rng = np.random.default_rng(location_seed(location))
    periods = 144

    dates = pd.date_range(
        end=pd.Timestamp.now().floor("h") + pd.Timedelta(days=3),
        periods=periods,
        freq="h",
    )

    phase = np.linspace(-0.8, 4.4 * np.pi, periods)
    temperature = (
        21.2
        + 4.4 * np.sin(phase)
        + rng.normal(0, 0.55, periods)
    )
    humidity = (
        81
        + 12 * np.cos(phase)
        + rng.normal(0, 2.0, periods)
    )

    precipitation = rng.choice(
        [0.0, 0.0, 0.0, 0.4, 1.8, 4.0],
        size=periods,
        p=[0.62, 0.13, 0.08, 0.08, 0.07, 0.02],
    )

    return pd.DataFrame(
        {
            "time": dates,
            "temperature": np.round(temperature, 1),
            "humidity": np.clip(np.round(humidity, 1), 45, 99),
            "precipitation": precipitation,
        }
    )


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather_data(
    location: str,
    lat: float,
    lon: float,
    use_offline_sim: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Fetch five-day weather data, falling back to deterministic demo data."""
    if use_offline_sim:
        return generate_simulation_data(location), True

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation"
        "&forecast_days=5&timezone=Asia%2FKolkata"
    )

    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()

        data = response.json()
        hourly = data["hourly"]

        weather = pd.DataFrame(
            {
                "time": pd.to_datetime(hourly["time"]),
                "temperature": hourly["temperature_2m"],
                "humidity": hourly["relative_humidity_2m"],
                "precipitation": hourly["precipitation"],
            }
        )

        return weather, False

    except (
        requests.RequestException,
        KeyError,
        TypeError,
        ValueError,
    ):
        return generate_simulation_data(location), True


def calculate_borer_risk(
    weather: pd.DataFrame,
    base_temp: float = 10.0,
    target_gdd: float = 110.0,
) -> tuple[pd.DataFrame, dict]:
    """Calculate degree-day development and climate suitability risk."""
    hourly = weather.copy().sort_values("time").reset_index(drop=True)

    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly["temperature"] = pd.to_numeric(
        hourly["temperature"],
        errors="coerce",
    )
    hourly["humidity"] = pd.to_numeric(
        hourly["humidity"],
        errors="coerce",
    )
    hourly["precipitation"] = pd.to_numeric(
        hourly["precipitation"],
        errors="coerce",
    ).fillna(0)

    hourly = hourly.dropna(
        subset=["temperature", "humidity"]
    )

    hourly["degree_hours"] = (
        hourly["temperature"] - base_temp
    ).clip(lower=0)

    hourly["gdd"] = hourly["degree_hours"] / 24
    hourly["cumulative_gdd"] = hourly["gdd"].cumsum()
    hourly["wet_hour"] = hourly["precipitation"] > 0.1

    recent = hourly.tail(48)

    temp_suitability = (
        (
            (recent["temperature"] >= 18)
            & (recent["temperature"] <= 29)
        ).mean()
        * 100
    )

    humidity_suitability = (
        (
            (recent["humidity"] >= 75)
            & (recent["humidity"] <= 96)
        ).mean()
        * 100
    )

    wetness = min(
        100.0,
        recent["wet_hour"].mean() * 170,
    )

    development = min(
        100.0,
        hourly["cumulative_gdd"].iloc[-1] / target_gdd * 100,
    )

    score = round(
        0.35 * temp_suitability
        + 0.25 * humidity_suitability
        + 0.15 * wetness
        + 0.25 * development
    )

    if score >= 72:
        status = "Critical"
        status_class = "critical"
    elif score >= 45:
        status = "Moderate"
        status_class = "moderate"
    else:
        status = "Low"
        status_class = "safe"

    gdd_total = float(
        hourly["cumulative_gdd"].iloc[-1]
    )

    if gdd_total >= target_gdd:
        stage = "Adult emergence window"
        stage_detail = (
            "Thermal accumulation has crossed the emergence threshold."
        )
    elif gdd_total >= target_gdd * 0.68:
        stage = "Late larval development"
        stage_detail = (
            "The crop is approaching the highest-concern window."
        )
    elif gdd_total >= target_gdd * 0.35:
        stage = "Active larval development"
        stage_detail = (
            "Conditions support continued borer development."
        )
    else:
        stage = "Early development"
        stage_detail = (
            "Thermal accumulation is still building."
        )

    avg_gdd = max(
        float(hourly["gdd"].tail(48).mean()),
        0.01,
    )

    remaining_gdd = max(
        target_gdd - gdd_total,
        0,
    )

    emergence_days = remaining_gdd / (avg_gdd * 24)

    emergence_label = (
        "Now / active window"
        if remaining_gdd == 0
        else f"~{max(1, round(emergence_days))} days"
    )

    summary = {
        "score": score,
        "status": status,
        "status_class": status_class,
        "stage": stage,
        "stage_detail": stage_detail,
        "gdd_total": gdd_total,
        "emergence": emergence_label,
        "temperature": float(
            recent["temperature"].mean()
        ),
        "humidity": float(
            recent["humidity"].mean()
        ),
        "rainfall": float(
            hourly["precipitation"].sum()
        ),
        "temp_suitability": round(temp_suitability),
        "humidity_suitability": round(humidity_suitability),
        "wetness": round(wetness),
        "development": round(development),
    }

    return hourly, summary


def build_actions(summary: dict) -> list[tuple[str, str]]:
    """Return practical scouting and intervention steps."""
    actions = [
        (
            "Scout 20 clumps",
            (
                "Inspect inner pseudostems and look for frass, "
                "entry holes, or wilting shoots."
            ),
        ),
        (
            "Log hotspots",
            (
                "Mark affected clumps with a flag so repeat scouting "
                "can verify spread."
            ),
        ),
    ]

    if summary["status"] == "Critical":
        actions.insert(
            0,
            (
                "Prioritize intervention",
                (
                    "Coordinate with your field advisor on an approved "
                    "control before the next peak window."
                ),
            ),
        )
    elif summary["status"] == "Moderate":
        actions.insert(
            0,
            (
                "Increase scouting cadence",
                (
                    "Move to twice-weekly checks while humidity and "
                    "thermal accumulation stay favorable."
                ),
            ),
        )
    else:
        actions.insert(
            0,
            (
                "Keep monitoring",
                (
                    "Maintain weekly scouting and revisit the risk "
                    "signal after the next rain event."
                ),
            ),
        )

    return actions


def render_metric_card(
    label: str,
    value: str,
    helper: str,
) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="risk-label">{label}</div>
            <div class="risk-score" style="font-size: 2rem;">
                {value}
            </div>
            <p>{helper}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("## 🌿 StemShield")
    st.caption("Cardamom borer lifecycle intelligence")
    st.divider()

    selected_location = st.selectbox(
        "Plantation zone",
        options=list(CARDAMOM_LOCATIONS),
        index=0,
        help=(
            "Choose the nearest weather point for your cardamom block."
        ),
    )

    offline_mode = st.toggle(
        "Use offline demo data",
        value=False,
        help=(
            "Uses a stable simulated microclimate series when internet "
            "access is unavailable."
        ),
    )

    st.divider()
    st.markdown("### Model settings")

    base_temperature = st.slider(
        "Development base temperature (°C)",
        min_value=6.0,
        max_value=14.0,
        value=10.0,
        step=0.5,
    )

    target_gdd = st.slider(
        "Emergence threshold (GDD)",
        min_value=80.0,
        max_value=150.0,
        value=110.0,
        step=5.0,
    )

    st.divider()
    st.caption("Model note")
    st.caption(
        "Risk is a decision-support signal based on degree-day "
        "accumulation, temperature, humidity, and wet-hour suitability. "
        "Confirm field actions with local agronomy guidance."
    )


coordinates = CARDAMOM_LOCATIONS[selected_location]

weather, is_offline = fetch_weather_data(
    selected_location,
    coordinates["lat"],
    coordinates["lon"],
    offline_mode,
)

hourly, risk = calculate_borer_risk(
    weather,
    base_temp=base_temperature,
    target_gdd=target_gdd,
)

actions = build_actions(risk)


st.markdown(
    f"""
    <section class="hero">
        <div class="hero-kicker">
            Lifecycle risk engine · High Range, Kerala
        </div>
        <h1>Know the window.<br>Protect the crop.</h1>
        <p>
            Translate microclimate signals into a clear scouting plan for
            <strong>{selected_location}</strong>.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


source_label = (
    "Offline demo simulation"
    if is_offline
    else "Open-Meteo live forecast"
)

source_color = "#a66a00" if is_offline else "#2f7d4a"

st.markdown(
    f"""
    <p class="small-note">
        ●
        <span style="color:{source_color};font-weight:700;">
            {source_label}
        </span>
        &nbsp;·&nbsp;
        Last refreshed {datetime.now().strftime('%d %b %Y, %H:%M')} IST
    </p>
    """,
    unsafe_allow_html=True,
)


left, right = st.columns(
    [1.2, 1],
    gap="large",
)

with left:
    st.markdown(
        '<div class="section-label">Current signal</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="risk-panel {risk['status_class']}">
            <div class="risk-label">Borer risk index</div>
            <div class="risk-score">
                {risk['score']}
                <span style="font-size:1.2rem;color:#6c7e76;">
                    / 100
                </span>
            </div>
            <div class="risk-status">
                {risk['status']} · {risk['stage']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        '<div class="section-label">What the model sees</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="info-card">
            <h3>{risk['stage']}</h3>
            <p>
                {risk['stage_detail']}
                Estimated emergence:
                <strong>{risk['emergence']}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="section-label">'
    "Field snapshot · trailing 48 hours"
    "</div>",
    unsafe_allow_html=True,
)

metric_columns = st.columns(4, gap="medium")

with metric_columns[0]:
    render_metric_card(
        "Thermal accumulation",
        f"{risk['gdd_total']:.1f} GDD",
        f"Target: {target_gdd:.0f} GDD",
    )

with metric_columns[1]:
    render_metric_card(
        "Mean temperature",
        f"{risk['temperature']:.1f}°C",
        "Development sweet spot: 18–29°C",
    )

with metric_columns[2]:
    render_metric_card(
        "Mean humidity",
        f"{risk['humidity']:.0f}%",
        "Favorable band: 75–96%",
    )

with metric_columns[3]:
    render_metric_card(
        "Rainfall signal",
        f"{risk['rainfall']:.1f} mm",
        "Across the available forecast window",
    )


overview_tab, climate_tab, action_tab = st.tabs(
    ["Risk overview", "Climate signal", "Field plan"]
)


with overview_tab:
    chart_left, chart_right = st.columns(
        [1.35, 1],
        gap="large",
    )

    with chart_left:
        st.markdown("#### Thermal accumulation")

        chart_data = hourly.set_index("time")[
            ["cumulative_gdd"]
        ].rename(
            columns={
                "cumulative_gdd": "Cumulative GDD",
            }
        )

        st.line_chart(
            chart_data,
            height=310,
            color="#2f7d4a",
        )

        st.caption(
            "The threshold is configurable in the sidebar. "
            "Crossing it indicates a higher-priority emergence window."
        )

    with chart_right:
        st.markdown("#### Risk components")

        components = pd.DataFrame(
            {
                "Component": [
                    "Temperature",
                    "Humidity",
                    "Wetness",
                    "Development",
                ],
                "Signal": [
                    risk["temp_suitability"],
                    risk["humidity_suitability"],
                    risk["wetness"],
                    risk["development"],
                ],
            }
        ).set_index("Component")

        st.bar_chart(
            components,
            height=310,
            color="#86b98f",
        )

        st.caption(
            "Each component is normalized to 100 and combined into "
            "the risk index."
        )


with climate_tab:
    st.markdown("#### Temperature and humidity profile")

    climate = hourly.set_index("time")[
        ["temperature", "humidity"]
    ].rename(
        columns={
            "temperature": "Temperature (°C)",
            "humidity": "Humidity (%)",
        }
    )

    st.line_chart(
        climate,
        height=350,
        color=["#d77a3d", "#4d8eac"],
    )

    st.markdown("#### Weather data")

    display_data = hourly[
        [
            "time",
            "temperature",
            "humidity",
            "precipitation",
            "gdd",
            "cumulative_gdd",
        ]
    ].tail(48).copy()

    display_data.columns = [
        "Time",
        "Temperature °C",
        "Humidity %",
        "Rain mm",
        "Hourly GDD",
        "Cumulative GDD",
    ]

    st.dataframe(
        display_data.round(
            {
                "Temperature °C": 1,
                "Humidity %": 1,
                "Rain mm": 1,
                "Hourly GDD": 2,
                "Cumulative GDD": 1,
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download weather data (CSV)",
        data=hourly.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"stemshield_"
            f"{selected_location.split(' · ')[0].lower()}"
            "_weather.csv"
        ),
        mime="text/csv",
    )


with action_tab:
    st.markdown("#### Recommended field plan")
    st.caption(
        "Use the plan as a prioritized checklist. "
        "It does not replace local agronomy advice."
    )

    for title, detail in actions:
        st.markdown(
            f"""
            <div class="action-card">
                <strong>{title}</strong><br>
                <span>{detail}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Signal breakdown")

    signal_columns = st.columns(4, gap="medium")

    for column, label, value in zip(
        signal_columns,
        [
            "Temperature fit",
            "Humidity fit",
            "Wet-hour signal",
            "Development",
        ],
        [
            risk["temp_suitability"],
            risk["humidity_suitability"],
            risk["wetness"],
            risk["development"],
        ],
    ):
        with column:
            st.metric(label, f"{value}/100")


st.divider()

st.markdown(
    '<p class="small-note">'
    "StemShield is an early-warning dashboard for field scouting. "
    "Weather data can change; use current local observations before "
    "making treatment decisions."
    "</p>",
    unsafe_allow_html=True,
)