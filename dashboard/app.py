import os
import requests
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="F&B Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "forecast_result" not in st.session_state:
    st.session_state["forecast_result"] = None

if "inventory_result" not in st.session_state:
    st.session_state["inventory_result"] = None


# ============================================================
# API HELPERS
# ============================================================

def get_api(endpoint: str):
    response = requests.get(
        f"{API_BASE_URL}{endpoint}",
        timeout=60,
    )

    response.raise_for_status()

    return response.json()

@st.cache_data(ttl=10)
def get_health():
    return get_api("/health")


@st.cache_data(ttl=10)
def get_readiness():
    return get_api("/ready")


@st.cache_data(ttl=10)
def get_model_info():
    return get_api("/model-info")


@st.cache_data(ttl=10)
def get_metrics():
    return get_api("/metrics")


def post_api(endpoint: str, payload: dict):
    response = requests.post(
        f"{API_BASE_URL}{endpoint}",
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


@st.cache_data(ttl=300)
def get_branches():
    result = get_api("/branches")
    return result["branches"]


@st.cache_data(ttl=300)
def get_products():
    result = get_api("/products")
    return result["products"]


def get_error_message(exc):
    try:
        return exc.response.json()["detail"]
    except Exception:
        return str(exc)


# ============================================================
# HEADER
# ============================================================

st.title("🍽️ F&B Intelligence")

st.caption(
    "AI-powered decision support system for Food & Beverage operations"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    try:

        available_branches = get_branches()
        available_products = get_products()

    except requests.RequestException:

        st.error(
            "Unable to load branch/product data from FastAPI."
        )

        st.stop()

    branch_id = st.selectbox(
        "Branch",
        options=available_branches,
        index=0,
    )

    product_id = st.selectbox(
        "Product",
        options=available_products,
        index=23 if len(available_products) > 23 else 0,
    )

    st.divider()

    st.subheader("Forecast Settings")

    horizon = st.slider(
        "Forecast Horizon",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of future days to forecast.",
    )

    discount = st.number_input(
        "Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=5.0,
    )

    payday = st.checkbox(
        "Payday",
        value=False,
    )

    holiday = st.checkbox(
        "Holiday",
        value=False,
    )

    st.divider()

    st.caption(
        "Connected to FastAPI"
    )

    st.code(
        API_BASE_URL,
        language="text",
    )


# ============================================================
# API STATUS
# ============================================================

try:

    health = get_api("/health")

    api_online = True

except requests.RequestException:

    api_online = False


if api_online:

    st.success(
        f"API Connected • {health['version']}"
    )

else:

    st.error(
        "FastAPI is not available. "
        "Start the API server first."
    )

    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

st.header("📊 Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "API Status",
        "ONLINE",
    )

with col2:

    st.metric(
        "Branch",
        branch_id,
    )

with col3:

    st.metric(
        "Product",
        product_id,
    )

with col4:

    st.metric(
        "Forecast Horizon",
        f"{horizon} days",
    )

# ============================================================
# System Monitoring
# ============================================================

st.subheader("System Monitoring")

try:
    health = get_health()
    readiness = get_readiness()
    model_info = get_model_info()
    metrics = get_metrics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "API Status",
            health.get("status", "unknown").upper(),
        )

    with col2:
        st.metric(
            "Readiness",
            readiness.get("status", "unknown").upper(),
        )

    with col3:
        st.metric(
            "Model Version",
            model_info.get("version", "unknown"),
        )

    with col4:
        st.metric(
            "Total Requests",
            metrics.get("total_requests", 0),
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Successful Requests",
            metrics.get("successful_requests", 0),
        )

    with col2:
        st.metric(
            "Failed Requests",
            metrics.get("failed_requests", 0),
        )

    with col3:
        st.metric(
            "Avg Latency",
            f"{metrics.get('average_latency_seconds', 0):.3f} s",
        )

    st.caption(
        f"Active model: "
        f"{model_info.get('model_name', 'unknown')} "
        f"@ {model_info.get('alias', 'unknown')}"
    )

except Exception as exc:
    st.error(f"Monitoring unavailable: {exc}")

# ============================================================
# FUTURE FORECAST
# ============================================================

st.divider()

st.header("📈 Future Demand Forecast")

st.write(
    "Generate recursive demand forecasts using the deployed XGBoost model."
)

start_date = st.date_input(
    "Forecast Start Date",
    value=pd.Timestamp("2026-01-01"),
)


# ============================================================
# GENERATE FORECAST
# ============================================================

if st.button(
    "🚀 Generate Forecast",
    type="primary",
    use_container_width=True,
):

    payload = {
        "branch_id": branch_id,
        "product_id": product_id,
        "start_date": start_date.isoformat(),
        "horizon": horizon,
        "discount_percentage": discount,
        "is_payday": int(payday),
        "is_holiday": int(holiday),
    }

    try:

        with st.spinner(
            "Generating forecast..."
        ):

            result = post_api(
                "/forecast/future",
                payload,
            )

        # Store complete API response
        st.session_state["forecast_result"] = result

        st.success(
            "Forecast generated successfully."
        )

    except requests.HTTPError as exc:

        st.error(
            f"Forecast request failed: "
            f"{get_error_message(exc)}"
        )

    except requests.RequestException as exc:

        st.error(
            f"Unable to connect to API: {exc}"
        )

    except Exception as exc:

        st.error(
            f"Unexpected error: {exc}"
        )


# ============================================================
# FORECAST RESULT
# ============================================================

forecast_result = st.session_state["forecast_result"]


if forecast_result is not None:

    result = forecast_result

    forecast_data = result["forecast"]

    forecast_df = pd.DataFrame(
        forecast_data
    )

    forecast_df["date"] = pd.to_datetime(
        forecast_df["date"]
    )

    forecast_df = forecast_df.sort_values(
        "date"
    ).reset_index(drop=True)

    # ========================================================
    # FORECAST CONTEXT
    # ========================================================

    st.subheader("Forecast Context")

    context_col1, context_col2, context_col3, context_col4 = (
        st.columns(4)
    )

    with context_col1:

        st.write("**Branch**")
        st.write(result["branch_id"])

    with context_col2:

        st.write("**Product**")
        st.write(result["product_id"])

    with context_col3:

        st.write("**Start Date**")
        st.write(result["start_date"])

    with context_col4:

        st.write("**Model**")
        st.write(result["model"]["name"])

    st.divider()

    # ========================================================
    # FORECAST KPI
    # ========================================================

    total_quantity = forecast_df[
        "predicted_quantity"
    ].sum()

    average_quantity = forecast_df[
        "predicted_quantity"
    ].mean()

    maximum_quantity = forecast_df[
        "predicted_quantity"
    ].max()

    minimum_quantity = forecast_df[
        "predicted_quantity"
    ].min()

    peak_index = forecast_df[
        "predicted_quantity"
    ].idxmax()

    peak_date = forecast_df.loc[
        peak_index,
        "date",
    ]

    peak_date_text = peak_date.strftime(
        "%d %b %Y"
    )

    # ========================================================
    # KPI CARDS
    # ========================================================

    st.subheader("Forecast Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Forecast",
            f"{total_quantity:,.2f}",
            help=(
                "Total predicted demand across "
                "the selected forecast horizon."
            ),
        )

    with col2:

        st.metric(
            "Average / Day",
            f"{average_quantity:,.2f}",
            help=(
                "Average predicted daily demand."
            ),
        )

    with col3:

        st.metric(
            "Peak Demand",
            f"{maximum_quantity:,.2f}",
            help=(
                "Highest predicted demand during "
                "the forecast horizon."
            ),
        )

    with col4:

        st.metric(
            "Peak Date",
            peak_date_text,
            help=(
                "Date with the highest predicted demand."
            ),
        )

    # ========================================================
    # OPERATIONAL INSIGHT
    # ========================================================

    st.subheader("💡 Operational Insight")

    demand_range = (
        maximum_quantity - minimum_quantity
    )

    if average_quantity > 0:

        peak_ratio = (
            maximum_quantity / average_quantity
        )

    else:

        peak_ratio = 0

    if peak_ratio >= 1.30:

        insight = (
            f"Demand is expected to peak at "
            f"**{maximum_quantity:,.2f} units** on "
            f"**{peak_date_text}**, approximately "
            f"**{(peak_ratio - 1) * 100:.1f}% above "
            f"the forecast average**. "
            f"Operations should prepare additional "
            f"inventory and capacity around this date."
        )

    elif demand_range <= max(
        5,
        average_quantity * 0.10,
    ):

        insight = (
            f"Demand is relatively stable throughout "
            f"the forecast horizon, averaging "
            f"**{average_quantity:,.2f} units/day**. "
            f"No major demand spike is currently indicated."
        )

    else:

        insight = (
            f"Forecast demand averages "
            f"**{average_quantity:,.2f} units/day** "
            f"and reaches a maximum of "
            f"**{maximum_quantity:,.2f} units** on "
            f"**{peak_date_text}**. "
            f"Operations should pay closer attention "
            f"to demand around the projected peak."
        )

    st.info(insight)

    # ========================================================
    # CHART
    # ========================================================

    st.subheader(
        "Forecasted Demand"
    )

    chart_df = forecast_df.set_index(
        "date"
    )

    st.line_chart(
        chart_df[
            ["predicted_quantity"]
        ],
        use_container_width=True,
    )

    # ========================================================
    # TABLE
    # ========================================================

    st.subheader(
        "Forecast Details"
    )

    display_df = forecast_df.copy()

    display_df["date"] = (
        display_df["date"]
        .dt.strftime("%Y-%m-%d")
    )

    display_df = display_df.rename(
        columns={
            "date": "Date",
            "predicted_quantity":
                "Predicted Quantity",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with st.expander(
        "Model Information"
    ):

        st.write(
            f"**Model:** "
            f"{result['model']['name']}"
        )

        st.write(
            f"**Type:** "
            f"{result['model']['type']}"
        )

        st.write(
            f"**Branch:** "
            f"{result['branch_id']}"
        )

        st.write(
            f"**Product:** "
            f"{result['product_id']}"
        )

        st.write(
            f"**Start Date:** "
            f"{result['start_date']}"
        )

        st.write(
            f"**Horizon:** "
            f"{result['horizon']} days"
        )

        st.write(
            "**Assumptions:**"
        )

        st.json(
            result["assumptions"]
        )

else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.info(
        "No forecast generated yet. "
        "Configure the parameters in the sidebar "
        "and click **Generate Forecast**."
    )


# ============================================================
# INVENTORY
# ============================================================

st.divider()

st.header("📦 Inventory Optimization")

st.write(
    "Current inventory policy and recommended ingredient orders."
)


if st.button(
    "🔄 Load Inventory Recommendations",
    use_container_width=True,
):

    try:

        with st.spinner(
            "Loading inventory recommendations..."
        ):

            inventory_result = get_api(
                "/inventory/recommend"
            )

        st.session_state["inventory_result"] = (
            inventory_result
        )

        st.success(
            "Inventory recommendations loaded successfully."
        )

    except requests.HTTPError as exc:

        st.error(
            f"Inventory request failed: "
            f"{get_error_message(exc)}"
        )

    except requests.RequestException as exc:

        st.error(
            f"Unable to connect to API: {exc}"
        )

    except Exception as exc:

        st.error(
            f"Unexpected error: {exc}"
        )


# ============================================================
# INVENTORY RESULT
# ============================================================

inventory_result = st.session_state[
    "inventory_result"
]


if inventory_result is not None:

    policy = inventory_result["policy"]
    summary = inventory_result["summary"]

    # ========================================================
    # POLICY
    # ========================================================

    st.info(
        f"**Policy:** {policy['name']}  |  "
        f"Target service level: "
        f"{policy['target_service_level']:.0%}  |  "
        f"Review period: "
        f"{policy['review_period_days']} day"
    )

    # ========================================================
    # KPIs
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Orders",
            summary[
                "order_recommendations"
            ],
        )

    with col2:

        st.metric(
            "Hold",
            summary[
                "hold_recommendations"
            ],
        )

    with col3:

        st.metric(
            "Recommended Qty",
            f"{summary['total_recommended_quantity']:,.2f}",
        )

    with col4:

        st.metric(
            "Estimated Value",
            f"Rp {summary['total_estimated_order_value']:,.0f}",
        )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.subheader(
        "Recommended Orders"
    )

    recommendation_df = pd.DataFrame(
        inventory_result[
            "recommendations"
        ]
    )

    st.dataframe(
        recommendation_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No inventory recommendation loaded yet. "
        "Click **Load Inventory Recommendations** "
        "to retrieve the latest recommendation."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "F&B Intelligence • AI Engineering Portfolio Project"
)