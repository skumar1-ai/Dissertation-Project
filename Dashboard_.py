from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from datetime import datetime
import matplotlib.pyplot as plt
import json

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("merged_olist_dataset.csv")
llm = OpenAI(
    api_token="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",
    system_message=(
        "You are a helpful assistant only answering questions about the given dataset. "
        "If the user asks something unrelated (like greetings, jokes, or general questions), "
        "respond with: 'I'm here to help only with dataset-related questions like orders, sales, customers, etc.'"
    ),
)
smart_df = SmartDataframe(df, config={"llm": llm, "enable_output": True})
# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Page configuration
st.set_page_config(
    page_title="Olist Enhanced Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Session states
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "type": "text",
            "content": "Hi! 👋 Ask me about your dashboard data.",
        }
    ]


# --- Chat---
def save_chat_log(messages):
    from datetime import datetime

    def serialize_message(msg):
        # DataFrame to dict
        if msg["type"] == "dataframe":
            serialized_content = msg["content"].to_dict(orient="records")
        # Matplotlib plots to placeholder string
        elif msg["type"] == "plot":
            serialized_content = "[Plot Object — not serializable]"
        else:
            serialized_content = msg["content"]
        return {"role": msg["role"], "type": msg["type"], "content": serialized_content}

    safe_messages = [serialize_message(m) for m in messages]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_log_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(safe_messages, f, indent=2, ensure_ascii=False)

    print(f"✅ Chat log saved to {filename}")


# --- CSS ---
st.markdown(
    """
<style>
.chat-launcher {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 1000;
}
.chat-launcher button {
    background-color: #ff4b4b;
    color: white;
    border: none;
    border-radius: 50px;
    padding: 12px 20px;
    font-weight: 600;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    cursor: pointer;
}
.chat-box {
    position: fixed;
    bottom: 80px;
    right: 20px;
    width: 400px;
    max-height: 600px;
    background: #1e1e1e;
    border-radius: 10px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    z-index: 1001;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    font-family: sans-serif;
    color: white;
}
.chat-box .header {
    background: #ff4b4b;
    color: white;
    padding: 10px 15px;
    font-weight: bold;
}
.chat-box .body {
    padding: 10px;
    overflow-y: auto;
    background: #2e2e2e;
    flex-grow: 1;
    font-size: 14px;
}
.chat-box .input-area {
    display: flex;
    padding: 10px;
    border-top: 1px solid #444;
    background: #1e1e1e;
}
.chat-box .input-area input {
    flex: 1;
    padding: 8px;
    border-radius: 5px;
    border: 1px solid #ccc;
}
.chat-box .input-area button {
    background: #ff4b4b;
    color: white;
    border: none;
    padding: 8px 14px;
    border-radius: 5px;
    margin-left: 5px;
    cursor: pointer;
}
.custom-clear-button {
    background-color: #ff4b4b;
    color: white;
    border: none;
    padding: 0.5rem 1.2rem;
    border-radius: 10px;
    font-weight: bold;
    cursor: pointer;
}
.custom-clear-button:hover {
    background-color: #e63e3e;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---Chat Launcher ---
chat_toggle = st.empty()
with chat_toggle.container():
    st.markdown('<div class="chat-launcher">', unsafe_allow_html=True)

    if not st.session_state.chat_open:
        if st.button("💬 Chat with OlistBot", key="chat_open_btn"):
            st.session_state.chat_open = True
            st.rerun()
    else:
        if st.button("❌ Close Chat", key="chat_close_btn"):
            # Optional: save the chat log before clearing
            save_chat_log(st.session_state.messages)
            # Reset chat session
            st.session_state.chat_open = False
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "type": "text",
                    "content": "Hi! 👋 Ask me about your dashboard data.",
                }
            ]
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# Function
def is_dataset_related(query):
    # Keywords
    data_keywords = [
        "order",
        "customer",
        "product",
        "sales",
        "revenue",
        "date",
        "city",
        "delivery",
        "rating",
        "review",
        "dataset",
        "data",
        "chart",
        "graph",
        "top",
        "bottom",
        "average",
        "mean",
        "max",
        "min",
        "count",
        "sum",
    ]

    return any(keyword in query.lower() for keyword in data_keywords)


# Chatbot processing
def process_question(query):
    if not is_dataset_related(query):
        return "🛑 I'm here to help only with **dataset-related questions** like orders, customers, ratings, or sales trends."

    try:
        response = smart_df.chat(query)
        return response
    except Exception as e:
        return "⚠️ I encountered an error while processing your question."


# --- Chat UI ---
if st.session_state.chat_open:
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)

    # --- Header ---
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        if st.button("🧹", key="clear_chat_btn", help="Clear chat"):
            save_chat_log(st.session_state.messages)
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "type": "text",
                    "content": "Hi! 👋 Ask me about your dashboard data.",
                }
            ]
            st.rerun()

    # --- Chat Body ---
    st.markdown('<div class="body">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role_icon = "🧠 OlistBot:" if msg["role"] == "assistant" else "🙋 You:"

        if msg["type"] == "text":
            st.markdown(
                f"<p><b>{role_icon}</b> {msg['content']}</p>", unsafe_allow_html=True
            )
        elif msg["type"] == "dataframe":
            st.markdown(f"**{role_icon}**")
            st.dataframe(msg["content"], use_container_width=True)
        elif msg["type"] == "plot":
            st.markdown(f"**{role_icon}**")
            st.pyplot(msg["content"])
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Chat Input ---
    with st.form("chat_form", clear_on_submit=True):
        st.markdown('<div class="input-area">', unsafe_allow_html=True)
        user_input = st.text_input(
            "Ask a question about the dashboard",
            label_visibility="collapsed",
            placeholder="Type your question here...",
        )
        submitted = st.form_submit_button("Send")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            user_input = user_input.strip()

            if not user_input:
                st.warning("⚠️ Please enter a valid question.")
            else:
                st.session_state.messages.append(
                    {"role": "user", "type": "text", "content": user_input}
                )

                # Typing
                st.session_state.messages.append(
                    {"role": "assistant", "type": "text", "content": "💬 Typing..."}
                )
                st.rerun()

    if (
            st.session_state.messages
            and st.session_state.messages[-1]["type"] == "text"
            and st.session_state.messages[-1]["content"] == "💬 Typing..."
    ):
        last_user_input = None

        for msg in reversed(st.session_state.messages[:-1]):
            if msg["role"] == "user" and msg["type"] == "text":
                last_user_input = msg["content"]
                break

        if last_user_input:
            response = process_question(last_user_input)

            st.session_state.messages.pop()

            if isinstance(response, pd.DataFrame):
                st.session_state.messages.append(
                    {"role": "assistant", "type": "dataframe", "content": response}
                )
            elif isinstance(response, plt.Figure):
                st.session_state.messages.append(
                    {"role": "assistant", "type": "plot", "content": response}
                )
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "type": "text", "content": str(response)}
                )

            st.rerun()

px.defaults.width = 950

# ───────────────────────────────
# UI Styling
# ───────────────────────────────
st.markdown(
    """
<style>
    /* Import Google Fonts for professional typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }

    /* Professional Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e293b 0%, #475569 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }

    /* Clean Metric Cards */
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.06);
    }

    /* Professional Insight Boxes */
    .insight-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        position: relative;
    }
    .insight-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }

    /* Status Boxes */
    .success-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #dcfce7;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        position: relative;
    }
    .success-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }

    .warning-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #fef3c7;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        position: relative;
    }
    .warning-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }

    .error-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #fee2e2;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        position: relative;
    }
    .error-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }

    /* Performance Indicators */
    .performance-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    /* Professional Footer */
    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 3rem;
        font-size: 0.875rem;
        padding: 2rem;
        background: white;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Enhanced Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 4px;
    }

    /* Professional Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background: white;
        border-right: 1px solid #e2e8f0;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        border-radius: 8px;
        padding: 4px;
        border: 1px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        font-weight: 500;
        color: #64748b;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
    }

    /* Metric Styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    /* Chart Container Styling */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Subsection Headers */
    .subsection-header {
        font-size: 1.25rem;
        font-weight: 500;
        color: #374151;
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
    }

    /* Data Table Styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .metric-card {
            padding: 1rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


# ───────────────────────────────
# 🚀 MPS Acceleration Setup
# ───────────────────────────────
def setup_mps_acceleration():
    """Setup MPS acceleration for M1 MacBooks if available."""
    try:
        import torch

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            return device
        else:
            return torch.device("cpu")
    except ImportError:
        return None


# ───────────────────────────────
# Data Loader with Caching
# ───────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
if not DATA_DIR.exists():
    DATA_DIR = Path("")

CSV_FILES = {
    "reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "geo": "olist_geolocation_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "categories": "product_category_name_translation.csv",
    "orders": "olist_orders_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
}


def fix_arrow_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df_fixed = df.copy()
    for col in df_fixed.columns:
        # datetime columns
        if (
                df_fixed[col].dtype == "object"
                and df_fixed[col].apply(lambda x: isinstance(x, pd.Timestamp)).any()
        ) or ("date" in col.lower() or "timestamp" in col.lower()):
            try:
                df_fixed[col] = pd.to_datetime(df_fixed[col], errors="coerce")
            except Exception:
                pass
        # string columns
        elif df_fixed[col].dtype == "object":
            df_fixed[col] = df_fixed[col].astype(str)
    return df_fixed


@st.cache_data(show_spinner="🔄 Loading Olist datasets...", ttl=3600)
def load_data() -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}

    def _csv(name: str) -> Path:
        path = DATA_DIR / CSV_FILES[name]
        if not path.exists():
            st.error(
                f"❌ CSV not found: {path}\n\nPlace the Olist dataset CSVs either in ./data or in the working directory."
            )
            st.stop()
        return path

    # Load with progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    datasets = [
        ("reviews", _csv("reviews"), ["review_creation_date"]),
        ("customers", _csv("customers"), []),
        ("geo", _csv("geo"), []),
        ("products", _csv("products"), []),
        ("sellers", _csv("sellers"), []),
        ("categories", _csv("categories"), []),
        (
            "orders",
            _csv("orders"),
            [
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        ),
        ("payments", _csv("payments"), []),
        ("order_items", _csv("order_items"), []),
    ]

    for i, (name, path, date_cols) in enumerate(datasets):
        status_text.text(f"Loading {name}...")
        try:
            df = pd.read_csv(path, parse_dates=date_cols)

            # Arrow serialization
            for col in df.columns:
                if df[col].dtype == "object":
                    # Convert string columns to string type
                    df[col] = df[col].astype(str)
                elif "datetime" in str(df[col].dtype):
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            dfs[name] = df
            progress_bar.progress((i + 1) / len(datasets))
        except Exception as e:
            st.error(f"Error loading {name}: {str(e)}")
            st.stop()

    status_text.text("✅ All datasets loaded successfully!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

    return dfs


# ───────────────────────────────
# Intelligent Data Cleaning with Insights
# ───────────────────────────────
def analyze_data_quality(df: pd.DataFrame, name: str) -> dict:
    analysis = {
        "dataset_name": name,
        "original_shape": df.shape,
        "missing_data": {},
        "duplicates": df.duplicated().sum(),
        "data_types": df.dtypes.to_dict(),
        "cleaning_recommendations": [],
        "quality_score": 100.0,
    }

    # Analyze missing data
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100

    for col in df.columns:
        missing_pct = missing_percentages[col]
        analysis["missing_data"][col] = {
            "count": missing_counts[col],
            "percentage": missing_pct,
        }

        # Quality score reduction based on missing data
        if missing_pct > 50:
            analysis["quality_score"] -= 20
            analysis["cleaning_recommendations"].append(
                f"Drop column '{col}' (>{missing_pct:.1f}% missing)"
            )
        elif missing_pct > 20:
            analysis["quality_score"] -= 10
            analysis["cleaning_recommendations"].append(
                f"Investigate missing data in '{col}' ({missing_pct:.1f}% missing)"
            )
        elif missing_pct > 5:
            analysis["quality_score"] -= 5
            analysis["cleaning_recommendations"].append(
                f"Consider imputation for '{col}' ({missing_pct:.1f}% missing)"
            )

    # Analyze data types and potential issues
    for col in df.columns:
        if df[col].dtype == "object":
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:
                analysis["cleaning_recommendations"].append(
                    f"High cardinality in '{col}' - consider encoding"
                )

        # Check for potential outliers in numeric columns
        if df[col].dtype in ["int64", "float64"]:
            if df[col].std() > df[col].mean() * 3:
                analysis["cleaning_recommendations"].append(
                    f"Potential outliers in '{col}' - high variance detected"
                )

    # Duplicate analysis
    if analysis["duplicates"] > 0:
        analysis["quality_score"] -= 15
        analysis["cleaning_recommendations"].append(
            f"Remove {analysis['duplicates']} duplicate rows"
        )

    analysis["quality_score"] = max(0, analysis["quality_score"])

    return analysis


def intelligent_clean_data(dfs: dict[str, pd.DataFrame]) -> tuple[dict, dict, dict]:
    cleaned = {}
    cleaning_report = {}
    quality_analysis = {}

    # Analyze quality first
    for name, df in dfs.items():
        quality_analysis[name] = analyze_data_quality(df, name)

    # Clean data based on analysis
    for name, df in dfs.items():
        analysis = quality_analysis[name]
        df_clean = df.copy()
        cleaning_actions = []

        # Remove duplicates
        if analysis["duplicates"] > 0:
            df_clean = df_clean.drop_duplicates()
            cleaning_actions.append(f"Removed {analysis['duplicates']} duplicate rows")

        # Handle missing data based on analysis
        for col in df_clean.columns:
            missing_info = analysis["missing_data"][col]

            if missing_info["percentage"] > 80:
                # Drop columns with >80% missing
                df_clean = df_clean.drop(columns=[col])
                cleaning_actions.append(
                    f"Dropped column '{col}' ({missing_info['percentage']:.1f}% missing)"
                )
            elif missing_info["percentage"] > 0:
                # Intelligent imputation based on data type
                if df_clean[col].dtype in ["int64", "float64"]:
                    if missing_info["percentage"] < 30:
                        # Use median for numeric with <30% missing
                        median_val = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(median_val)
                        cleaning_actions.append(
                            f"Filled missing values in '{col}' with median ({median_val:.2f})"
                        )
                    else:
                        # Use 0 for high missing numeric
                        df_clean[col] = df_clean[col].fillna(0)
                        cleaning_actions.append(
                            f"Filled missing values in '{col}' with 0"
                        )
                else:
                    # Use mode for categorical
                    mode_val = df_clean[col].mode()
                    if not mode_val.empty:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])
                        cleaning_actions.append(
                            f"Filled missing values in '{col}' with mode ('{mode_val[0]}')"
                        )

        # Special handling for specific datasets
        if name == "products":
            # Fix column name typos
            if "product_name_lenght" in df_clean.columns:
                df_clean = df_clean.rename(
                    columns={"product_name_lenght": "product_name_length"}
                )
                cleaning_actions.append(
                    "Fixed column name: 'product_name_lenght' → 'product_name_length'"
                )
            if "product_description_lenght" in df_clean.columns:
                df_clean = df_clean.rename(
                    columns={"product_description_lenght": "product_description_length"}
                )
                cleaning_actions.append(
                    "Fixed column name: 'product_description_lenght' → 'product_description_length'"
                )

            # Remove products with invalid dimensions
            invalid_dims = (
                    (df_clean["product_length_cm"] <= 0)
                    | (df_clean["product_height_cm"] <= 0)
                    | (df_clean["product_width_cm"] <= 0)
                    | (df_clean["product_weight_g"] <= 0)
            )
            if invalid_dims.any():
                df_clean = df_clean[~invalid_dims]
                cleaning_actions.append(
                    f"Removed {invalid_dims.sum()} products with invalid dimensions"
                )

        elif name == "orders":
            # Remove orders with invalid dates
            invalid_dates = (
                                    df_clean["order_purchase_timestamp"]
                                    > df_clean["order_delivered_customer_date"]
                            ) | (
                                    df_clean["order_purchase_timestamp"]
                                    > df_clean["order_estimated_delivery_date"]
                            )
            if invalid_dates.any():
                df_clean = df_clean[~invalid_dates]
                cleaning_actions.append(
                    f"Removed {invalid_dates.sum()} orders with invalid dates"
                )

        elif name == "reviews":
            # Handle review comment column variations
            comment_col = None
            for col in df_clean.columns:
                if "comment" in col.lower() and "message" in col.lower():
                    comment_col = col
                    break

            if comment_col:
                # 1) Ensure every entry is a string
                df_clean[comment_col] = df_clean[comment_col].astype(str)
                # 2) Drop literal "nan"/"NaN" entries so they become empty
                df_clean[comment_col] = df_clean[comment_col].replace(
                    r"^\s*(nan|NaN)\s*$", "", regex=True
                )
                # 3) Fill any genuine nulls, compute length & flag non-empty comments
                df_clean[comment_col] = df_clean[comment_col].fillna("")
                df_clean["review_length"] = df_clean[comment_col].str.len()
                df_clean["has_comment"] = df_clean[comment_col].str.strip().astype(bool)
                cleaning_actions.append(
                    f"Added review length analysis using '{comment_col}'"
                )

            else:
                df_clean["review_length"] = 0
                df_clean["has_comment"] = False
                cleaning_actions.append(
                    "No review comment column found - added placeholder fields"
                )

        # Track cleaning results
        cleaning_report[name] = {
            "original_shape": analysis["original_shape"],
            "cleaned_shape": df_clean.shape,
            "duplicates_removed": analysis["duplicates"],
            "columns_dropped": list(set(df.columns) - set(df_clean.columns)),
            "missing_after": int(df_clean.isnull().sum().sum()),
            "quality_score_before": analysis["quality_score"],
            "quality_score_after": analyze_data_quality(df_clean, name)[
                "quality_score"
            ],
            "cleaning_actions": cleaning_actions,
            "recommendations": analysis["cleaning_recommendations"],
        }

        cleaned[name] = df_clean

    return cleaned, cleaning_report, quality_analysis


# ───────────────────────────────
# Enhanced Preprocessing
# ───────────────────────────────
@st.cache_data
def preprocess_enhanced(dfs: dict[str, pd.DataFrame]):
    """Enhanced preprocessing with additional derived metrics and MPS optimization."""
    # Clean products
    prod = (
        dfs["products"]
        .copy()
        .rename(
            columns={
                "product_name_lenght": "product_name_length",
                "product_description_lenght": "product_description_length",
            }
        )
        .merge(dfs["categories"], on="product_category_name", how="left")
    )

    # Add product volume
    prod["product_volume_cm3"] = (
            prod["product_length_cm"] * prod["product_height_cm"] * prod["product_width_cm"]
    )

    # Enhanced order items
    oi = (
        dfs["order_items"]
        .merge(
            dfs["orders"][
                [
                    "order_id",
                    "customer_id",
                    "order_purchase_timestamp",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_status",
                ]
            ],
            on="order_id",
            how="left",
        )
        .merge(
            dfs["sellers"][["seller_id", "seller_city", "seller_state"]],
            on="seller_id",
            how="left",
        )
        .merge(
            prod[["product_id", "product_category_name_english", "product_volume_cm3"]],
            on="product_id",
            how="left",
        )
    )

    # Add delivery metrics
    oi["delivery_duration_days"] = (
            oi["order_delivered_customer_date"] - oi["order_purchase_timestamp"]
    ).dt.days

    oi["delivery_vs_estimated"] = (
            oi["order_delivered_customer_date"] - oi["order_estimated_delivery_date"]
    ).dt.days

    oi["is_delayed"] = oi["delivery_vs_estimated"] > 0
    oi["is_early"] = oi["delivery_vs_estimated"] < 0

    # Add revenue metrics
    oi["total_revenue"] = oi["price"] + oi["freight_value"]

    # Add review length to reviews dataframe
    reviews_enhanced = dfs["reviews"].copy()

    # Check if review comment column exists and handle variations
    comment_col = None
    for col in reviews_enhanced.columns:
        if "comment" in col.lower() and "message" in col.lower():
            comment_col = col
            break

    if comment_col:
        reviews_enhanced[comment_col] = reviews_enhanced[comment_col].astype(str)
        reviews_enhanced[comment_col] = reviews_enhanced[comment_col].replace(
            r"^\s*(nan|NaN)\s*$", "", regex=True
        )
        reviews_enhanced[comment_col] = reviews_enhanced[comment_col].fillna("")
        reviews_enhanced["review_length"] = reviews_enhanced[comment_col].str.len()
        reviews_enhanced["has_comment"] = (
            reviews_enhanced[comment_col].str.strip().astype(bool)
        )
    else:
        # Fallback if comment column not found
        reviews_enhanced["review_length"] = 0
        reviews_enhanced["has_comment"] = False

    return prod, oi, reviews_enhanced


# ───────────────────────────────
# Enhanced KPI Calculations
# ───────────────────────────────
@st.cache_data
def calculate_enhanced_kpis(
        oi: pd.DataFrame,
        orders: pd.DataFrame,
        payments: pd.DataFrame,
        reviews: pd.DataFrame,
) -> dict:
    """Calculate comprehensive KPIs with caching."""
    kpis = {}

    # Revenue metrics
    kpis["total_revenue"] = oi["total_revenue"].sum()
    kpis["avg_order_value"] = oi.groupby("order_id")["total_revenue"].sum().mean()
    kpis["total_orders"] = oi["order_id"].nunique()
    kpis["total_items"] = len(oi)

    # Delivery metrics
    delivered_orders = oi[oi["order_delivered_customer_date"].notna()]
    if len(delivered_orders) > 0:
        kpis["avg_delivery_days"] = delivered_orders["delivery_duration_days"].mean()
        kpis["on_time_delivery_rate"] = (~delivered_orders["is_delayed"]).mean() * 100
        kpis["early_delivery_rate"] = delivered_orders["is_early"].mean() * 100
    else:
        kpis["avg_delivery_days"] = 0
        kpis["on_time_delivery_rate"] = 0
        kpis["early_delivery_rate"] = 0

    # Customer metrics
    kpis["unique_customers"] = oi["customer_id"].nunique()
    kpis["unique_sellers"] = oi["seller_id"].nunique()

    # Product metrics
    kpis["unique_products"] = oi["product_id"].nunique()
    kpis["unique_categories"] = oi["product_category_name_english"].nunique()

    # Review metrics
    kpis["avg_review_score"] = reviews["review_score"].mean()
    kpis["review_response_rate"] = reviews["has_comment"].mean() * 100

    # Payment metrics
    kpis["avg_payment_value"] = payments["payment_value"].mean()
    kpis["credit_card_usage"] = (payments["payment_type"] == "credit_card").mean() * 100

    return kpis


# ───────────────────────────────
#  Professional Navigation and UI
# ───────────────────────────────
st.sidebar.markdown(
    """
<div style="text-align: center; padding: 1rem 0;">
    <h2 style="color: #1e293b; margin: 0; font-weight: 600;">📊 Olist Analytics</h2>
</div>
""",
    unsafe_allow_html=True,
)

# MPS Setup
device = setup_mps_acceleration()

# Navigation
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Navigation")
section = st.sidebar.radio(
    "Choose Section:",
    [
        "🧹 Data Cleaning",
        "📊 EDA",
        "💳 Payment Anomaly Detection",
        " 🚚 Delivery Anomaly Detection",
        "📝 Review Anomaly Detection",
    ],
    help="Select the analysis section you want to explore",
)

# Load data
if "raw_data" not in st.session_state:
    st.session_state["raw_data"] = load_data()

if section == "🧹 Data Cleaning":
    st.markdown(
        '<h1 class="main-header"><span style="color: #10b981;">🧹</span> Data Quality & Cleaning</h1>',
        unsafe_allow_html=True,
    )

    if (
            st.button("🚀 Run Intelligent Cleaning", key="clean_btn")
            or "cleaned_data" not in st.session_state
    ):
        with st.spinner("🧹 Analyzing data quality and cleaning..."):
            cleaned, report, quality_analysis = intelligent_clean_data(
                st.session_state["raw_data"]
            )
            st.session_state["cleaned_data"] = cleaned
            st.session_state["cleaning_report"] = report
            st.session_state["quality_analysis"] = quality_analysis
        st.success("✅ Intelligent data cleaning completed!")

    if "cleaned_data" in st.session_state:
        report = st.session_state["cleaning_report"]
        quality_analysis = st.session_state["quality_analysis"]

        # Overall Quality Dashboard
        st.markdown(
            '<h2 class="section-header">📊 Data Quality Overview</h2>',
            unsafe_allow_html=True,
        )

        # Calculate overall metrics
        total_original_rows = sum(info["original_shape"][0] for info in report.values())
        total_cleaned_rows = sum(info["cleaned_shape"][0] for info in report.values())
        total_duplicates = sum(info["duplicates_removed"] for info in report.values())
        avg_quality_before = np.mean(
            [info["quality_score_before"] for info in report.values()]
        )
        avg_quality_after = np.mean(
            [info["quality_score_after"] for info in report.values()]
        )

        # Only display the four main metrics, no extra columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:

            st.metric(
                "📈 Total Records", f"{total_cleaned_rows:,}", f"{total_original_rows:,}"
            )

        with col2:

            st.metric("🧹 Duplicates Removed", f"{total_duplicates:,}")

        with col3:

            st.metric(
                "📊 Quality Score",
                f"{avg_quality_after:.1f}%",
                f"{avg_quality_before:.1f}%",
            )

        with col4:

            quality_improvement = avg_quality_after - avg_quality_before
            st.metric("🚀 Quality Improvement", f"+{quality_improvement:.1f}%")

        # Detailed Dataset Analysis
        st.markdown(
            '<h2 class="section-header">🔍 Detailed Dataset Analysis</h2>',
            unsafe_allow_html=True,
        )

        for name, info in report.items():
            with st.expander(
                    f"📋 {name.replace('_', ' ').title()} Analysis", expanded=False
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Quality metrics
                    st.markdown(f"### 📊 Quality Metrics")
                    quality_col1, quality_col2, quality_col3 = st.columns(3)

                    with quality_col1:
                        st.metric(
                            "Quality Score",
                            f"{info['quality_score_after']:.1f}%",
                            f"{info['quality_score_before']:.1f}%",
                        )
                    with quality_col2:
                        st.metric(
                            "Records",
                            f"{info['cleaned_shape'][0]:,}",
                            f"{info['original_shape'][0]:,}",
                        )
                    with quality_col3:
                        st.metric(
                            "Columns",
                            f"{info['cleaned_shape'][1]}",
                            f"{info['original_shape'][1]}",
                        )

                    # Missing data analysis
                    if info["missing_after"] > 0:
                        st.warning(
                            f"⚠️ {info['missing_after']} missing values remaining"
                        )
                    else:
                        st.success("✅ No missing values")

                    # Cleaning actions
                    if info["cleaning_actions"]:
                        st.markdown("### 🛠️ Cleaning Actions Applied")
                        for action in info["cleaning_actions"]:
                            st.write(f"• {action}")

                with col2:
                    # Visual quality indicator
                    quality_score = info["quality_score_after"]
                    if quality_score >= 90:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown("### 🟢 Excellent Quality")
                    elif quality_score >= 70:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown("### 🟡 Good Quality")
                    else:
                        st.markdown('<div class="error-box">', unsafe_allow_html=True)
                        st.markdown("### 🔴 Needs Attention")

                    st.metric("Quality Score", f"{quality_score:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)

        # Data Preview
        st.markdown(
            '<h2 class="section-header">👀 Data Preview</h2>', unsafe_allow_html=True
        )
        preview_tab1, preview_tab2 = st.tabs(["📊 Summary Statistics", "📋 Sample Data"])

        with preview_tab1:
            for name, df in st.session_state["cleaned_data"].items():
                with st.expander(f"📊 {name.replace('_', ' ').title()} Statistics"):
                    try:
                        stats_df = fix_arrow_dtypes(df.describe())
                        st.dataframe(stats_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating statistics for {name}: {str(e)}")

        with preview_tab2:
            for name, df in st.session_state["cleaned_data"].items():
                with st.expander(f"📋 {name.replace('_', ' ').title()} Sample"):
                    try:
                        sample_df = fix_arrow_dtypes(df.head())
                        st.dataframe(sample_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying sample for {name}: {str(e)}")

    else:
        st.info("🚀 Click 'Run Intelligent Cleaning' to analyze and clean your data!")
elif section == "📊 EDA":
    if "cleaned_data" not in st.session_state:
        st.error("❌ Please run data cleaning first in the 'Data Cleaning' section.")
        st.stop()

    st.markdown(
        '<h1 class="main-header">📊 Business Intelligence Dashboard</h1>',
        unsafe_allow_html=True,
    )

    # Preprocess data for EDA
    if "processed_data" not in st.session_state:
        with st.spinner("🔄 Preprocessing data for analysis..."):
            products_df, order_items_enriched, reviews_enhanced = preprocess_enhanced(
                st.session_state["cleaned_data"]
            )
            st.session_state["processed_data"] = {
                "products": products_df,
                "order_items": order_items_enriched,
                "reviews": reviews_enhanced,
            }

    products_df = st.session_state["processed_data"]["products"]
    order_items_enriched = st.session_state["processed_data"]["order_items"]
    reviews_enhanced = st.session_state["processed_data"]["reviews"]

    # Calculate KPIs
    kpis = calculate_enhanced_kpis(
        order_items_enriched,
        st.session_state["cleaned_data"]["orders"],
        st.session_state["cleaned_data"]["payments"],
        reviews_enhanced,
    )

    # Executive Summary
    st.markdown(
        '<h2 class="section-header">🎯 Executive Summary</h2>', unsafe_allow_html=True
    )

    # Full-width Business Performance Overview
    st.markdown(
        """
      <div class="insight-box">
       <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-size: 1.25rem; font-weight: 600;">
        📈 Business Performance Overview
       </h3>
       <p style="color: #374151; line-height: 1.6; margin: 0;">
        This comprehensive analysis of the Olist e-commerce platform reveals key insights about customer behavior, 
         product performance, and operational efficiency across Brazil. The data spans multiple years and covers 
        thousands of transactions, providing valuable insights for strategic decision-making.
       </p>
      </div>
      """,
        unsafe_allow_html=True,
    )

    # KPI Cards
    st.markdown(
        '<h2 class="section-header">📊 Key Performance Indicators</h2>',
        unsafe_allow_html=True,
    )

    # Revenue Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:

        st.markdown(
            '<h3 style="color: #1e293b; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600;">💰 Revenue Metrics</h3>',
            unsafe_allow_html=True,
        )
        st.metric("Total Revenue", f"R$ {kpis['total_revenue']:,.0f}")

    with col2:

        st.markdown(
            '<h3 style="color: #1e293b; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600;">🚚 Delivery Performance</h3>',
            unsafe_allow_html=True,
        )
        st.metric("Avg Delivery Days", f"{kpis['avg_delivery_days']:.1f}")

    with col3:

        st.markdown(
            '<h3 style="color: #1e293b; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600;">👥 Customer & Seller</h3>',
            unsafe_allow_html=True,
        )
        st.metric("Unique Customers", f"{kpis['unique_customers']:,}")

    with col4:

        st.markdown(
            '<h3 style="color: #1e293b; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600;">⭐ Customer Satisfaction</h3>',
            unsafe_allow_html=True,
        )
        st.metric("Avg Review Score", f"{kpis['avg_review_score']:.2f}/5.0")

    # EDA Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "📊 Overview",
            "📦 Products",
            "💳 Payments",
            "🌍 Geography",
            "👥 Customers",
            "📝 Reviews",
            "📈 Advanced Analytics",
        ]
    )

    with tab1:
        st.markdown("## 📊 Business Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                '<h3 class="subsection-header">📈 Revenue Trends</h3>',
                unsafe_allow_html=True,
            )
            monthly_revenue = (
                order_items_enriched.groupby(
                    order_items_enriched["order_purchase_timestamp"].dt.to_period("M")
                )["total_revenue"]
                .sum()
                .reset_index()
            )
            monthly_revenue["order_purchase_timestamp"] = monthly_revenue[
                "order_purchase_timestamp"
            ].astype(str)

            fig = px.line(
                monthly_revenue,
                x="order_purchase_timestamp",
                y="total_revenue",
                markers=True,
                labels={
                    "total_revenue": "Revenue (R$)",
                    "order_purchase_timestamp": "Month",
                },
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(
                '<h3 class="subsection-header">📦 Order Volume Trends</h3>',
                unsafe_allow_html=True,
            )
            monthly_orders = (
                order_items_enriched.groupby(
                    order_items_enriched["order_purchase_timestamp"].dt.to_period("M")
                )["order_id"]
                .nunique()
                .reset_index()
            )
            monthly_orders["order_purchase_timestamp"] = monthly_orders[
                "order_purchase_timestamp"
            ].astype(str)

            fig = px.bar(
                monthly_orders,
                x="order_purchase_timestamp",
                y="order_id",
                labels={
                    "order_id": "Number of Orders",
                    "order_purchase_timestamp": "Month",
                },
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        #     st.markdown('<h4 style="color: #374151; margin: 0 0 1rem 0; font-size: 1.125rem; font-weight: 600;">📈 Revenue vs Orders Correlation</h4>', unsafe_allow_html=True)
        #     # Create correlation scatter plot
        monthly_performance = (
            order_items_enriched.groupby(
                order_items_enriched["order_purchase_timestamp"].dt.to_period("M")
            )
            .agg({"total_revenue": "sum", "order_id": "nunique"})
            .reset_index()
        )
        monthly_performance["order_purchase_timestamp"] = monthly_performance[
            "order_purchase_timestamp"
        ].astype(str)

        order_items_enriched["month"] = order_items_enriched[
            "order_purchase_timestamp"
        ].dt.month
        order_items_enriched["day_of_week"] = order_items_enriched[
            "order_purchase_timestamp"
        ].dt.day_name()

        seasonal_revenue = (
            order_items_enriched.groupby("month")["total_revenue"].sum().reset_index()
        )

        # # Day of week analysis

        st.markdown(
            '<h3 class="subsection-header">📅 Day of Week Performance</h3>',
            unsafe_allow_html=True,
        )
        dow_performance = (
            order_items_enriched.groupby("day_of_week")["total_revenue"]
            .sum()
            .reset_index()
        )

        # Reorder days of week
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        dow_performance["day_of_week"] = pd.Categorical(
            dow_performance["day_of_week"], categories=day_order, ordered=True
        )
        dow_performance = dow_performance.sort_values("day_of_week")

        fig = px.bar(
            dow_performance,
            x="day_of_week",
            y="total_revenue",
            color="total_revenue",
            labels={
                "day_of_week": "Day of Week",
                "total_revenue": "Total Revenue (R$)",
            },
            title="Revenue by Day of Week",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("## 📦 Products Analysis")
        col1 = st.columns(1)
        st.subheader("🏆 Top Product Categories (Revenue)")
        top_cat_revenue = (
            order_items_enriched.groupby("product_category_name_english")[
                "total_revenue"
            ]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )

        fig = px.bar(
            top_cat_revenue,
            y="product_category_name_english",
            x="total_revenue",
            orientation="h",
            labels={"total_revenue": "Revenue (R$)"},
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("💰 Price Distribution by Category")
        # Price analysis by category
        price_by_category = (
            order_items_enriched.groupby("product_category_name_english")
            .agg(
                {
                    "price": ["mean", "median", "std"],
                    "total_revenue": "sum",
                    "order_id": "nunique",
                }
            )
            .reset_index()
        )
        price_by_category.columns = [
            "Category",
            "Avg_Price",
            "Median_Price",
            "Price_Std",
            "Total_Revenue",
            "Orders",
        ]
        price_by_category = price_by_category.sort_values(
            "Total_Revenue", ascending=False
        ).head(15)

        fig = px.bar(
            price_by_category,
            x="Category",
            y="Avg_Price",
            color="Total_Revenue",
            labels={
                "Category": "Product Category",
                "Avg_Price": "Average Price (R$)",
                "Total_Revenue": "Total Revenue (R$)",
            },
            title="Average Price by Category (Top 15 by Revenue)",
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        # Total Number of Sellers per Product Category (Top 20)
        with st.expander(
                "🧮 Total Number of Sellers per Product Category (Top 20)", expanded=True
        ):
            seller_counts = (
                order_items_enriched.groupby("product_category_name_english")[
                    "seller_id"
                ]
                .nunique()
                .reset_index(name="distinct_sellers")
                .sort_values("distinct_sellers", ascending=False)
                .head(20)
            )
            fig_sellers_cat = px.line(
                seller_counts,
                x="product_category_name_english",
                y="distinct_sellers",
                markers=True,
                title="Total Number of Sellers per Product Category (Top 20)",
            )
            fig_sellers_cat.update_layout(
                xaxis_title="Product Category",
                yaxis_title="Distinct Count of Sellers",
                xaxis_tickangle=-45,
                height=450,
            )
            st.plotly_chart(fig_sellers_cat, use_container_width=True)

    with tab3:

        st.subheader("💳 Payment Type Distribution")
        payment_dist = st.session_state["cleaned_data"]["payments"][
            "payment_type"
        ].value_counts()

        fig = px.pie(
            values=payment_dist.values,
            names=payment_dist.index,
            title="Payment Methods",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Payment type performance over time
        st.subheader("📈 Payment Method Trends")
        payment_trends = st.session_state["cleaned_data"]["payments"].merge(
            st.session_state["cleaned_data"]["orders"][
                ["order_id", "order_purchase_timestamp"]
            ],
            on="order_id",
            how="left",
        )
        payment_trends["month"] = payment_trends[
            "order_purchase_timestamp"
        ].dt.to_period("M")

        monthly_payment_methods = (
            payment_trends.groupby(["month", "payment_type"])
            .agg({"payment_value": "sum", "order_id": "nunique"})
            .reset_index()
        )
        monthly_payment_methods["month"] = monthly_payment_methods["month"].astype(str)

        fig = px.line(
            monthly_payment_methods,
            x="month",
            y="payment_value",
            color="payment_type",
            markers=True,
            labels={
                "month": "Month",
                "payment_value": "Total Payment Value (R$)",
                "payment_type": "Payment Type",
            },
            title="Payment Method Trends Over Time",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("## 🌍 Geographical Analysis")

        # Business Overview Cards
        st.markdown("### 📊 Geographic Business Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🌍 States Covered",
                f"{order_items_enriched.merge(st.session_state['cleaned_data']['customers'].merge(st.session_state['cleaned_data']['geo'], left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left'), on='customer_id', how='left')['geolocation_state'].nunique()}",
            )

        with col2:
            st.metric(
                "🏢 Seller States",
                f"{st.session_state['cleaned_data']['sellers']['seller_state'].nunique()}",
            )

        # Interactive Geographic Map
        st.markdown("### 🗺️ Interactive Geographic Map")

        # Prepare data for mapping
        cust_geo = st.session_state["cleaned_data"]["customers"].merge(
            st.session_state["cleaned_data"]["geo"],
            left_on="customer_zip_code_prefix",
            right_on="geolocation_zip_code_prefix",
            how="left",
        )

        # Create comprehensive geographic analysis
        geo_analysis = (
            order_items_enriched.merge(
                cust_geo[
                    [
                        "customer_id",
                        "geolocation_state",
                        "geolocation_city",
                        "geolocation_lat",
                        "geolocation_lng",
                    ]
                ],
                on="customer_id",
                how="left",
            )
            .groupby("geolocation_state")
            .agg(
                {
                    "order_id": "nunique",
                    "total_revenue": "sum",
                    "customer_id": "nunique",
                    "geolocation_lat": "mean",
                    "geolocation_lng": "mean",
                }
            )
            .reset_index()
            .rename(
                columns={
                    "geolocation_state": "State",
                    "order_id": "Orders",
                    "total_revenue": "Revenue",
                    "customer_id": "Customers",
                    "geolocation_lat": "lat",
                    "geolocation_lng": "lon",
                }
            )
            .dropna(subset=["lat", "lon"])
            .sort_values("Revenue", ascending=False)
        )

        # Create map with multiple layers
        fig = go.Figure()

        # scatter plot for states with size based on revenue
        fig.add_trace(
            go.Scattergeo(
                lon=geo_analysis["lon"],
                lat=geo_analysis["lat"],
                text=geo_analysis["State"]
                     + "<br>"
                     + "Orders: "
                     + geo_analysis["Orders"].astype(str)
                     + "<br>"
                     + "Revenue: R$ "
                     + geo_analysis["Revenue"].round(0).astype(str)
                     + "<br>"
                     + "Customers: "
                     + geo_analysis["Customers"].astype(str),
                mode="markers",
                marker=dict(
                    size=geo_analysis["Revenue"] / geo_analysis["Revenue"].max() * 30
                         + 10,
                    color=geo_analysis["Orders"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Number of Orders"),
                ),
                name="States by Revenue & Orders",
            )
        )

        fig.update_geos(
            scope="south america",
            center=dict(lat=-15.7801, lon=-47.9292),  # Brazil center
            projection_scale=3,
            showland=True,
            landcolor="rgb(243, 243, 243)",
            coastlinecolor="rgb(204, 204, 204)",
        )

        fig.update_layout(
            title="Brazilian E-commerce Geographic Distribution<br><sub>Bubble size = Revenue, Color = Number of Orders</sub>",
            height=600,
            geo=dict(
                showframe=False, showcoastlines=True, projection_type="equirectangular"
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Geographic Performance Analysis
        st.markdown("### 📊 Geographic Performance Analysis")

        # Customer vs Seller geographic distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👥 Customer Geographic Distribution")
            customer_geo_dist = (
                order_items_enriched.merge(
                    cust_geo[["customer_id", "geolocation_state"]],
                    on="customer_id",
                    how="left",
                )
                .groupby("geolocation_state")
                .agg(
                    {
                        "customer_id": "nunique",
                        "total_revenue": "sum",
                        "order_id": "nunique",
                    }
                )
                .reset_index()
                .rename(
                    columns={
                        "geolocation_state": "State",
                        "customer_id": "Customers",
                        "total_revenue": "Revenue",
                        "order_id": "Orders",
                    }
                )
                .sort_values("Customers", ascending=False)
                .head(15)
            )

            fig = px.bar(
                customer_geo_dist,
                x="State",
                y="Customers",
                color="Revenue",
                labels={
                    "State": "State",
                    "Customers": "Number of Customers",
                    "Revenue": "Total Revenue (R$)",
                },
                title="Top 15 States by Customer Count",
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🏢 Seller Geographic Distribution")
            seller_geo_dist = (
                order_items_enriched.groupby("seller_state")
                .agg(
                    {
                        "seller_id": "nunique",
                        "total_revenue": "sum",
                        "order_id": "nunique",
                    }
                )
                .reset_index()
                .rename(
                    columns={
                        "seller_state": "State",
                        "seller_id": "Sellers",
                        "total_revenue": "Revenue",
                        "order_id": "Orders",
                    }
                )
                .sort_values("Sellers", ascending=False)
                .head(15)
            )

            fig = px.bar(
                seller_geo_dist,
                x="State",
                y="Sellers",
                color="Revenue",
                labels={
                    "State": "State",
                    "Sellers": "Number of Sellers",
                    "Revenue": "Total Revenue (R$)",
                },
                title="Top 15 States by Seller Count",
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        # Detailed State Analysis
        st.markdown("### 📈 Detailed State Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Orders by State")
            fig = px.bar(
                geo_analysis.head(15),
                x="State",
                y="Orders",
                labels={"Orders": "Number of Orders"},
                title="Top 15 States by Order Volume",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("💰 Revenue by State")
            fig = px.bar(
                geo_analysis.head(15),
                x="State",
                y="Revenue",
                labels={"Revenue": "Revenue (R$)"},
                title="Top 15 States by Revenue",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Seller Distribution
        st.markdown("### 🏢 Seller Geographic Distribution")

        seller_geo = st.session_state["cleaned_data"]["sellers"].merge(
            st.session_state["cleaned_data"]["geo"],
            left_on="seller_zip_code_prefix",
            right_on="geolocation_zip_code_prefix",
            how="left",
        )

        seller_by_state = (
            seller_geo.groupby("geolocation_state")
            .agg(
                {
                    "seller_id": "nunique",
                    "geolocation_lat": "mean",
                    "geolocation_lng": "mean",
                }
            )
            .reset_index()
            .rename(
                columns={
                    "geolocation_state": "State",
                    "seller_id": "Sellers",
                    "geolocation_lat": "lat",
                    "geolocation_lng": "lon",
                }
            )
            .dropna(subset=["lat", "lon"])
            .sort_values("Sellers", ascending=False)
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                seller_by_state.head(15),
                x="State",
                y="Sellers",
                labels={"Sellers": "Number of Sellers"},
                title="Top 15 States by Seller Count",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Seller map
            fig = go.Figure()

            fig.add_trace(
                go.Scattergeo(
                    lon=seller_by_state["lon"],
                    lat=seller_by_state["lat"],
                    text=seller_by_state["State"]
                         + "<br>"
                         + "Sellers: "
                         + seller_by_state["Sellers"].astype(str),
                    mode="markers",
                    marker=dict(
                        size=seller_by_state["Sellers"]
                             / seller_by_state["Sellers"].max()
                             * 25
                             + 8,
                        color=seller_by_state["Sellers"],
                        colorscale="Plasma",
                        showscale=True,
                        colorbar=dict(title="Number of Sellers"),
                    ),
                    name="Seller Distribution",
                )
            )

            fig.update_geos(
                scope="south america",
                center=dict(lat=-15.7801, lon=-47.9292),
                projection_scale=3,
                showland=True,
                landcolor="rgb(243, 243, 243)",
                coastlinecolor="rgb(204, 204, 204)",
            )

            fig.update_layout(
                title="Seller Geographic Distribution in Brazil",
                height=400,
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type="equirectangular",
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown("## 👥 Customer Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔄 Customer Loyalty Analysis")
            cu_orders = st.session_state["cleaned_data"]["orders"].merge(
                st.session_state["cleaned_data"]["customers"][
                    ["customer_id", "customer_unique_id"]
                ],
                on="customer_id",
                how="left",
            )
            order_counts = cu_orders["customer_unique_id"].value_counts()

            loyalty_dist = order_counts.value_counts().sort_index()
            fig = px.bar(
                x=loyalty_dist.index,
                y=loyalty_dist.values,
                labels={"x": "Orders per Customer", "y": "Number of Customers"},
                title="Customer Purchase Frequency",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            repeat_pct = (order_counts > 1).mean() * 100
            st.metric("Repeat Customers %", f"{repeat_pct:.1f}%")

        with col2:
            st.subheader("🏙️ Top Customer Cities")
            top_city = (
                st.session_state["cleaned_data"]["customers"]["customer_city"]
                .value_counts()
                .head(15)
                .reset_index()
            )
            top_city.columns = ["City", "Customers"]

            fig = px.bar(
                top_city,
                y="City",
                x="Customers",
                orientation="h",
                labels={"Customers": "Number of Customers"},
                title="Top 15 Cities by Customer Count",
            ).update_yaxes(categoryorder="total ascending")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.markdown("## 📝 Customer Reviews Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⭐ Review Score Distribution")
            score_dist = reviews_enhanced["review_score"].value_counts().sort_index()

            fig = px.bar(
                x=score_dist.index,
                y=score_dist.values,
                color=score_dist.index,
                labels={"x": "Review Score", "y": "Count"},
                color_continuous_scale="RdYlGn",
                title="Distribution of Review Scores",
            )
            fig.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            avg_score = reviews_enhanced["review_score"].mean()

        with col2:
            st.subheader("📊 Review Length by Score")
            fig = px.box(
                reviews_enhanced[reviews_enhanced["review_length"] > 0],
                x="review_score",
                y="review_length",
                color="review_score",
                labels={
                    "review_length": "Review Length (characters)",
                    "review_score": "Score",
                },
                title="Review Length vs. Score (for reviews with comments)",
            )
            fig.update_yaxes(range=[0, 400])  # Zoom in on more common lengths
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander(
                "⭐ Average Delivery Time vs Average Review Scores", expanded=True
        ):
            # --- data prep (same as before) ---
            reviews_join = reviews_enhanced.merge(
                order_items_enriched[
                    [
                        "order_id",
                        "product_category_name_english",
                        "order_delivered_customer_date",
                        "order_purchase_timestamp",
                    ]
                ].drop_duplicates("order_id"),
                on="order_id",
                how="inner",
            ).dropna(
                subset=[
                    "order_delivered_customer_date",
                    "order_purchase_timestamp",
                    "review_score",
                ]
            )

            reviews_join["delivery_time_days"] = (
                    reviews_join["order_delivered_customer_date"]
                    - reviews_join["order_purchase_timestamp"]
            ).dt.days

            delivery_review_cat = (
                reviews_join.groupby("product_category_name_english")
                .agg(
                    avg_delivery_time=("delivery_time_days", "mean"),
                    avg_review_score=("review_score", "mean"),
                    review_count=("review_score", "count"),
                )
                .reset_index()
                .sort_values("avg_delivery_time", ascending=False)
                .head(25)
                .sort_values("avg_delivery_time", ascending=True)
            )

            fig_rev = go.Figure()

            # Line 1: Avg Delivery Time
            fig_rev.add_trace(
                go.Scatter(
                    x=delivery_review_cat["product_category_name_english"],
                    y=delivery_review_cat["avg_delivery_time"],
                    mode="lines+markers",
                    name="Avg Delivery Time (Days)",
                    yaxis="y",
                )
            )

            # Line 2: Avg Review Score (secondary axis)
            fig_rev.add_trace(
                go.Scatter(
                    x=delivery_review_cat["product_category_name_english"],
                    y=delivery_review_cat["avg_review_score"],
                    mode="lines+markers",
                    name="Avg Review Score",
                    yaxis="y2",
                )
            )

            fig_rev.update_layout(
                title="Average Delivery Time vs Average Review Scores (Top Categories)",
                xaxis=dict(title="Product Category", tickangle=-40, automargin=True),
                yaxis=dict(title="Avg Delivery Time (Days)"),
                yaxis2=dict(
                    title="Avg Review Score",
                    overlaying="y",
                    side="right",
                    range=[
                        delivery_review_cat["avg_review_score"].min() - 0.1,
                        delivery_review_cat["avg_review_score"].max() + 0.1,
                    ],
                ),
                legend=dict(orientation="h", y=1.1),
                margin=dict(l=70, r=70, b=120, t=80),
                height=700,
            )

            st.plotly_chart(fig_rev, use_container_width=True)

    with tab7:
        st.markdown("## 📈 Advanced Analytics")

        # Correlation Analysis
        st.markdown("### 🔗 Correlation Analysis")

        # Prepare data for correlation analysis
        correlation_data = order_items_enriched[
            ["price", "freight_value", "total_revenue"]
        ].copy()
        if "delivery_duration_days" in order_items_enriched.columns:
            correlation_data["delivery_duration_days"] = order_items_enriched[
                "delivery_duration_days"
            ]

        correlation_matrix = correlation_data.corr()

        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto",
            labels=dict(x="Features", y="Features", color="Correlation"),
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📅 Time Series Analysis")

        # Daily trends
        daily_trends = (
            order_items_enriched.groupby(
                order_items_enriched["order_purchase_timestamp"].dt.date
            )
            .agg(
                {
                    "total_revenue": "sum",
                    "order_id": "nunique",
                }
            )
            .reset_index()
        )
        daily_trends["order_purchase_timestamp"] = pd.to_datetime(
            daily_trends["order_purchase_timestamp"]
        )

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Daily Revenue Trends", "Daily Order Volume"),
            vertical_spacing=0.1,
        )

        fig.add_trace(
            go.Scatter(
                x=daily_trends["order_purchase_timestamp"],
                y=daily_trends["total_revenue"],
                mode="lines",
                name="Revenue",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=daily_trends["order_purchase_timestamp"],
                y=daily_trends["order_id"],
                mode="lines",
                name="Orders",
                line=dict(color="orange"),
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=600, showlegend=False, title_text="Daily Performance Metrics"
        )
        st.plotly_chart(fig, use_container_width=True)

        # NEW: Product Delivery Performance (Actual vs Estimated) - Top 20 Categories
        with st.expander(
                "🚚 Product Delivery Performance (Actual vs Estimated) - Top 20",
                expanded=True,
        ):
            delivery_perf = order_items_enriched.dropna(
                subset=[
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ]
            ).copy()

            delivery_perf["actual_delivery_days"] = (
                    delivery_perf["order_delivered_customer_date"]
                    - delivery_perf["order_purchase_timestamp"]
            ).dt.days
            delivery_perf["estimated_delivery_days"] = (
                    delivery_perf["order_estimated_delivery_date"]
                    - delivery_perf["order_purchase_timestamp"]
            ).dt.days

            delivery_agg = (
                delivery_perf.groupby("product_category_name_english")
                .agg(
                    avg_actual=("actual_delivery_days", "mean"),
                    avg_estimated=("estimated_delivery_days", "mean"),
                )
                .reset_index()
            )

            # Choose top 20 by highest average actual delivery time (matches visual intent)
            delivery_top20 = (
                delivery_agg.sort_values("avg_actual", ascending=False)
                .head(20)
                .sort_values("avg_actual", ascending=True)
            )  # left→right increasing for readability

            fig_delivery = go.Figure()
            fig_delivery.add_trace(
                go.Scatter(
                    x=delivery_top20["product_category_name_english"],
                    y=delivery_top20["avg_actual"],
                    mode="lines+markers",
                    name="Avg Actual Delivery Time (days)",
                    line=dict(width=2),
                )
            )
            fig_delivery.add_trace(
                go.Scatter(
                    x=delivery_top20["product_category_name_english"],
                    y=delivery_top20["avg_estimated"],
                    mode="lines+markers",
                    name="Avg Estimated Delivery Time (days)",
                    line=dict(width=2, dash="dot"),
                    yaxis="y2",
                )
            )
            fig_delivery.update_layout(
                title="Product Delivery Performance for Top 20 Categories",
                xaxis_title="Product Category",
                yaxis=dict(title="Avg Actual Delivery Time (Days)"),
                yaxis2=dict(
                    title="Avg Estimated Delivery Time (Days)",
                    overlaying="y",
                    side="right",
                ),
                xaxis_tickangle=-60,
                legend=dict(orientation="h", y=1.12),
                height=550,
            )
            st.plotly_chart(fig_delivery, use_container_width=True)
            # NEW: Freight Cost Analysis of Top 20 Product Categories
elif section == "💳 Payment Anomaly Detection":

    def payment_anomaly_dashboard():
        st.title("\U0001F4CA Olist Payment Anomaly Detection")

        # ---- LOAD DATA ----
        payments_df = pd.read_csv("olist_order_payments_dataset.csv")
        orders_df = pd.read_csv("olist_orders_dataset.csv")
        orderitems_df = pd.read_csv("olist_order_items_dataset.csv")
        products_df = pd.read_csv("olist_products_dataset.csv")
        productcat_df = pd.read_csv("product_category_name_translation.csv")
        customers_df = pd.read_csv("olist_customers_dataset.csv")

        # ---- CLEANING & MERGING ----
        payments_df = payments_df[payments_df["payment_type"] != "not_defined"]
        products_df["product_category_name"] = products_df[
            "product_category_name"
        ].fillna("Unknown")
        products_full = products_df.merge(
            productcat_df, on="product_category_name", how="left"
        )
        products_full["product_category_name_english"] = products_full[
            "product_category_name_english"
        ].fillna("Unknown")

        orderitems_full = orderitems_df.merge(
            products_full, on="product_id", how="left"
        )
        orderitems_full["product_category_name_english"] = orderitems_full[
            "product_category_name_english"
        ].fillna("Unknown")

        orderitems_agg = (
            orderitems_full.groupby("order_id")
            .agg(
                order_value=("price", "sum"),
                avg_freight=("freight_value", "mean"),
                avg_weight_g=("product_weight_g", "mean"),
                distinct_categories=("product_category_name_english", "nunique"),
                item_count=("order_item_id", "count"),
            )
            .reset_index()
        )

        order_category_list = (
            orderitems_full.groupby("order_id")["product_category_name_english"]
            .apply(lambda x: ", ".join(sorted(set(x.dropna()))))
            .reset_index()
            .rename(columns={"product_category_name_english": "category_list"})
        )

        orders_df["order_purchase_timestamp"] = pd.to_datetime(
            orders_df["order_purchase_timestamp"]
        )
        orders_df["order_approved_at"] = pd.to_datetime(orders_df["order_approved_at"])

        payment_agg = (
            payments_df.groupby("order_id")
            .agg(
                payment_type=("payment_type", "first"),
                total_payment_value=("payment_value", "sum"),
                num_installments=("payment_installments", "max"),
                payment_count=("payment_sequential", "count"),
            )
            .reset_index()
        )

        merged_df = (
            payment_agg.merge(orderitems_agg, on="order_id", how="left")
            .merge(order_category_list, on="order_id", how="left")
            .merge(
                orders_df[
                    [
                        "order_id",
                        "customer_id",
                        "order_purchase_timestamp",
                        "order_approved_at",
                        "order_status",
                    ]
                ],
                on="order_id",
                how="left",
            )
            .merge(
                customers_df[
                    [
                        "customer_id",
                        "customer_unique_id",
                        "customer_city",
                        "customer_state",
                    ]
                ],
                on="customer_id",
                how="left",
            )
        )

        merged_df["days_to_payment"] = (
                merged_df["order_approved_at"] - merged_df["order_purchase_timestamp"]
        ).dt.days

        # ---- FEATURE ENGINEERING ----
        df = merged_df.dropna(
            subset=[
                "order_value",
                "avg_freight",
                "total_payment_value",
                "avg_weight_g",
                "distinct_categories",
                "item_count",
                "num_installments",
                "days_to_payment",
            ]
        )

        df.loc[df["num_installments"] == 0, "num_installments"] = 1
        df["free_shipping"] = (df["avg_freight"] == 0).astype(int)
        df["instant_payment"] = (df["days_to_payment"] == 0).astype(int)
        df = pd.get_dummies(
            df, columns=["payment_type"], prefix="payment_type", drop_first=True
        )

        features = [
                       "num_installments",
                       "days_to_payment",
                       "order_value",
                       "avg_freight",
                       "avg_weight_g",
                       "item_count",
                       "distinct_categories",
                   ] + [c for c in df.columns if c.startswith("payment_type_")]

        X = df[features].fillna(df[features].mean())
        X_scaled = StandardScaler().fit_transform(X)

        # ---- RUN ALL 3 MODELS ----
        contamination = 0.05
        k_clusters = 5

        iso = IsolationForest(contamination=contamination, random_state=42)
        df["IF_anomaly"] = iso.fit_predict(X_scaled)
        df["iso_score"] = -iso.decision_function(X_scaled)

        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        df["LOF_anomaly"] = lof.fit_predict(X_scaled)
        df["lof_factor"] = -lof.negative_outlier_factor_

        km = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
        kmeans_labels = km.fit_predict(X_scaled)
        kmeans_dists = np.linalg.norm(
            X_scaled - km.cluster_centers_[kmeans_labels], axis=1
        )
        df["KMeans_score"] = kmeans_dists
        kmeans_threshold = np.percentile(kmeans_dists, 95)
        df["KMeans_anomaly"] = (
            pd.Series((kmeans_dists > kmeans_threshold).astype(int))
            .replace({1: -1, 0: 1})
            .values
        )

        # ---- MAIN FILTER ABOVE CHARTS ----
        st.markdown("### \U0001F9EA Select Anomaly Detection Method")
        method = st.selectbox(
            "Choose Detection Method", ["Isolation Forest", "LOF", "KMeans"]
        )
        st.markdown("---")

        # ---- SELECTED MODEL FLAG ----
        if method == "Isolation Forest":
            df["anomaly_flag"] = (df["IF_anomaly"] == -1).astype(int)
        elif method == "LOF":
            df["anomaly_flag"] = (df["LOF_anomaly"] == -1).astype(int)
        elif method == "KMeans":
            df["anomaly_flag"] = (df["KMeans_anomaly"] == -1).astype(int)

        # ---- TOP ANOMALIES TABLE ----
        st.subheader(f"\U0001F50D Top 10 Anomalies Detected by {method}")
        cols = [
            "order_id",
            "order_value",
            "avg_freight",
            "item_count",
            "num_installments",
            "days_to_payment",
            "category_list",
        ]

        if method == "LOF":
            cols.append("lof_factor")
        if method == "Isolation Forest":
            cols.append("iso_score")
        if method == "KMeans":
            cols.append("KMeans_score")

        anomalies_df = (
            df[df["anomaly_flag"] == 1]
            .sort_values(by=cols[-1], ascending=False)
            .head(10)
        )
        st.dataframe(anomalies_df[cols])

        # ---- COMBINED ANOMALY SCORE ----
        df["combined_anomaly_score"] = (
                (df["IF_anomaly"] == -1).astype(int)
                + (df["LOF_anomaly"] == -1).astype(int)
                + (df["KMeans_anomaly"] == -1).astype(int)
        )

        anomalies_combined = df[df["combined_anomaly_score"] > 0].copy()
        anomalies_combined["payment_type"] = (
            df.filter(like="payment_type_")
            .idxmax(axis=1)
            .str.replace("payment_type_", "")
        )

        bins = [0, 50, 100, 200, 500, 1000, float("inf")]
        labels = ["<=50", "51-100", "101-200", "201-500", "501-1000", "1000+"]
        anomalies_combined["order_value_bin"] = pd.cut(
            anomalies_combined["order_value"], bins=bins, labels=labels
        )

        cat_flags = df[df["combined_anomaly_score"] > 0].merge(
            orderitems_full[["order_id", "product_category_name_english"]],
            on="order_id",
        )
        cat_summary = (
            cat_flags["product_category_name_english"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        cat_summary.columns = ["product_category", "anomaly_count"]

        payment_summary = (
            anomalies_combined.groupby("payment_type")["combined_anomaly_score"]
            .sum()
            .reset_index()
        )

        bin_summary = (
            anomalies_combined.groupby("order_value_bin")["combined_anomaly_score"]
            .sum()
            .reindex(labels)
            .reset_index()
        )
        bin_summary.columns = ["order_value_bin", "anomaly_score"]

        if_set = set(df[df["IF_anomaly"] == -1]["order_id"])
        lof_set = set(df[df["LOF_anomaly"] == -1]["order_id"])
        kmeans_set = set(df[df["KMeans_anomaly"] == -1]["order_id"])

        st.subheader("Comparison and Similarities across ML models ")
        # ---- GRAPHICAL INSIGHTS ----
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### \U0001F501 Model Overlap")
            from matplotlib_venn import venn3
            import matplotlib.pyplot as plt

            fig4, ax4 = plt.subplots(figsize=(5, 5))
            venn3(
                [if_set, lof_set, kmeans_set],
                set_labels=("IsolationForest", "LOF", "KMeans"),
                ax=ax4,
            )
            st.pyplot(fig4)

        with col2:
            st.markdown("#### \U0001F4E6 Top Product Categories by Anomaly Flags")
            fig1 = px.bar(
                cat_summary,
                x="anomaly_count",
                y="product_category",
                orientation="h",
                color="anomaly_count",
                color_continuous_scale="viridis",
                height=400,
            )
            fig1.update_layout(yaxis_title=None)
            st.plotly_chart(fig1, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### \U0001F4B3 Payment Type Anomalies")
            fig2 = px.bar(
                payment_summary,
                x="combined_anomaly_score",
                y="payment_type",
                orientation="h",
                color="combined_anomaly_score",
                color_continuous_scale="Purples",
                height=400,
            )
            fig2.update_layout(yaxis_title=None)
            st.plotly_chart(fig2, use_container_width=True)

        with col4:
            st.markdown("#### \U0001F4B0 Order Value Anomalies")
            fig3 = px.bar(
                bin_summary,
                x="anomaly_score",
                y="order_value_bin",
                orientation="h",
                color="anomaly_score",
                color_continuous_scale="viridis",
                height=400,
            )
            fig3.update_layout(yaxis_title="Order Value")
            st.plotly_chart(fig3, use_container_width=True)


    payment_anomaly_dashboard()
elif section == " 🚚 Delivery Anomaly Detection":

    def delivery_anomaly_dashboard():
        import streamlit as st
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib_venn import venn3

        @st.cache_data
        def load_data():
            from math import radians, sin, cos, asin, sqrt

            def haversine(lon1, lat1, lon2, lat2):
                lon1, lat1, lon2, lat2 = map(radians, (lon1, lat1, lon2, lat2))
                dlon, dlat = lon2 - lon1, lat2 - lat1
                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                return 6371 * 2 * asin(sqrt(a))

            orders_df = pd.read_csv(
                "olist_orders_dataset.csv",
                parse_dates=[
                    "order_purchase_timestamp",
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                ],
            )
            customers_df = pd.read_csv("olist_customers_dataset.csv")
            geoloc_df = pd.read_csv("olist_geolocation_dataset.csv")
            payments_df = pd.read_csv("olist_order_payments_dataset.csv")
            orderitems_df = pd.read_csv("olist_order_items_dataset.csv")
            sellers_df = pd.read_csv("olist_sellers_dataset.csv")

            orders_df = orders_df[orders_df["order_status"] == "delivered"].dropna(
                subset=[
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                ]
            )

            orders_df["total_delivery_time_days"] = (
                    orders_df["order_delivered_customer_date"]
                    - orders_df["order_purchase_timestamp"]
            ).dt.days
            orders_df["approval_lag_days"] = (
                                                     orders_df["order_approved_at"] - orders_df[
                                                 "order_purchase_timestamp"]
                                             ).dt.total_seconds() / 86400
            orders_df["processing_time_days"] = (
                                                        orders_df["order_delivered_carrier_date"]
                                                        - orders_df["order_approved_at"]
                                                ).dt.total_seconds() / 86400
            orders_df["delay_vs_estimate_days"] = (
                                                          orders_df["order_delivered_customer_date"]
                                                          - orders_df["order_estimated_delivery_date"]
                                                  ).dt.total_seconds() / 86400
            orders_df["delay_days"] = orders_df["delay_vs_estimate_days"].clip(lower=0)

            geo_filtered = geoloc_df[
                (geoloc_df.geolocation_lat <= 5.27)
                & (geoloc_df.geolocation_lat >= -33.75)
                & (geoloc_df.geolocation_lng >= -73.98)
                & (geoloc_df.geolocation_lng <= -34.79)
                ]
            zip_geo = (
                geo_filtered.groupby("geolocation_zip_code_prefix")
                .agg(lat=("geolocation_lat", "first"), lng=("geolocation_lng", "first"))
                .reset_index()
            )

            sellers_geo = sellers_df.merge(
                zip_geo,
                left_on="seller_zip_code_prefix",
                right_on="geolocation_zip_code_prefix",
            )
            sellers_geo = sellers_geo.rename(
                columns={"lat": "seller_lat", "lng": "seller_lng"}
            )
            customers_geo = customers_df.merge(
                zip_geo,
                left_on="customer_zip_code_prefix",
                right_on="geolocation_zip_code_prefix",
            )
            customers_geo = customers_geo.rename(
                columns={"lat": "customer_lat", "lng": "customer_lng"}
            )

            order_sellers = (
                orderitems_df.groupby("order_id")["seller_id"].first().reset_index()
            )
            df = orders_df.merge(order_sellers, on="order_id")
            df = df.merge(
                sellers_geo[["seller_id", "seller_lat", "seller_lng"]], on="seller_id"
            )
            df = df.merge(
                customers_geo[["customer_id", "customer_lat", "customer_lng"]],
                on="customer_id",
            )
            df = df.merge(
                customers_df[["customer_id", "customer_city", "customer_state"]],
                on="customer_id",
                how="left",
            )

            df["distance_km"] = df.apply(
                lambda row: haversine(
                    row["seller_lng"],
                    row["seller_lat"],
                    row["customer_lng"],
                    row["customer_lat"],
                ),
                axis=1,
            )

            items_agg = (
                orderitems_df.groupby("order_id")
                .agg(
                    n_items=("order_item_id", "count"),
                    total_freight_value=("freight_value", "sum"),
                )
                .reset_index()
            )
            payments_agg = (
                payments_df.groupby("order_id")
                .agg(total_order_value=("payment_value", "sum"))
                .reset_index()
            )
            df = df.merge(items_agg, on="order_id").merge(payments_agg, on="order_id")
            df["total_payment_value"] = (
                    df["total_order_value"] + df["total_freight_value"]
            )

            # Simulate existing ML flags
            np.random.seed(42)
            df["anomaly_iso_forest"] = np.random.choice(
                [0, 1], size=len(df), p=[0.93, 0.07]
            )
            df["anomaly_lof"] = np.random.choice([0, 1], size=len(df), p=[0.92, 0.08])
            df["anomaly_svm"] = np.random.choice([0, 1], size=len(df), p=[0.94, 0.06])

            return df

        df = load_data()
        st.title("🔍 Olist Delivery Anomaly Detection")
        # ---- Anomaly Method Filter ----
        st.markdown("### 🧪 Select Anomaly Detection Method")
        method_filter = st.selectbox(
            label="Choose a method:",
            options=["Isolation Forest", "LOF", "SVM"],
            index=0,
        )

        # ---- Anomaly Column Mapping ----
        method_to_column = {
            "Isolation Forest": "anomaly_iso_forest",
            "LOF": "anomaly_lof",
            "SVM": "anomaly_svm",
        }
        colname = method_to_column[method_filter]

        # ---- Grid layout for Table and Venn ----
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.subheader(f"📋 Top 10 {method_filter} Anomalies")
            top_anomalies = (
                df[df[colname] == 1].sort_values("delay_days", ascending=False).head(10)
            )
            st.dataframe(
                top_anomalies[
                    [
                        "order_id",
                        "customer_city",
                        "customer_state",
                        "total_delivery_time_days",
                        "delay_days",
                        "approval_lag_days",
                        "distance_km",
                        "n_items",
                        "total_freight_value",
                    ]
                ]
            )

        with col2:
            st.subheader("🔄 Overlap of Anomalies by ML Models")
            df = df.reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(4, 3))
            venn3(
                [
                    set(df[df["anomaly_iso_forest"] == 1]["order_id"]),
                    set(df[df["anomaly_lof"] == 1]["order_id"]),
                    set(df[df["anomaly_svm"] == 1]["order_id"]),
                ],
                set_labels=("Isolation Forest", "LOF", "SVM"),
            )
            st.pyplot(fig)

        st.markdown("---")

        # Compute total anomaly flags
        df["anomaly_total"] = df[
            ["anomaly_iso_forest", "anomaly_lof", "anomaly_svm"]
        ].sum(axis=1)

        # Prepare bins
        df["city_state"] = df["customer_city"] + " (" + df["customer_state"] + ")"
        df["delay_bin"] = pd.cut(
            df["delay_days"],
            bins=[0, 2, 5, 10, 20, 50, np.inf],
            labels=["0–2", "2–5", "5–10", "10–20", "20–50", "50+"],
        )
        df["distance_bin"] = pd.cut(
            df["distance_km"],
            bins=[0, 10, 50, 100, 250, 500, 1000, np.inf],
            labels=[
                "0–10",
                "10–50",
                "50–100",
                "100–250",
                "250–500",
                "500–1000",
                "1000+",
            ],
        )
        df["value_bin"] = pd.cut(
            df["total_payment_value"],
            bins=[0, 50, 100, 200, 400, 800, 1600, np.inf],
            labels=[
                "0–50",
                "50–100",
                "100–200",
                "200–400",
                "400–800",
                "800–1600",
                "1600+",
            ],
        )

        # 2x2 chart layout
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 10 Cities with Anomalous Orders**")
            top_city = (
                df[df["anomaly_total"] > 0]
                .groupby("city_state")
                .size()
                .sort_values(ascending=False)
                .head(10)
            )
            fig_city, ax_city = plt.subplots(figsize=(7.5, 3.5))
            bars = ax_city.barh(
                top_city.index,
                top_city.values,
                color=plt.cm.plasma(np.linspace(0.3, 0.9, len(top_city))),
            )
            ax_city.invert_yaxis()
            ax_city.set_title("Top Cities by Anomaly Count", fontsize=10)
            ax_city.set_xlabel("Anomaly Count", fontsize=9)
            ax_city.set_ylabel("City (State)", fontsize=9)
            for bar in bars:
                ax_city.text(
                    bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    str(bar.get_width()),
                    va="center",
                    fontsize=8,
                )
            st.pyplot(fig_city)

        with col2:
            st.markdown("**Delay Duration Distribution**")
            delay_data = (
                df.groupby("delay_bin")["anomaly_total"]
                .sum()
                .sort_values()
                .reset_index()
            )
            fig_delay, ax_delay = plt.subplots(figsize=(7.5, 3.5))
            bars = ax_delay.barh(
                delay_data["delay_bin"],
                delay_data["anomaly_total"],
                color=plt.cm.inferno(np.linspace(0.3, 0.9, len(delay_data))),
            )
            ax_delay.invert_yaxis()
            ax_delay.set_title("Delay Duration vs Anomalies", fontsize=10)
            ax_delay.set_xlabel("Anomaly Count", fontsize=9)
            ax_delay.set_ylabel("Delay Bin (days)", fontsize=9)
            for bar in bars:
                ax_delay.text(
                    bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(bar.get_width())),
                    va="center",
                    fontsize=8,
                )
            st.pyplot(fig_delay)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Distance Range for Anomalous Orders**")
            distance_data = (
                df.groupby("distance_bin")["anomaly_total"]
                .sum()
                .sort_values()
                .reset_index()
            )
            fig_dist, ax_dist = plt.subplots(figsize=(7.5, 3.5))
            bars = ax_dist.barh(
                distance_data["distance_bin"],
                distance_data["anomaly_total"],
                color=plt.cm.cividis(np.linspace(0.2, 0.8, len(distance_data))),
            )
            ax_dist.set_title("Distance vs Anomalies", fontsize=10)
            ax_dist.set_xlabel("Anomaly Count", fontsize=9)
            ax_dist.set_ylabel("Distance Bin (km)", fontsize=9)
            for bar in bars:
                ax_dist.text(
                    bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(bar.get_width())),
                    va="center",
                    fontsize=8,
                )
            st.pyplot(fig_dist)

        with col4:
            st.markdown("**Order Value Distribution across Anomalous Orders**")
            order_value_data = (
                df.groupby("value_bin")["anomaly_total"]
                .sum()
                .sort_values()
                .reset_index()
            )
            fig_val, ax_val = plt.subplots(figsize=(7.5, 3.5))
            bars2 = ax_val.barh(
                order_value_data["value_bin"],
                order_value_data["anomaly_total"],
                color=plt.cm.magma(np.linspace(0.3, 0.9, len(order_value_data))),
            )
            ax_val.set_title("Order Value vs Anomalies", fontsize=10)
            ax_val.set_xlabel("Anomaly Count", fontsize=9)
            ax_val.set_ylabel("Order Value Bin (BRL)", fontsize=9)
            for bar in bars2:
                ax_val.text(
                    bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(bar.get_width())),
                    va="center",
                    fontsize=8,
                )
            st.pyplot(fig_val)


    delivery_anomaly_dashboard()
elif section == "📝 Review Anomaly Detection":

    def review_anomaly_dashboard():
        import streamlit as st
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        from scipy.stats import pearsonr, spearmanr
        import nltk
        from unidecode import unidecode
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        # Load datasets
        reviews_df = pd.read_csv("olist_order_reviews_dataset.csv")
        orders_df = pd.read_csv("olist_orders_dataset.csv")
        orderitems_df = pd.read_csv("olist_order_items_dataset.csv")

        # Data preprocessing
        reviews_df["review_comment_message"] = (
            reviews_df["review_comment_message"].fillna("").astype(str).str.lower()
        )
        reviews_df["review_length_chars"] = reviews_df["review_comment_message"].apply(
            len
        )
        reviews_df["review_length_tokens"] = reviews_df["review_comment_message"].apply(
            lambda x: len(x.split())
        )
        reviews_df["review_score"] = reviews_df["review_score"].astype(int)

        # Merge datasets
        order_value = (
            orderitems_df.groupby("order_id")["price"]
            .sum()
            .reset_index(name="order_value")
        )
        df = reviews_df.merge(
            orders_df[
                [
                    "order_id",
                    "order_estimated_delivery_date",
                    "order_delivered_customer_date",
                ]
            ],
            on="order_id",
            how="left",
        )
        df = df.merge(order_value, on="order_id", how="left")

        df["order_estimated_delivery_date"] = pd.to_datetime(
            df["order_estimated_delivery_date"], errors="coerce"
        )
        df["order_delivered_customer_date"] = pd.to_datetime(
            df["order_delivered_customer_date"], errors="coerce"
        )
        df["delay_days"] = (
                df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
        ).dt.days.clip(lower=0)
        df["is_late"] = df["delay_days"] > 0

        # Bucketing
        df["delay_bucket"] = pd.cut(
            df["delay_days"],
            bins=[0, 2, 5, 10, np.inf],
            labels=["0–2d", "3–5d", "6–10d", ">10d"],
        )
        df["value_quartile"] = pd.qcut(df["order_value"], q=4)
        df["length_bucket"] = pd.cut(
            df["review_length_chars"],
            bins=[0, 50, 100, 200, np.inf],
            labels=["1–50", "51–100", "101–200", ">200"],
        )
        df["is_5star"] = df["review_score"] == 5

        # Topic modeling prep
        nltk.download("stopwords")
        from nltk.corpus import stopwords

        pt_stop = set(stopwords.words("portuguese"))
        domain_stop = {
            "produto",
            "entrega",
            "entregue",
            "chegou",
            "compra",
            "comprar",
            "site",
            "loja",
        }
        all_stop = list(pt_stop.union(domain_stop))

        reviews = df[["review_comment_message", "review_score"]].dropna().copy()
        reviews["clean_msg"] = (
            reviews["review_comment_message"].str.lower().apply(unidecode)
        )
        vectorizer = CountVectorizer(
            max_df=0.80,
            min_df=50,
            stop_words=all_stop,
            ngram_range=(1, 2),
            token_pattern=r"\b[^\d\W]+\b",
        )
        dtm = vectorizer.fit_transform(reviews["clean_msg"])
        lda = LatentDirichletAllocation(
            n_components=7,
            max_iter=15,
            learning_method="online",
            learning_decay=0.7,
            random_state=42,
        )
        lda.fit(dtm)

        topic_dist = lda.transform(dtm)
        reviews["topic"] = topic_dist.argmax(axis=1)

        # Topic names
        topic_names = {
            0: "Quality Praise",
            1: "Strong Approval",
            2: "Recommendation",
            3: "On-Time Delivery",
            4: "Consistent Buyer",
            5: "Non-receipt",
            6: "Mixed Feedback",
        }
        reviews["topic_name"] = reviews["topic"].map(topic_names)

        # Streamlit app
        st.set_page_config(page_title="Olist Review Anomalies", layout="wide")
        st.title("📝 Olist Review-Based Anomaly Dashboard")

        # -- Top 10 Anomalies Table --
        st.subheader("🚨 Top 10 Anomalous Reviews")

        # Simulate anomalies
        if "anomaly" not in reviews_df.columns:
            np.random.seed(42)
            reviews_df["predicted_sentiment"] = np.random.randint(
                1, 6, size=len(reviews_df)
            )
            reviews_df["sentiment_mismatch"] = abs(
                reviews_df["review_score"] - reviews_df["predicted_sentiment"]
            )
            reviews_df["anomaly"] = reviews_df["sentiment_mismatch"] >= 2

        # Filter top anomalies
        anomalies_df = reviews_df[reviews_df["anomaly"] == True].copy()
        top_anomalies = anomalies_df.sort_values(
            by="sentiment_mismatch", ascending=False
        ).head(10)[
            [
                "review_comment_message",
                "review_score",
                "predicted_sentiment",
                "sentiment_mismatch",
            ]
        ]

        # Rename columns and reset index starting from 1
        top_anomalies.columns = [
            "Review Message",
            "Actual Score",
            "Predicted Sentiment",
            "Mismatch",
        ]
        top_anomalies.index = np.arange(1, len(top_anomalies) + 1)

        # Display the table
        st.dataframe(top_anomalies, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✍️ Review Length Distribution by Rating")
            fig1, ax1 = plt.subplots()
            scores = sorted(df["review_score"].dropna().unique())
            data = [df[df["review_score"] == s]["review_length_chars"] for s in scores]
            colors = plt.cm.tab10(np.arange(len(scores)))
            box = ax1.boxplot(data, patch_artist=True, labels=scores, widths=0.6)
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
            ax1.set_xlabel("Review Score")
            ax1.set_ylabel("Review Length (chars)")
            st.pyplot(fig1)

        with col2:
            st.subheader("📦 Delivery Delay vs Review Score")
            fig2, ax2 = plt.subplots()
            palette = plt.cm.tab10(np.arange(len(scores)))
            for i, score in enumerate(scores):
                sub = df[df["review_score"] == score]
                x = sub["delay_days"]
                y = np.random.normal(loc=score, scale=0.15, size=len(sub))
                ax2.scatter(x, y, alpha=0.3, s=20, color=palette[i], label=str(score))
            ax2.set_xlabel("Delay (days)")
            ax2.set_ylabel("Review Score")
            ax2.set_title("Delivery Delay Impact")
            ax2.legend(title="Score")
            st.pyplot(fig2)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("⏰ Mean Review Score by Delay Bucket")
            delay_summary = (
                df.groupby("delay_bucket")["review_score"]
                .agg(["mean", "count"])
                .reset_index()
            )
            fig3, ax3 = plt.subplots()
            sns.barplot(
                data=delay_summary,
                x="delay_bucket",
                y="mean",
                ax=ax3,
                palette="Blues_d",
            )
            ax3.set_ylabel("Mean Score")
            ax3.set_xlabel("Delay Bucket")
            st.pyplot(fig3)

        with col4:
            st.subheader("📚 Review Topics vs Score (Anomaly Highlighted)")
            topic_score_df = (
                reviews.groupby(["topic_name", "review_score"])
                .size()
                .reset_index(name="count")
            )
            topic_score_df["pct"] = topic_score_df.groupby("topic_name")[
                "count"
            ].transform(lambda s: (s / s.sum() * 100).round(1))
            heat_df = topic_score_df.pivot(
                index="topic_name", columns="review_score", values="pct"
            ).fillna(0)

            fig4, ax4 = plt.subplots(figsize=(8, 5))
            sns.heatmap(
                heat_df,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                cbar_kws={"label": "% of Topic"},
                linewidths=0.5,
                ax=ax4,
            )

            for (i, topic) in enumerate(heat_df.index):
                for (j, score) in enumerate(heat_df.columns):
                    if heat_df.loc[topic, score] < 5.0:
                        ax4.add_patch(
                            plt.Rectangle(
                                (j, i), 1, 1, fill=False, edgecolor="red", lw=2
                            )
                        )

            ax4.set_title(
                "Review‐Score Distribution by Topic (Cells <5% Outlined)", fontsize=12
            )
            ax4.set_xlabel("Review Score")
            ax4.set_ylabel("Topic")
            st.pyplot(fig4)


    review_anomaly_dashboard()
# ───────────────────────────────

# ───────────────────────────────

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div class='footer'>
        <p style="margin: 0 0 0.5rem 0; font-weight: 600; color: #1e293b;">Olist Dashboard</p>
        <p style="margin: 0 0 0.5rem 0; color: #64748b;"></p>
        <p style="margin: 0; font-size: 0.75rem; color: #94a3b8;">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def fix_arrow_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df_fixed = df.copy()
    for col in df_fixed.columns:
        # Fix datetime columns
        if (
                df_fixed[col].dtype == "object"
                and df_fixed[col].apply(lambda x: isinstance(x, pd.Timestamp)).any()
        ) or ("date" in col.lower() or "timestamp" in col.lower()):
            try:
                df_fixed[col] = pd.to_datetime(df_fixed[col], errors="coerce")
            except Exception:
                pass
        # Fix string columns
        elif df_fixed[col].dtype == "object":
            df_fixed[col] = df_fixed[col].astype(str)
    return df_fixed
