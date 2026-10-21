import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="Warehouse Stockout Risk",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background: #f6f8fb; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
    .hero {
        padding: 28px 32px; border-radius: 20px;
        background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #374151 100%);
        color: white; margin-bottom: 25px; box-shadow: 0 12px 35px rgba(0,0,0,.12);
    }
    .hero h1 { margin: 0; font-size: 2.3rem; font-weight: 750; }
    .hero p { margin: 8px 0 0; color: #d1d5db; font-size: 1rem; }
    .card {
        background: white; padding: 22px; border-radius: 16px;
        border: 1px solid #e5e7eb; box-shadow: 0 5px 18px rgba(0,0,0,.05); margin-bottom: 18px;
    }
    .result-card {
        padding: 30px; border-radius: 18px; background: white;
        border: 1px solid #e5e7eb; text-align: center; box-shadow: 0 8px 25px rgba(0,0,0,.06);
    }
    .risk-title { font-size: 1rem; color: #6b7280; margin-bottom: 8px; }
    .risk-value { font-size: 3rem; font-weight: 800; margin: 5px 0; }
    .metric-label { color: #6b7280; font-size: .85rem; }
    .metric-value { font-size: 1.45rem; font-weight: 700; color: #111827; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e5e7eb; padding: 15px; border-radius: 14px;
    }
    .small-note { color: #6b7280; font-size: .85rem; }
    .stButton > button { width: 100%; border-radius: 10px; font-weight: 700; padding: 0.65rem; }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "Current_Stock", "Average_Daily_Sales", "Sales_Last_7_Days",
    "Sales_Last_30_Days", "Supplier_Lead_Time_Days", "Reorder_Level",
]
TARGET = "Stockout_Risk"

if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "accuracy" not in st.session_state:
    st.session_state.accuracy = None
if "df" not in st.session_state:
    st.session_state.df = None

st.markdown("""
<div class="hero">
    <h1>📦 Warehouse Stockout Risk Prediction</h1>
    <p>Machine Learning system for predicting whether a product has
    <b>Low</b>, <b>Medium</b>, or <b>High</b> stockout risk.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Model Setup")
    uploaded_file = st.file_uploader(
        "Upload warehouse_stockout.csv", type=["csv"],
        help="Upload the same CSV dataset used in the notebook."
    )
    st.divider()
    st.markdown("### Model")
    st.info(
        "Logistic Regression\n\nFeatures used:\n"
        "• Current Stock\n• Average Daily Sales\n• Sales Last 7 Days\n"
        "• Sales Last 30 Days\n• Supplier Lead Time\n• Reorder Level"
    )
    st.divider()
    st.markdown("### Notebook model")
    st.caption("Train/Test split: 80/20")
    st.caption("Scaler: StandardScaler")
    st.caption("Random state: 42")

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.session_state.df = raw_df.copy()
        missing_required = [col for col in FEATURES + [TARGET] if col not in raw_df.columns]
        if missing_required:
            st.error("The uploaded CSV is missing required columns: " + ", ".join(missing_required))
            st.stop()

        df = raw_df.copy()
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        if "Product_Category" in df.columns:
            df["Product_Category"] = df["Product_Category"].str.strip()
        df = df[df["Current_Stock"] >= 0]
        df = df[df["Average_Daily_Sales"] >= 0]
        X, y = df[FEATURES], df[TARGET]
        if y.nunique() < 2:
            st.error("Stockout_Risk must contain at least two classes.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.accuracy = accuracy
        st.session_state.clean_df = df
        st.session_state.report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
    except Exception as e:
        st.error(f"Could not process the dataset: {e}")
        st.stop()

if st.session_state.model is None:
    st.markdown("""
    <div class="card">
        <h3>🚀 Get Started</h3>
        <p>Upload your <b>warehouse_stockout.csv</b> from the sidebar.
        The app will clean the data, train the Logistic Regression model,
        and activate the prediction panel.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Dataset", "Not loaded")
    with c2:
        st.metric("Model", "Logistic Regression")
    with c3:
        st.metric("Status", "Waiting")
    st.stop()

df = st.session_state.clean_df
st.subheader("📊 Model Overview")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Records Used", f"{len(df):,}")
with m2:
    st.metric("Features", len(FEATURES))
with m3:
    st.metric("Model Accuracy", f"{st.session_state.accuracy * 100:.2f}%")
with m4:
    st.metric("Risk Classes", df[TARGET].nunique())
st.caption(
    f"Cleaning reduced the dataset to {len(df):,} usable rows. "
    "The final prediction model uses six numerical features."
)

tab_predict, tab_data, tab_performance = st.tabs(
    ["🔮 Prediction", "🗃️ Dataset", "📈 Model Performance"]
)

with tab_predict:
    st.markdown("""
    <div class="card">
        <h3>🔮 Predict Stockout Risk</h3>
        <p class="small-note">Enter the six inputs used by the trained model.</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        current_stock = st.number_input("Current Stock", min_value=0.0, value=10.0, step=1.0)
        avg_daily_sales = st.number_input("Average Daily Sales", min_value=0.0, value=1.0, step=0.1)
        sales_7 = st.number_input("Sales Last 7 Days", min_value=0.0, value=2.0, step=1.0)
    with col2:
        sales_30 = st.number_input("Sales Last 30 Days", min_value=0.0, value=3.0, step=1.0)
        supplier_lead = st.number_input("Supplier Lead Time (Days)", min_value=0.0, value=5.0, step=1.0)
        reorder_level = st.number_input("Reorder Level", min_value=0.0, value=5.0, step=1.0)

    if st.button("🚀 Predict Stockout Risk", type="primary"):
        input_data = pd.DataFrame([[
            current_stock, avg_daily_sales, sales_7, sales_30, supplier_lead, reorder_level
        ]], columns=FEATURES)
        input_scaled = st.session_state.scaler.transform(input_data)
        prediction = st.session_state.model.predict(input_scaled)[0]
        probabilities = None
        if hasattr(st.session_state.model, "predict_proba"):
            probabilities = st.session_state.model.predict_proba(input_scaled)[0]
            classes = st.session_state.model.classes_

        st.divider()
        st.subheader("Prediction Result")
        if prediction == "High":
            symbol, message = "🔴", "Immediate attention recommended."
        elif prediction == "Medium":
            symbol, message = "🟠", "Monitor inventory and supplier timing."
        else:
            symbol, message = "🟢", "Current inputs indicate relatively lower risk."
        st.markdown(f"""
        <div class="result-card">
            <div class="risk-title">Predicted Stockout Risk</div>
            <div class="risk-value">{symbol} {prediction}</div>
            <div>{message}</div>
        </div>
        """, unsafe_allow_html=True)

        if probabilities is not None:
            st.subheader("Model Confidence")
            prob_df = pd.DataFrame({"Risk Level": classes, "Probability": probabilities})
            prob_df = prob_df.sort_values("Probability", ascending=False)
            st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True, hide_index=True)
        st.subheader("Input Summary")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Current Stock", f"{current_stock:g}")
        with s2:
            st.metric("Daily Sales", f"{avg_daily_sales:.1f}")
        with s3:
            st.metric("Approx. Days of Stock", "—" if avg_daily_sales == 0 else f"{current_stock / avg_daily_sales:.1f}")

with tab_data:
    st.subheader("🗃️ Cleaned Dataset")
    st.dataframe(df, use_container_width=True, height=520)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Cleaned Dataset", data=csv,
        file_name="cleaned_warehouse_stockout.csv", mime="text/csv")

with tab_performance:
    st.subheader("📈 Logistic Regression Performance")
    st.metric("Test Accuracy", f"{st.session_state.accuracy * 100:.2f}%")
    report = st.session_state.report
    performance_rows = []
    for label in ["High", "Low", "Medium"]:
        if label in report:
            performance_rows.append({
                "Risk": label, "Precision": report[label]["precision"],
                "Recall": report[label]["recall"], "F1 Score": report[label]["f1-score"],
                "Support": int(report[label]["support"])
            })
    if performance_rows:
        perf_df = pd.DataFrame(performance_rows)
        st.dataframe(perf_df.style.format({
            "Precision": "{:.2f}", "Recall": "{:.2f}", "F1 Score": "{:.2f}"
        }), use_container_width=True, hide_index=True)
    st.info("This app follows the final model structure in the notebook: six numerical inputs → StandardScaler → LogisticRegression → Low / Medium / High stockout risk.")

st.divider()
st.caption("📦 Warehouse Stockout Risk Prediction System • Streamlit")
