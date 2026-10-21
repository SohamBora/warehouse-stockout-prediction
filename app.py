import streamlit as st
import joblib


model =  joblib.load("model.joblib")
Check_these = joblib.load("Check_These.joblib")
st.title("📦 Warehouse Stockout Risk Prediction & Recommendation System")


Product_ID = st.number_input(
    "📦 Enter Product ID",
    min_value=0,
    max_value=120,
    value=0,
    step=1,
    format="%d"
)



Current_Stock = st.number_input(
    "📦 Enter Current Stock")


Average_Daily_Sales = st.number_input(
    "📈 Enter Average Daily Sales")


Sales_Last_7_Days = st.number_input(
    "📊 Enter Sales Last 7 Days")


Sales_Last_30_Days = st.number_input(
    "📊 Enter Sales Last 30 Days")


Supplier_Lead_Time = st.number_input(
    "🚚 Enter Supplier Lead Time")

Supplier_Reliability = st.number_input("🚚 Enter Supplier  Reliability")


Reorder_Level = st.number_input(
    "🔄 Enter Reorder Level")

Pending_Orders = st.number_input("🔄 Enter Pending Orders")



if st.button(
    "🔮 Predict Stockout Risk"
):

    INPUT = [[
        Product_ID,
        Current_Stock,
        Average_Daily_Sales,
        Sales_Last_7_Days,
        Sales_Last_30_Days,
        Supplier_Lead_Time,
        Supplier_Reliability,
        Reorder_Level,
        Pending_Orders
    ]]


    # Make prediction
    Prediction = model.predict(INPUT)[0]


    # Display prediction
    st.write("🚨 Stockout Risk:", Prediction)


    # Recommendation
    if Prediction == "High":

        st.error(
            f"🔴 HIGH STOCKOUT RISK: {Prediction}"
        )

        st.warning(
            "⚠️ Order inventory immediately!"
        )


    elif Prediction == "Medium":

        st.warning(
            f"🟡 MEDIUM STOCKOUT RISK: {Prediction}"
        )

        st.info(
            "📅 Order inventory within the next few days."
        )


    else:

        st.success(
            f"🟢 LOW STOCKOUT RISK: {Prediction}"
        )

        st.success(
            "✅ No need to order inventory right now."
        )

    st.warning("Look at These Products they are also in High Risk")
    st.dataframe(Check_these)
