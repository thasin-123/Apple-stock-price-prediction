import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Apple Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Apple Stock Price Prediction using XGBoost")

st.markdown("""
### Forecast Apple's Stock Price using a trained XGBoost Machine Learning Model

This application allows you to:

- 📊 View Historical Stock Prices
- 📈 Forecast Future Prices
- 📅 Select Forecast Horizon (1–30 Business Days)
- 📥 Download Forecast Results
""")

# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv("P675 DATASET.csv")

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    return df


df = load_data()

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = joblib.load("xgboost_stock_model.pkl")
    

    return model


model = load_model()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Forecast Settings")

forecast_days = st.sidebar.slider(

    "Select Forecast Days",

    min_value=1,

    max_value=30,

    value=30

)

show_table = st.sidebar.checkbox(

    "Show Forecast Table",

    value=True

)

show_chart = st.sidebar.checkbox(

    "Show Forecast Chart",

    value=True

)

download_csv = st.sidebar.checkbox(

    "Enable CSV Download",

    value=True

)

# ==========================================
# MODEL INFORMATION
# ==========================================

st.subheader("Model Information")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Algorithm",
    "XGBoost"
)

c2.metric(
    "Features",
    "12"
)

c3.metric(
    "Dataset Size",
    len(df)
)

c4.metric(
    "Forecast",
    f"{forecast_days} Days"
)

st.divider()

# ==========================================
# HISTORICAL STOCK PRICE
# ==========================================

st.subheader("📊 Historical Apple Stock Price")

col1, col2 = st.columns(2)

with col1:

    start_date = st.date_input(
        "Start Date",
        value=df["Date"].min()
    )

with col2:

    end_date = st.date_input(
        "End Date",
        value=df["Date"].max()
    )

filtered_df = df[
    (df["Date"] >= pd.to_datetime(start_date))
    &
    (df["Date"] <= pd.to_datetime(end_date))
]

fig, ax = plt.subplots(figsize=(15,6))

ax.plot(

    filtered_df["Date"],

    filtered_df["Adj Close"],

    color="royalblue",

    linewidth=2,

    label="Adj Close"

)

ax.set_title(
    "Apple Historical Stock Price"
)

ax.set_xlabel("Date")

ax.set_ylabel("Price ($)")

ax.grid(True)

ax.legend()

st.pyplot(fig)

# ==========================================
# DATASET INFORMATION
# ==========================================

st.subheader("📁 Dataset Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Rows",
    len(df)
)

c2.metric(
    "Columns",
    len(df.columns)
)

c3.metric(
    "First Date",
    str(df["Date"].min().date())
)

c4.metric(
    "Last Date",
    str(df["Date"].max().date())
)

# ==========================================
# HISTORICAL DATA TABLE
# ==========================================

with st.expander("View Historical Dataset"):

    st.dataframe(

        filtered_df[
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume"
            ]
        ],

        use_container_width=True

    )

# ==========================================
# BASIC STATISTICS
# ==========================================

with st.expander("Statistical Summary"):

    st.dataframe(

        filtered_df.describe(),

        use_container_width=True

    )

st.divider()

# ==========================================
# FORECAST SECTION
# ==========================================

st.subheader("📈 Future Stock Price Forecast")

predict_button = st.button(
    "Generate Forecast"
)



# ==========================================
# XGBOOST FORECAST
# ==========================================

if predict_button:

    with st.spinner("Generating Forecast..."):

        history = df["Adj Close"].tolist()

        future_predictions = []

        progress_bar = st.progress(0)

        for i in range(forecast_days):

            lag_1 = history[-1]
            lag_7 = history[-7]
            lag_30 = history[-30]

            ma10 = np.mean(history[-10:])
            ma30 = np.mean(history[-30:])
            ma50 = np.mean(history[-50:])

            returns = pd.Series(history).pct_change()

            volatility = returns[-30:].std()

            ema10 = (
                pd.Series(history[-10:])
                .ewm(span=10, adjust=False)
                .mean()
                .iloc[-1]
            )

            future_date = (
                df["Date"].iloc[-1]
                + pd.offsets.BDay(i + 1)
            )

            future_features = pd.DataFrame({

                "MA10":[ma10],
                "MA30":[ma30],
                "MA_50":[ma50],
                "Volatility":[volatility],
                "Lag_1":[lag_1],
                "Lag_7":[lag_7],
                "Lag_30":[lag_30],
                "EMA_10":[ema10],
                "Year":[future_date.year],
                "Month":[future_date.month],
                "Day":[future_date.day],
                "Weekday":[future_date.weekday()]

            })

            next_price = model.predict(future_features)[0]

            future_predictions.append(next_price)

            history.append(next_price)

            progress_bar.progress((i + 1) / forecast_days)

        future_dates = pd.bdate_range(

            start=df["Date"].iloc[-1] + pd.Timedelta(days=1),

            periods=forecast_days

        )

        forecast_df = pd.DataFrame({

            "Date":future_dates,

            "Forecast Price":np.round(future_predictions,2)

        })

    st.success("Forecast Generated Successfully!")

    st.divider()
    # ==========================================
    # FORECAST TABLE
    # ==========================================

    if show_table:

        st.subheader("📋 Forecasted Stock Prices")
        forecast_df["Date"] = forecast_df["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            forecast_df,
            use_container_width=True
        )

    # ==========================================
    # DOWNLOAD CSV
    # ==========================================

    if download_csv:

        csv = forecast_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name="Apple_Forecast.csv",
            mime="text/csv"
        )

    # ==========================================
    # FORECAST GRAPH
    # ==========================================

    if show_chart:

        st.subheader("📈 Historical + Forecast")

        fig2, ax2 = plt.subplots(figsize=(15,7))

        ax2.plot(
            df["Date"],
            df["Adj Close"],
            color="royalblue",
            linewidth=2,
            label="Historical Price"
        )

        ax2.plot(
            future_dates,
            future_predictions,
            color="red",
            linewidth=3,
            marker="o",
            label="Forecast"
        )

        ax2.axvline(
            x=df["Date"].iloc[-1],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Forecast Start"
        )

        ax2.set_title("Apple Stock Price Forecast")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Adj Close Price ($)")
        ax2.grid(True)
        ax2.legend()

        st.pyplot(fig2)


