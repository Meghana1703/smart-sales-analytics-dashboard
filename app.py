import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model.sales_model import train_model, predict_revenue

st.set_page_config(page_title="Sales Analytics", layout="wide")

# Load styles
st.markdown(f"<style>{open('assets/style.css').read()}</style>", unsafe_allow_html=True)
st.markdown(open("assets/header.html").read(), unsafe_allow_html=True)

df = pd.read_csv("data/sales_data.csv")

# KPIs
total_revenue = df["revenue"].sum()
total_profit = df["profit"].sum()
total_units = df["units_sold"].sum()

col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='card'><h3>Total Revenue</h3><div class='metric'>₹{total_revenue:,}</div></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card'><h3>Total Profit</h3><div class='metric'>₹{total_profit:,}</div></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card'><h3>Units Sold</h3><div class='metric'>{total_units}</div></div>", unsafe_allow_html=True)

# Charts
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📈 Revenue Trend")
monthly = df.groupby("month")["revenue"].sum()
fig, ax = plt.subplots()
ax.plot(monthly.index, monthly.values, marker='o')
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# ML Prediction
model = train_model(df)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔮 Revenue Forecast")
month = st.slider("Select Future Month", 13, 24)
prediction = predict_revenue(model, month)
st.success(f"Predicted Revenue: ₹{int(prediction):,}")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(open("assets/footer.html").read(), unsafe_allow_html=True)
