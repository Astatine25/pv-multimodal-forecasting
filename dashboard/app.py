import streamlit as st
import numpy as np

st.title("PV Power Forecast Dashboard")

forecast = np.random.rand(24)
st.line_chart(forecast)
