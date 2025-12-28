import streamlit as st
import numpy as np

st.title("PV Power Forecast Dashboard")

data = np.random.rand(100)
st.line_chart(data)
st.caption("Prototype visualization for real-time forecasting")
