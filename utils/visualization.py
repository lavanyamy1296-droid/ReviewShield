import streamlit as st

def show_distribution(counts):
    st.markdown("### Prediction distribution")
    st.bar_chart({"Number of reviews": counts})
