import streamlit as st
import pandas as pd
import numpy as np

st.title(':herb: Plant Disease Detection')
st.write("This is the App to detect Plant disease by given a example image:")

enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)

if picture:
    st.image(picture)

st.button("Reset", type="primary")
if st.button("Submit"):
    st.write("Submit")
else:
    st.write("Failed to Submit")
