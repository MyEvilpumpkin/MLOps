import sys

import streamlit as st
from streamlit.web import cli as stcli

from model import load as load_model


@st.cache_resource
def get_model():
    return load_model()


def main():
    st.title("Iris Classification")
    model = get_model()

    st.subheader("Inputs")
    sliders = list()
    for feature in model.feature_names:
        sliders.append(st.slider(feature, min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f"))

    st.subheader("Prediction")
    predicted_class = model.predict([sliders])[0]
    predicted_class_name = model.target_names[predicted_class]
    st.write(predicted_class_name)


if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
