
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="EcoSense AI",
    page_icon="🌿",
    layout="wide"
)

# ---------------- PAGE HEADER ----------------

st.title("🌿 EcoSense AI")
st.subheader("Smart Air Quality Anomaly Detection")

st.write(
    "An experimental Machine Learning application that identifies "
    "unusual patterns in air-quality sensor measurements."
)

# ---------------- LOAD MODEL AND DATA ----------------

@st.cache_resource
def load_model():
    return joblib.load("anomaly_model.joblib")

@st.cache_data
def load_features():
    return joblib.load("feature_names.joblib")

@st.cache_data
def load_dataset():
    return pd.read_csv("cleaned_air_quality.csv")

model = load_model()
feature_names = load_features()
dataset = load_dataset()

# Ensure dataset has the same feature order as the model
dataset = dataset[feature_names]

# ---------------- SIDEBAR INPUTS ----------------

st.sidebar.header("Sensor Measurements")
st.sidebar.caption("Use a sample reading or enter values manually.")

if "sample_row" not in st.session_state:
    st.session_state.sample_row = 0

if st.sidebar.button("Use Sample Reading"):
    st.session_state.sample_row = (
        st.session_state.sample_row + 1
    ) % len(dataset)

sample = dataset.iloc[st.session_state.sample_row]

with st.sidebar.form("sensor_form"):
    values = {}

    for feature in feature_names:
        values[feature] = st.number_input(
            feature,
            value=float(sample[feature]),
            format="%.4f",
            key=f"input_{feature}"
        )

    submitted = st.form_submit_button(
        "Analyze Air Quality",
        type="primary"
    )

# ---------------- PREDICTION ----------------

if submitted:
    input_data = pd.DataFrame(
        [values],
        columns=feature_names
    )

    prediction = model.predict(input_data)[0]
    anomaly_score = model.decision_function(input_data)[0]

    st.divider()
    st.header("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == -1:
            st.warning("Unusual Pattern Detected")
        else:
            st.success("No Unusual Pattern Detected")

    with col2:
        st.metric(
            "Model Anomaly Score",
            f"{anomaly_score:.4f}"
        )

    with col3:
        st.metric(
            "Input Features",
            len(feature_names)
        )

    st.caption(
        "A negative decision score indicates an anomaly under this "
        "model's decision rule; a non-negative score indicates a "
        "normal prediction."
    )

    if prediction == -1:
        st.write(
            "The model considers this sensor pattern unusual compared "
            "with patterns learned during training. Review the readings "
            "and sensor quality before drawing conclusions."
        )
    else:
        st.write(
            "The model did not flag this pattern as unusual. "
            "This does not guarantee that the air is safe."
        )

    # Show the entered measurements
    st.subheader("Your Sensor Readings")

    readings_df = pd.DataFrame({
        "Sensor": feature_names,
        "Reading": [values[name] for name in feature_names]
    })

    st.dataframe(readings_df, use_container_width=True)

    # Show a CO reading chart when the column is available
    if "CO(GT)" in feature_names:
        st.subheader("CO Reading Compared with Dataset")

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.hist(
            dataset["CO(GT)"],
            bins=30,
            edgecolor="black"
        )

        ax.axvline(
            values["CO(GT)"],
            linestyle="--",
            linewidth=2,
            label="Your input"
        )

        ax.set_xlabel("CO(GT) sensor reading")
        ax.set_ylabel("Number of observations")
        ax.set_title("Input Reading vs Dataset Distribution")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

# ---------------- DATASET OVERVIEW ----------------

st.divider()
st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Observations", f"{len(dataset):,}")

with col2:
    st.metric("Sensor Features", len(feature_names))

with col3:
    st.metric("Missing Values", int(dataset.isna().sum().sum()))

with st.expander("Preview cleaned dataset"):
    st.dataframe(dataset.head(10), use_container_width=True)

st.caption(
    "EcoSense AI is an educational anomaly-detection prototype. "
    "It is not a certified pollution monitor and does not determine "
    "whether air is safe for human health."
)
