import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load artifacts
# -----------------------------
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

explainer = shap.TreeExplainer(model)

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 Nigerian House Price Prediction System")
st.write("Predict house prices across Nigerian locations and understand what drives the estimate.")

# -----------------------------
# Feature explanations
# -----------------------------
def clean_feature_name(feature):
    if feature.startswith("title_"):
        return f"Property Type: {feature.replace('title_', '')}"
    elif feature.startswith("town_"):
        return f"Town: {feature.replace('town_', '')}"
    elif feature.startswith("state_"):
        return f"State: {feature.replace('state_', '')}"
    else:
        return feature.replace("_", " ").title()

# -----------------------------
# Extract dynamic categories
# -----------------------------
towns = sorted([
    col.replace("town_", "")
    for col in columns if col.startswith("town_")
])

states = sorted([
    col.replace("state_", "")
    for col in columns if col.startswith("state_")
])

property_types = [
    "Detached Duplex",
    "Semi Detached Duplex",
    "Terraced Duplexes",
    "Detached Bungalow"
]

# -----------------------------
# User Inputs
# -----------------------------
st.header("Property Details")

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 10, 3)
    toilets = st.number_input("Toilets", 1, 10, 3)

with col2:
    parking_space = st.number_input("Parking Space", 0, 10, 2)
    property_type = st.selectbox("Property Type", property_types)
    state = st.selectbox("State", states)

town = st.selectbox("Town", towns)

# -----------------------------
# Build input
# -----------------------------
input_dict = {col: 0 for col in columns}

input_dict["bedrooms"] = bedrooms
input_dict["bathrooms"] = bathrooms
input_dict["toilets"] = toilets
input_dict["parking_space"] = parking_space

title_col = f"title_{property_type}"
town_col = f"town_{town}"
state_col = f"state_{state}"

if title_col in input_dict:
    input_dict[title_col] = 1

if town_col in input_dict:
    input_dict[town_col] = 1

if state_col in input_dict:
    input_dict[state_col] = 1

input_df = pd.DataFrame([input_dict])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):

    prediction = model.predict(input_df)[0]

    st.subheader("💰 Estimated Price")
    st.write(f"₦{prediction:,.0f}")

    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    st.subheader("🔍 Why this price?")

    shap_values = explainer.shap_values(input_df, check_additivity=False)

    shap_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Impact": shap_values[0]
    })

    shap_df["abs_impact"] = shap_df["Impact"].abs()


    # Keep only meaningful features
    def is_meaningful(feature, value):
        if feature.startswith(("town_", "state_", "title_")):
            return value == 1  # only selected category
        return True  # keep numeric features


    filtered = shap_df[
        shap_df.apply(lambda row: is_meaningful(row["Feature"], input_df[row["Feature"]].values[0]), axis=1)
    ]

    top_features = filtered.sort_values("abs_impact", ascending=False).head(5)

    # -----------------------------
    # STORY-STYLE EXPLANATION
    # -----------------------------
    st.subheader("🧠 Model Insight")

    # Separate positive drivers
    positive = top_features[top_features["Impact"] > 0]

    reasons = []

    for _, row in positive.iterrows():
        feature = row["Feature"]
        value = input_df[feature].values[0]

        if feature.startswith("town_"):
            reasons.append(f"it is located in {feature.replace('town_', '')}")
        elif feature.startswith("state_"):
            reasons.append(f"{feature.replace('state_', '')}")
        elif feature.startswith("title_"):
            reasons.append(f"it is a {feature.replace('title_', '')}")
        elif feature == "bedrooms":
            reasons.append(f"it has {int(value)} bedrooms")
        elif feature == "bathrooms":
            reasons.append(f"{int(value)} bathrooms")
        elif feature == "toilets":
            reasons.append(f"{int(value)} toilets")
        elif feature == "parking_space":
            reasons.append(f"{int(value)} parking spaces")

    # Combine nicely
    if reasons:
        explanation = "This property is priced this way mainly because "

        if len(reasons) == 1:
            explanation += reasons[0] + "."
        else:
            explanation += ", ".join(reasons[:-1]) + " and " + reasons[-1] + "."

        st.success(explanation)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots()

    ax.barh(
        [clean_feature_name(f) for f in top_features["Feature"]],
        top_features["Impact"]
    )

    ax.set_title("Top Factors Influencing Price")

    st.pyplot(fig)

    # -----------------------------
    # Human Explanation
    # -----------------------------
    st.subheader("📊 Key Factors Driving This Price")

    positive = top_features[top_features["Impact"] > 0]
    # negative = top_features[top_features["Impact"] < 0]

    if positive.empty:
        st.info("The model did not identify strong positive drivers for this prediction.")

    if not positive.empty:
        st.write("💹 Factors increasing the price:")
        for _, row in positive.iterrows():
            feature = row["Feature"]
            readable = clean_feature_name(feature)
            value = input_df.get(feature, "N/A")
            value = value.values[0] if hasattr(value, "values") else value

            st.write(f"- {readable} ({value}) increased the price")

        # if not negative.empty:
        #     st.write("📉 Factors reducing the price:")
        #     for _, row in negative.iterrows():
        #         feature = row["Feature"]
        #         readable = clean_feature_name(feature)
        #
        #         value = input_df.get(feature, "N/A")
        #         value = value.values[0] if hasattr(value, "values") else value
        #
        #         st.write(f"- {readable} ({value}) reduced the price")