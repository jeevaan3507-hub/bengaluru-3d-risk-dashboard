# ========================
# 1. Imports & Configuration
# ========================
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import pydeck as pdk
import time  # Imported for the animation delay

# ML Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# FIX 3: Set Random Seed for Consistency
np.random.seed(42)

# ========================
# 2. Page Config
# ========================
st.set_page_config(
    page_title="Bengaluru 3D Risk Predictor",
    page_icon="🚦",
    layout="wide"
)

# ========================
# 3. Premium 3D Background & UI Styling
# ========================
st.markdown(
    """
    <style>
    /* Deep 3D Gradient Background */
    .stApp {
        background: radial-gradient(circle at top left, #1a2a3a, #050a0f 80%);
        background-attachment: fixed;
        color: #e0e0e0;
    }

    /* FIX 6: Subtle Particle Background */
    .stApp::before {
        content: "";
        position: fixed;
        top:0; left:0; width:100%; height:100%;
        background: radial-gradient(circle, rgba(255,255,255,0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        animation: moveBackground 60s linear infinite;
        z-index: -1;
    }

    @keyframes moveBackground {
        from { background-position: 0 0; }
        to { background-position: 1000px 1000px; }
    }

    /* Glassmorphism Container for Inputs */
    [data-testid="stSidebar"] {
        background: rgba(15, 25, 35, 0.85);
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 5px 0 20px rgba(0,0,0,0.5);
    }

    /* Floating Card Effect for Main Content */
    .block-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 2rem;
    }

    /* 3D Animated Button */
    .stButton>button {
        border-radius: 12px;
        background: linear-gradient(145deg, #ff512f, #dd2476);
        color: white;
        border: none;
        box-shadow: 0 10px 20px rgba(221, 36, 118, 0.4);
        transition: all 0.3s ease;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(221, 36, 118, 0.6);
    }

    .stButton>button:active {
        transform: translateY(1px);
    }

    h1 {
        text-shadow: 0 4px 10px rgba(0,0,0,0.8);
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    label, .stMarkdown {
        color: #cccccc !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========================
# 4. Data & Model Management
# ========================

AREA_LIST = ["Silk Board", "Whitefield", "Indiranagar", "Electronic City",
             "Marathahalli", "MG Road", "JP Nagar", "Bellandur"]
WEATHER_LIST = ["Sunny", "Rain", "Cloudy", "Fog"]
ROAD_LIST = ["Good", "Potholes", "Under Construction", "Wet"]
VEHICLE_LIST = ["Car", "Two-Wheeler", "Auto", "Bus", "Truck"]
SEVERITY_LIST = ["Slight Injury", "Grievous Injury", "Fatal"]

# Coordinates for Mapping
AREA_COORDS = {
    "Silk Board": (12.9177, 77.6233),
    "Whitefield": (12.9698, 77.7500),
    "Indiranagar": (12.9719, 77.6412),
    "Electronic City": (12.8452, 77.6602),
    "Marathahalli": (12.9592, 77.6974),
    "MG Road": (12.9742, 77.6033),
    "JP Nagar": (12.9077, 77.5859),
    "Bellandur": (12.9304, 77.6784)
}


def create_dummy_models():
    """Generates synthetic models and OVERWRITES existing files."""
    # 1. Generate Synthetic Data
    n_samples = 1000
    data = {
        "Hour": np.random.randint(0, 24, n_samples),
        "Area": np.random.choice(AREA_LIST, n_samples),
        "Weather": np.random.choice(WEATHER_LIST, n_samples),
        "Road": np.random.choice(ROAD_LIST, n_samples),
        "Vehicle": np.random.choice(VEHICLE_LIST, n_samples),
        "Severity": np.random.choice(SEVERITY_LIST, n_samples, p=[0.6, 0.3, 0.1])
    }
    df = pd.DataFrame(data)

    # 2. Encode Data
    le_area = LabelEncoder().fit(AREA_LIST)
    le_weather = LabelEncoder().fit(WEATHER_LIST)
    le_road = LabelEncoder().fit(ROAD_LIST)
    le_vehicle = LabelEncoder().fit(VEHICLE_LIST)
    le_severity = LabelEncoder().fit(SEVERITY_LIST)

    X = df[['Hour', 'Area', 'Weather', 'Road', 'Vehicle']].apply(
        lambda col: le_area.transform(col) if col.name == 'Area' else
        le_weather.transform(col) if col.name == 'Weather' else
        le_road.transform(col) if col.name == 'Road' else
        le_vehicle.transform(col) if col.name == 'Vehicle' else col
    )
    y = le_severity.transform(df['Severity'])

    # 3. Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 4. Save Models (This overwrites your old files)
    joblib.dump(model, "accident_model.pkl")
    joblib.dump(le_area, "encoder_area.pkl")
    joblib.dump(le_weather, "encoder_weather.pkl")
    joblib.dump(le_road, "encoder_road.pkl")
    joblib.dump(le_vehicle, "encoder_vehicle.pkl")
    joblib.dump(le_severity, "encoder_severity.pkl")

    # 5. Create Dummy Map Data
    map_data = []
    for area, coords in AREA_COORDS.items():
        for _ in range(15):
            map_data.append({
                "Area": area,
                "Lat": coords[0] + np.random.uniform(-0.005, 0.005),
                "Lon": coords[1] + np.random.uniform(-0.005, 0.005)
            })
    pd.DataFrame(map_data).to_csv("bengaluru_accidents_synthetic.csv", index=False)
    return True


# FIX: Force model generation to ensure UI and Model match
if 'model_generated' not in st.session_state:
    with st.spinner("🔄 Initializing AI Core..."):
        create_dummy_models()
        st.session_state['model_generated'] = True

# Load the freshly updated models
model = joblib.load("accident_model.pkl")
le_area = joblib.load("encoder_area.pkl")
le_weather = joblib.load("encoder_weather.pkl")
le_road = joblib.load("encoder_road.pkl")
le_vehicle = joblib.load("encoder_vehicle.pkl")
le_severity = joblib.load("encoder_severity.pkl")

# ========================
# 5. Title & Description
# ========================
st.title("🚦 Bengaluru Traffic Accident Severity Predictor")
# FIX 7: Product Demo Text
st.markdown(
    """
    <div style='font-size:18px; opacity:0.8; margin-bottom: 20px;'>
    🚀 Powered by Machine Learning & 3D Geospatial Intelligence
    </div>
    """,
    unsafe_allow_html=True
)
st.write("Predict accident severity based on environmental conditions and visualize risk hotspots in 3D.")

# ========================
# 6. Sidebar Inputs
# ========================
st.sidebar.header("⚙️ Input Conditions")

area = st.sidebar.selectbox("Area", AREA_LIST)
weather = st.sidebar.selectbox("Weather", WEATHER_LIST)
road = st.sidebar.selectbox("Road Condition", ROAD_LIST)
vehicle = st.sidebar.selectbox("Vehicle Type", VEHICLE_LIST)
hour = st.sidebar.slider("Hour of Day (0–23)", 0, 23, 18)


# ========================
# 7. Explainability Function
# ========================
def explain_risk(area, weather, road, vehicle, hour):
    reasons = []
    if hour >= 18 or hour <= 6:
        reasons.append("🌙 Time Risk: Night/Evening hours reduce visibility and increase driver fatigue.")
    if weather in ["Rain", "Cloudy", "Fog"]:
        reasons.append("🌧 Weather Risk: Adverse conditions increase braking distance significantly.")
    if road in ["Potholes", "Under Construction", "Wet"]:
        reasons.append("🛣 Road Risk: Surface anomalies heighten chances of loss of control.")
    if vehicle in ["Two-Wheeler", "Auto"]:
        reasons.append("🏍 Vehicle Risk: Vulnerable road users face higher injury severity.")
    if area in ["Silk Board", "Marathahalli", "Bellandur", "Whitefield", "JP Nagar"]:
        reasons.append("📍 Location Risk: High-density zone with complex intersections.")
    if not reasons:
        reasons.append("✅ Standard risk profile based on combined factors.")
    return reasons


# ========================
# 8. Prediction Logic & UI
# ========================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📊 Prediction Analysis")

    if st.button("🔮 Predict Severity"):
        # --- 1. Model Inference ---
        X_input = np.array([[
            hour,
            le_area.transform([area])[0],
            le_weather.transform([weather])[0],
            le_road.transform([road])[0],
            le_vehicle.transform([vehicle])[0]
        ]])

        pred_encoded = model.predict(X_input)[0]
        pred_label = le_severity.inverse_transform([pred_encoded])[0]
        probs = model.predict_proba(X_input)[0]
        confidence = np.max(probs) * 100

        # --- 2. Animated Risk Meter (FIX 1) ---
        st.markdown("### 🎯 Risk Confidence Level")
        progress = st.progress(0)
        for i in range(int(confidence)):
            progress.progress(i + 1)
            # Tiny delay for visual effect (optional but nice)
            time.sleep(0.005)

            # --- 3. Dynamic Glow Result Box (FIX 3) ---
        if pred_label == "Fatal":
            box_color = "#ff4b4b"
            icon = "🚨"
        elif pred_label == "Grievous Injury":
            box_color = "#ffa500"
            icon = "⚠️"
        else:
            box_color = "#21c354"
            icon = "✅"

        result_html = f"""
        <div style="
            padding: 25px;
            border-radius: 18px;
            background: rgba(255,255,255,0.05);
            box-shadow: 0 0 40px {box_color};
            border: 1px solid {box_color};
            color: white;
        ">
            <h2 style="margin:0;">{icon} {pred_label}</h2>
            <p style="color: #cccccc; font-size: 1.1em;">Confidence: {confidence:.2f}%</p>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)
        st.markdown(" ")

        # --- 4. AI Thinking Effect (FIX 4) ---
        with st.expander("🧠 AI Risk Explanation"):
            with st.spinner("Analyzing contributing factors..."):
                explanations = explain_risk(area, weather, road, vehicle, hour)
            for reason in explanations:
                st.markdown(f"- {reason}")

with col2:
    # ========================
    # 9. 3D Heatmap Section (PyDeck)
    # ========================
    st.subheader("🗺 3D Risk Visualization")

    try:
        df_map = pd.read_csv("bengaluru_accidents_synthetic.csv")
    except:
        df_map = pd.DataFrame(columns=["Area", "Lat", "Lon"])

    # Prepare Visualization Data
    map_data = []
    for area_name, (lat, lon) in AREA_COORDS.items():
        # Count accidents for this area
        count = len(df_map[df_map["Area"] == area_name]) if "Area" in df_map.columns else 10
        map_data.append({"lat": lat, "lon": lon, "count": count, "area": area_name})

    df_vis = pd.DataFrame(map_data)

    # --- 5. Live Risk Spike (FIX 5) ---
    # Increase the height of the selected area to show immediate reaction
    df_vis.loc[df_vis["area"] == area, "count"] += 15

    max_count = df_vis['count'].max() if df_vis['count'].max() > 0 else 1


    def get_color_ramp(count):
        ratio = count / max_count
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        b = 0
        return [r, g, b]


    df_vis['rgb_color'] = df_vis['count'].apply(get_color_ramp)

    # Define 3D Layer
    layer = pdk.Layer(
        "ColumnLayer",
        data=df_vis,
        get_position=["lon", "lat"],
        get_elevation="count",
        elevation_scale=100,
        radius=600,
        pickable=True,
        extruded=True,
        get_fill_color="rgb_color",
    )

    # --- 2. Cinematic View State (FIX 2) ---
    view_state = pdk.ViewState(
        latitude=12.9716,
        longitude=77.5946,
        zoom=10.5,
        pitch=60,  # Increased tilt for 3D effect
        bearing=20  # Slight rotation for cinematic feel
    )

    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={
        "html": "<b>Area:</b> {area}<br/><b>Risk Score:</b> {count}",
        "style": {"backgroundColor": "rgba(0,0,0,0.8)", "color": "white"}
    })

    st.pydeck_chart(r)

st.markdown("---")
st.caption("Built by Jeevan | Bengaluru Accident Severity Prediction | Advanced 3D AI Dashboard")
