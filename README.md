# 🚦 Bengaluru 3D Traffic Accident Risk Intelligence Dashboard

🌐 **Live App:** https://bengaluru-3d-risk-dashboard-wraebvbapdee5wwmppvbzh.streamlit.app/ 

An AI-powered spatial intelligence dashboard that predicts traffic accident severity in Bengaluru and visualizes urban risk hotspots in immersive 3D.

---

## 🔥 Why This Project Matters

Urban traffic risk is influenced by time, weather, infrastructure, and vehicle type.  
This dashboard combines **Machine Learning + Geospatial Visualization** to simulate how accident severity changes across key Bengaluru zones.

It transforms raw conditions into:
- 🎯 Predictive severity analysis
- 📊 Confidence scoring
- 🧠 Risk explanation insights
- 🗺 Interactive 3D spatial risk mapping

---

## 🚀 Key Features

✔ Machine Learning-based severity prediction  
✔ Random Forest classification model  
✔ Encoded categorical feature engineering  
✔ Real-time confidence meter  
✔ AI-driven risk explanation layer  
✔ Interactive 3D extruded heatmap (PyDeck)  
✔ Cinematic dashboard UI with animated risk feedback  
✔ Fully deployed production app  

---

## 🧠 Machine Learning Pipeline

**Input Features**
- Area
- Weather
- Road Condition
- Vehicle Type
- Hour of Day

**Processing**
- Label Encoding for categorical variables
- Random Forest Classifier (100 estimators)
- Probability-based confidence scoring

**Output**
- Slight Injury
- Grievous Injury
- Fatal

---

## 🗺 3D Spatial Intelligence Layer

The dashboard visualizes risk intensity using:

- Extruded column layers
- Dynamic color gradients (Green → Red)
- Real-time elevation spikes based on user selection
- Tilted 3D perspective for urban risk simulation

This mimics real-world spatial risk concentration analysis used in smart city systems.

---

## 🛠 Tech Stack

- Python
- Streamlit
- Scikit-learn
- PyDeck (WebGL-based visualization)
- Pandas & NumPy
- Joblib (model persistence)

---

## 📂 Project Structure
 bengaluru-3d-risk-dashboard/
    ├── app.py
    ├── requirements.txt
    ├── accident_model.pkl
    ├── encoder_*.pkl
    ├── bengaluru_accidents_synthetic.csv
