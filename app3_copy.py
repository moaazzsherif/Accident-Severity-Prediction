import streamlit as st
import pandas as pd
import joblib

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Accident Severity Prediction",
    layout="wide",
    page_icon="🚦"
)

# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model():
    return joblib.load(r"D:\nti crativa\test\models3_copy\best_model_Bagging.pkl")

model = load_model()

# ===============================
# Sidebar
# ===============================
st.sidebar.title("🚗 Accident Severity App")
page = st.sidebar.radio(
    "Navigation",
    ["Project Overview", "Model Performance", "Prediction", "Deployment Report"]
)

# ===============================
# 1️⃣ Project Overview
# ===============================
if page == "Project Overview":
    st.title("🚦 Accident Severity Prediction")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Size", "270,941")
    col2.metric("Training Records", "216,752")
    col3.metric("Test Records", "54,189")

    st.markdown("### 🎯 Problem Definition")
    st.write("""
    - **Target Variable:** Accident_Severity  
    - **Problem Type:** Multi-class Classification  
    - **Best Model:** Bagging  
    """)

    st.markdown("### 🧠 Selected Features (20)")
    features = [
        "1st_Road_Class", "Avg_Casualty_Age", "Avg_Casualty_Severity",
        "Avg_Driver_Age", "Day_of_Week", "Junction_Control",
        "Junction_Detail", "Light_Conditions", "Male_Casualties",
        "Male_Drivers", "Number_of_Casualties", "Number_of_Vehicles",
        "Pedestrian_Crossing-Physical_Facilities", "Police_Force",
        "Road_Type", "Speed_limit", "Total_Casualties",
        "Total_Vehicles", "Urban_or_Rural_Area", "Weather_Conditions"
    ]

    st.write(pd.DataFrame(features, columns=["Features"]))

# ===============================
# 2️⃣ Model Performance
# ===============================
elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.markdown("---")

    col1, col2 = st.columns(2)

    col1.metric("Best Model", "Bagging")
    col2.metric("Test Accuracy", "96.95%")

    st.success("✔ Model shows excellent generalization with high accuracy on unseen data.")

# ===============================
# 3️⃣ Prediction Page
# ===============================
elif page == "Prediction":
    st.title("🔮 Accident Severity Prediction")
    st.markdown("---")

    st.info("Enter feature values to predict accident severity")

    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            speed_limit = st.number_input("Speed Limit", 0, 150, 50)
            num_vehicles = st.number_input("Number of Vehicles", 1, 10, 1)
            num_casualties = st.number_input("Number of Casualties", 0, 20, 0)

        with col2:
            avg_driver_age = st.number_input("Avg Driver Age", 16, 90, 35)
            avg_casualty_age = st.number_input("Avg Casualty Age", 0, 100, 30)
            day_of_week = st.selectbox("Day of Week", list(range(1, 8)))

        with col3:
            weather = st.selectbox("Weather Conditions", [0, 1, 2, 3, 4, 5])
            light = st.selectbox("Light Conditions", [0, 1, 2, 3])
            urban = st.selectbox("Urban or Rural", [0, 1])

        submit = st.form_submit_button("Predict")

    if submit:
        input_data = pd.DataFrame([{
            "1st_Road_Class": 1,
            "Avg_Casualty_Age": avg_casualty_age,
            "Avg_Casualty_Severity": 2,
            "Avg_Driver_Age": avg_driver_age,
            "Day_of_Week": day_of_week,
            "Junction_Control": 1,
            "Junction_Detail": 1,
            "Light_Conditions": light,
            "Male_Casualties": 1,
            "Male_Drivers": 1,
            "Number_of_Casualties": num_casualties,
            "Number_of_Vehicles": num_vehicles,
            "Pedestrian_Crossing-Physical_Facilities": 0,
            "Police_Force": 1,
            "Road_Type": 1,
            "Speed_limit": speed_limit,
            "Total_Casualties": num_casualties,
            "Total_Vehicles": num_vehicles,
            "Urban_or_Rural_Area": urban,
            "Weather_Conditions": weather
        }])

        prediction = model.predict(input_data)[0]

        st.success(f"🚨 Predicted Accident Severity: **{prediction}**")

# ===============================
# 4️⃣ Deployment Report
# ===============================
elif page == "Deployment Report":
    st.title("📄 Deployment Report")
    st.markdown("---")

    try:
        with open(r"D:\nti crativa\test\models3_copy\deployment_report.txt", "r", encoding="utf-8") as f:
            report = f.read()
        st.text_area("Report Content", report, height=500)
    except:
        st.error("Deployment report file not found.")