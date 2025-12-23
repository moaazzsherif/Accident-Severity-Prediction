🚦 Road Traffic Accident Severity Prediction
End-to-End Machine Learning Project
📌 Project Overview

This project focuses on analyzing road traffic accident data to understand the key factors that influence accident severity and to build robust machine learning classification models capable of predicting accident severity levels.

Target Variable: Accident_Severity

Problem Type: Multi-class Classification

Domain: Transportation Safety & Risk Analysis

The project follows a full data science workflow, from data understanding and preprocessing to model training, evaluation, and interpretation.

🎯 Project Objective

The main objectives of this project are:

Identify the most influential factors contributing to accident severity

Build and compare multiple classification models

Handle class imbalance effectively

Reduce feature dimensionality while maintaining performance

Prepare the model for potential deployment in real-world scenarios

📊 Dataset Description

The dataset represents real-world road traffic accident records and is highly detailed.

Each row represents a casualty involved in an accident

A single accident may appear multiple times if it involved multiple casualties

The data combines multiple dimensions:

Accident-level details

Road and environmental conditions

Vehicle characteristics

Driver information

Casualty attributes

Spatial and temporal data

This richness makes the dataset realistic, challenging, and suitable for production-grade ML solutions.

📍 Spatial (Location-Based) Features

These features describe where the accident occurred:

Latitude, Longitude

Location_Easting_OSGR, Location_Northing_OSGR

Local_Authority_(District)

Local_Authority_(Highway)

Urban_or_Rural_Area

LSOA_Frequency (accident frequency within the same area)

Goal:
Identify geographical risk patterns and accident-prone locations.

⏰ Temporal (Time-Based) Features

These features describe when the accident occurred:

Hour, Day, Month, Year

Day_of_Week

Goal:
Capture temporal trends such as rush hours, night-time accidents, and seasonal effects.

🚨 Accident-Level Features

These features describe the overall accident context:

Accident_Severity (Target)

Number_of_Vehicles

Number_of_Casualties

Police_Force

Did_Police_Officer_Attend_Scene_of_Accident

⚠️ Important:
Outcome-related variables are carefully reviewed to avoid data leakage.

🛣️ Road & Environmental Conditions

These features describe the road infrastructure and environment:

Road Characteristics

Road_Type, Speed_limit

1st_Road_Class, 2nd_Road_Class

Junction_Control, Junction_Detail, Junction_Location

Environmental Conditions

Light_Conditions

Weather_Conditions

Road_Surface_Conditions

Special_Conditions_at_Site

Carriageway_Hazards

Goal:
Analyze how infrastructure, lighting, weather, and surface conditions affect severity.

🚶 Pedestrian-Related Features

These features capture pedestrian involvement:

Pedestrian_Crossing-Human_Control

Pedestrian_Crossing-Physical_Facilities

Pedestrian_Location

Pedestrian_Movement

Pedestrian_Road_Maintenance_Worker

Goal:
Assess pedestrian-related risk factors.

👤 Casualty Information

These features describe injured individuals:

Casualty_Class, Casualty_Type

Sex_of_Casualty, Age_of_Casualty

Age_Band_of_Casualty

Casualty_Home_Area_Type

Casualty_IMD_Decile

⚠️ Leakage Warning:
Direct injury outcome features (e.g. Casualty_Severity) are excluded or handled carefully.

🚗 Vehicle Characteristics

These features describe vehicle behavior and impact:

Vehicle_Type, Vehicle_Manoeuvre

Vehicle_Leaving_Carriageway

Skidding_and_Overturning

1st_Point_of_Impact

Hit_Object_in_Carriageway

Hit_Object_off_Carriageway

👨‍✈️ Driver Information

These features describe driver demographics and vehicle background:

Sex_of_Driver, Age_of_Driver

Journey_Purpose_of_Driver

Engine_Capacity_(CC)

Age_of_Vehicle

Propulsion_Code

Driver_IMD_Decile, Vehicle_IMD_Decile

🧠 Modeling Strategy

Key modeling considerations:

Many integer features are categorical, not continuous

Class imbalance is significant and handled using:

Class weights

Robust evaluation metrics

Tree-based models are best suited for this dataset:

CatBoost

LightGBM

XGBoost

ExtraTrees / RandomForest

Feature importance analysis is used to:

Reduce dimensionality

Improve generalization

Prepare the model for deployment

🔄 Machine Learning Workflow

Data understanding & exploration

Data cleaning & preprocessing

Feature engineering & selection

Class imbalance handling

Model training & comparison

Performance evaluation

Feature importance & interpretation

✅ Final Notes

This project demonstrates a realistic, production-oriented machine learning pipeline applied to road safety analysis.
It emphasizes data quality, leakage prevention, model robustness, and interpretability.

📌 Suitable for:

Portfolio projects

Academic research

Real-world ML deployment scenarios
