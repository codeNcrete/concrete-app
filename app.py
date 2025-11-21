import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import kagglehub
import os
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
# Sets the browser tab title, icon, and enables "Wide Mode" for better chart visibility
st.set_page_config(
    page_title="Concrete AI Lab",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR UI POLISH ---
# This injects simple CSS to style buttons and containers
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        font-size: 20px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
    .metric-container {
        padding: 20px;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    h1 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER SECTION ---
col1, col2 = st.columns([1, 5])
with col1:
    # Display a concrete mixer icon
    st.image("https://img.icons8.com/color/96/000000/concrete-mixer.png", width=80)
with col2:
    st.title("Concrete Strength Predictor AI")
    st.markdown("### Virtual Lab: Design your mix & predict strength instantly.")

# --- 1. LOAD & TRAIN MODEL (Cached) ---
# The @st.cache_resource decorator prevents the app from retraining the model
# every time you change a slider. It runs once and saves the model in memory.
@st.cache_resource
def train_model():
    with st.spinner('🤖 Initializing AI Model... (Downloading Dataset & Training)'):
        try:
            # Download dataset using KaggleHub
            path = kagglehub.dataset_download("niteshyadav3103/concrete-compressive-strength")
            csv_file_path = None
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".csv"):
                        csv_file_path = os.path.join(root, file)
                        break
            
            if not csv_file_path:
                st.error("Could not find dataset file.")
                return None, None

            # Load Data
            df = pd.read_csv(csv_file_path)
            
            # X = Ingredients, y = Strength
            X = df.drop(df.columns[-1], axis=1)
            y = df[df.columns[-1]]
            
            # Train Model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Quick Evaluation
            y_pred = model.predict(X_test)
            accuracy = r2_score(y_test, y_pred)
            
            return model, accuracy
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None

model, accuracy = train_model()

if model:
    # --- 2. SIDEBAR INPUTS (Grouped for Better UX) ---
    st.sidebar.header("🧪 Mix Design Controls")
    st.sidebar.success(f"Model Accuracy: **{accuracy*100:.1f}%**")
    
    # Group 1: Binders
    with st.sidebar.expander("🧱 Binders (Cementitious)", expanded=True):
        cement = st.number_input("Cement (kg/m³)", 0.0, 1000.0, 350.0, step=10.0, help="Primary binder. More cement usually means higher strength.")
        slag = st.number_input("Blast Furnace Slag", 0.0, 500.0, 0.0, step=10.0, help="By-product additive to improve durability.")
        ash = st.number_input("Fly Ash", 0.0, 500.0, 0.0, step=10.0, help="Coal combustion by-product, improves workability.")

    # Group 2: Aggregates
    with st.sidebar.expander("🪨 Aggregates (Rocks/Sand)", expanded=True):
        coarse = st.number_input("Coarse Aggregate", 0.0, 1500.0, 950.0, step=10.0, help="Gravel or crushed stone.")
        fine = st.number_input("Fine Aggregate", 0.0, 1000.0, 750.0, step=10.0, help="Sand.")

    # Group 3: Fluids
    with st.sidebar.expander("💧 Fluids & Additives", expanded=True):
        water = st.number_input("Water", 0.0, 300.0, 180.0, step=5.0, help="Less water generally increases strength (lower w/c ratio).")
        super_p = st.number_input("Superplasticizer", 0.0, 50.0, 0.0, step=0.5, help="Allows reducing water while maintaining flow.")

    # Group 4: Time
    with st.sidebar.expander("⏳ Curing", expanded=True):
        age = st.slider("Age (Days)", 1, 365, 28, help="How long the concrete hardens. Standard test is 28 days.")

    # --- 3. MAIN DASHBOARD AREA ---
    
    # Prepare Data for Prediction & Charts
    input_dict = {
        'Cement': cement, 'Slag': slag, 'Ash': ash, 
        'Water': water, 'Superplasticizer': super_p, 
        'Coarse Agg': coarse, 'Fine Agg': fine
    }
    # Convert inputs to the list format required by the model
    features_list = [[cement, slag, ash, water, super_p, coarse, fine, age]]
    
    # Section: Visualizing the Mix
    st.subheader("📊 Current Mix Composition")
    col_chart, col_pred_btn = st.columns([2, 1])

    with col_chart:
        # Interactive Donut Chart using Plotly
        chart_df = pd.DataFrame(list(input_dict.items()), columns=['Ingredient', 'Amount'])
        # Filter out zero values to keep chart clean
        chart_df = chart_df[chart_df['Amount'] > 0]
        
        fig = px.pie(chart_df, values='Amount', names='Ingredient', hole=0.4, 
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_pred_btn:
        st.markdown("### Ready to Test?")
        st.markdown("Click below to simulate a compression test on this specific mix.")
        st.write("") # Spacer
        predict_btn = st.button("🚀 RUN TEST")

    # Section: Prediction Results
    if predict_btn:
        st.markdown("---")
        st.subheader("🧪 Test Results")
        
        with st.spinner('Calculating chemical interactions...'):
            prediction = model.predict(features_list)[0]
            
            # Layout: Gauge Chart on Left, Text Analysis on Right
            col_res1, col_res2 = st.columns([1, 1])
            
            with col_res1:
                # Gauge Chart for Visual Impact
                gauge_fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Compressive Strength (MPa)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 20], 'color': "#ffcccb"},  # Low (Red-ish)
                            {'range': [20, 40], 'color': "#ffffcc"}, # Medium (Yellow-ish)
                            {'range': [40, 100], 'color': "#90ee90"} # High (Green-ish)
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': prediction
                        }
                    }
                ))
                gauge_fig.update_layout(height=350, margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col_res2:
                # Text Analysis
                st.markdown(f"<div class='metric-container'><h2>{prediction:.2f} MPa</h2><p>Estimated Strength</p></div>", unsafe_allow_html=True)
                
                if prediction < 20:
                    st.error("⚠️ **Low Strength**\n\nSuitable for non-structural uses like garden paths, blinding layers, or temporary walls. Not safe for building foundations.")
                elif prediction < 40:
                    st.warning("🏠 **Standard Strength**\n\nExcellent for residential construction: driveways, slabs, footings, and sidewalks.")
                elif prediction < 60:
                    st.success("🏢 **High Strength**\n\nSuitable for heavy infrastructure: high-rise columns, bridge decks, and pre-stressed concrete.")
                else:
                    st.balloons()
                    st.success("💎 **Ultra-High Strength**\n\nSpecialized engineering use cases. This is exceptionally strong concrete!")

else:
    # Fallback if model fails to load
    st.info("Please wait while the AI is being trained in the background...")