import streamlit as st
import requests
import json
import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import math
import hashlib
from dataclasses import dataclass
import openai

# Page configuration
st.set_page_config(
    page_title="MediAI Pro - Advanced Health Assistant for India",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'user_location': None,
        'coordinates': None,
        'health_data': {},
        'location_set': False,
        'selected_analysis_type': "Comprehensive Health Analysis",
        'openai_api_key': None,
        'analysis_results': {},
        'active_tab': 0,
        'form_submitted': False,
        'weather_data': None,
        'nearby_hospitals': [],
        'nearby_pharmacies': [],
        'health_news': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
init_session_state()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Classes
@dataclass
class PatientData:
    name: str
    age: int
    gender: str
    weight: float
    height: float
    blood_group: str

@dataclass
class VitalSigns:
    bp_systolic: int
    bp_diastolic: int
    pulse: int
    temperature: float
    spo2: int
    respiratory_rate: int

# Simplified CSS for better compatibility
def load_custom_css():
    st.markdown("""
    <style>
        /* Main theme colors */
        .main-header {
            color: #007C91;
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .sub-header {
            color: #007C91;
            font-size: 1.8em;
            font-weight: 600;
            margin: 20px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid #B1AFFF;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 15px 0;
            border-left: 4px solid #007C91;
        }
        
        .emergency-alert {
            background: #F44336;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-weight: bold;
        }
        
        .success-message {
            background: #4CAF50;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }
        
        .warning-message {
            background: #FF9800;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }
        
        .info-box {
            background: #E3F2FD;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #2196F3;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            background-color: #f0f2f6;
            border-radius: 8px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #007C91;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# OpenAI Service Class
class OpenAIService:
    def __init__(self):
        self.model = "gpt-3.5-turbo"
        self.max_tokens = 2000
        self.temperature = 0.7
    
    def setup_client(self, api_key: str):
        """Setup OpenAI client with API key"""
        openai.api_key = api_key
    
    def get_health_analysis(self, prompt: str, patient_data: Dict, analysis_type: str) -> str:
        """Get health analysis from OpenAI"""
        api_key = st.session_state.get('openai_api_key')
        
        if not api_key or api_key == 'demo-key':
            return self._get_demo_response(analysis_type, patient_data)
        
        try:
            self.setup_client(api_key)
            
            system_message = f"""You are MediAI Pro, an advanced AI medical assistant specifically designed for Indian healthcare. 
            You have deep knowledge of:
            1. Indian healthcare system, medical practices, and standards
            2. Common health issues in Indian population
            3. Indian dietary patterns, lifestyle, and cultural factors
            4. Integration of Ayurveda with modern medicine
            5. Indian medical terminology and local healthcare infrastructure
            
            ANALYSIS TYPE: {analysis_type}
            
            Provide comprehensive, culturally sensitive medical information tailored for Indian patients."""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"{prompt}\n\nPatient Data: {json.dumps(patient_data, indent=2)}"}
            ]
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._get_demo_response(analysis_type, patient_data)
    
    def _get_demo_response(self, analysis_type: str, patient_data: Dict) -> str:
        """Provide demo response when API is not available"""
        patient_info = patient_data.get('patient', {})
        return f"""
## {analysis_type} - Demo Mode

**Patient:** {patient_info.get('name', 'Patient')}
**Age:** {patient_info.get('age', 'Unknown')} years

### Analysis Summary
This is a demo response. To get real-time AI analysis:
1. Add your OpenAI API key in the sidebar
2. Ensure you have a stable internet connection
3. Complete the health assessment form

### Key Health Indicators
- BMI Status: Based on Asian standards
- Vital Signs: Within normal range
- Risk Factors: Age and lifestyle-appropriate screening recommended

### Recommendations
1. Regular health checkups every 6 months
2. Maintain balanced Indian diet with seasonal foods
3. Include yoga and pranayama in daily routine
4. Monitor vitals regularly

**Note:** This is a demo response. Connect OpenAI for personalized analysis.
"""

# Weather Service
def get_weather_data(lat: float, lon: float) -> Dict:
    """Get weather data with Indian context"""
    try:
        # OpenWeatherMap API call (you'll need to add your API key)
        api_key = os.getenv('OPENWEATHER_API_KEY', '')
        if api_key:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'description': data['weather'][0]['description'],
                    'feels_like': data['main']['feels_like'],
                    'pressure': data['main']['pressure']
                }
    except:
        pass
    
    # Fallback to simulated data
    return {
        'temperature': round(28 + random.uniform(-5, 5), 1),
        'humidity': random.randint(60, 85),
        'description': random.choice(['clear sky', 'partly cloudy', 'light rain']),
        'feels_like': round(30 + random.uniform(-5, 5), 1),
        'pressure': random.randint(1008, 1018)
    }

# Indian Cities Database
INDIAN_CITIES = {
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777, 'state': 'Maharashtra'},
    'Delhi': {'lat': 28.7041, 'lon': 77.1025, 'state': 'Delhi'},
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946, 'state': 'Karnataka'},
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'state': 'Telangana'},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'state': 'Tamil Nadu'},
    'Kolkata': {'lat': 22.5726, 'lon': 88.3639, 'state': 'West Bengal'},
    'Pune': {'lat': 18.5204, 'lon': 73.8567, 'state': 'Maharashtra'},
    'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'state': 'Gujarat'},
    'Jaipur': {'lat': 26.9124, 'lon': 75.7873, 'state': 'Rajasthan'},
    'Surat': {'lat': 21.1702, 'lon': 72.8311, 'state': 'Gujarat'}
}

# Health Score Calculator
def calculate_health_score(vitals: Dict, bmi: float, age: int, lifestyle: Dict) -> Dict:
    """Calculate health score with Indian parameters"""
    score = 100
    risk_factors = []
    recommendations = []
    
    # BMI Assessment (Asian standards)
    if bmi < 18.5:
        score -= 15
        risk_factors.append("Underweight")
        recommendations.append("Increase caloric intake with nutritious Indian foods")
    elif bmi > 23:  # Asian BMI threshold
        score -= 20
        risk_factors.append("Overweight (Asian BMI standards)")
        recommendations.append("Focus on portion control and regular exercise")
    
    # Blood Pressure
    if vitals.get('bp_systolic', 120) > 130 or vitals.get('bp_diastolic', 80) > 85:
        score -= 20
        risk_factors.append("Elevated blood pressure")
        recommendations.append("Reduce salt intake, practice yoga and meditation")
    
    # Age factors
    if age > 40:
        recommendations.append("Annual health checkups recommended")
        recommendations.append("Diabetes and cardiac screening important for Indians over 40")
    
    # Lifestyle
    if lifestyle.get('exercise_frequency') in ['Never', 'Rarely']:
        score -= 15
        risk_factors.append("Sedentary lifestyle")
        recommendations.append("Start with 30 minutes daily walking or yoga")
    
    score = max(0, min(100, score))
    
    return {
        'score': score,
        'risk_factors': risk_factors,
        'recommendations': recommendations,
        'status': 'Excellent' if score >= 85 else 'Good' if score >= 70 else 'Fair' if score >= 55 else 'Poor'
    }

# Main App
def main():
    load_custom_css()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 MediAI Pro - Advanced Health Assistant for India</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1em; color: #666; margin-bottom: 30px;">Integrating Modern Medicine with Traditional Indian Healthcare</p>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## 🔧 Configuration")
        
        # OpenAI API Configuration
        with st.expander("🔑 OpenAI API Setup", expanded=True):
            api_key = st.text_input(
                "OpenAI API Key:",
                type="password",
                value=st.session_state.get('openai_api_key', ''),
                help="Enter your OpenAI API key for real-time AI analysis"
            )
            
            if api_key:
                st.session_state.openai_api_key = api_key
                if api_key.startswith('sk-'):
                    st.success("✅ API Key configured!")
                else:
                    st.error("❌ Invalid API key format")
        
        st.markdown("---")
        
        # Location Setup
        st.markdown("## 📍 Location Setup")
        
        location_method = st.selectbox(
            "Choose location method:",
            ["Select City", "Manual Entry"]
        )
        
        if location_method == "Select City":
            selected_city = st.selectbox(
                "Select your city:",
                [""] + list(INDIAN_CITIES.keys())
            )
            
            if selected_city and st.button("📍 Set Location"):
                city_data = INDIAN_CITIES[selected_city]
                st.session_state.user_location = {
                    'city': selected_city,
                    'state': city_data['state'],
                    'country': 'India',
                    'lat': city_data['lat'],
                    'lon': city_data['lon']
                }
                st.session_state.coordinates = (city_data['lat'], city_data['lon'])
                st.session_state.location_set = True
                st.success(f"✅ Location set to {selected_city}")
                st.rerun()
        
        # Display current location
        if st.session_state.user_location:
            st.markdown("### 📍 Current Location")
            loc = st.session_state.user_location
            st.info(f"{loc['city']}, {loc['state']}")
    
    # Weather Display
    if st.session_state.coordinates and st.session_state.location_set:
        weather_data = get_weather_data(st.session_state.coordinates[0], st.session_state.coordinates[1])
        st.session_state.weather_data = weather_data
        
        st.markdown('<h2 class="sub-header">🌤️ Current Weather</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌡️ Temperature", f"{weather_data['temperature']}°C", f"Feels like {weather_data['feels_like']}°C")
        with col2:
            st.metric("💧 Humidity", f"{weather_data['humidity']}%")
        with col3:
            st.metric("🌤️ Condition", weather_data['description'].title())
        with col4:
            st.metric("🔵 Pressure", f"{weather_data['pressure']} hPa")
    
    st.markdown("---")
    
    # Health Assessment Form
    st.markdown('<h2 class="sub-header">👤 Health Assessment Form</h2>', unsafe_allow_html=True)
    
    with st.form("health_assessment_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            name = st.text_input("Full Name*", help="Enter your complete name")
            age = st.number_input("Age*", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
        
        with col2:
            weight = st.number_input("Weight (kg)*", min_value=1.0, max_value=300.0, value=70.0)
            height = st.number_input("Height (cm)*", min_value=50.0, max_value=250.0, value=170.0)
            blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"])
        
        with col3:
            exercise_freq = st.selectbox("Exercise Frequency", 
                ["Never", "Rarely", "1-2 times/week", "3-4 times/week", "5+ times/week"])
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            sleep_hours = st.slider("Average Sleep (hours)", 3, 12, 7)
        
        st.markdown("### 🩺 Vital Signs")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bp_systolic = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120)
            bp_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80)
        
        with col2:
            pulse = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=75)
            temperature = st.number_input("Temperature (°F)", min_value=90.0, max_value=110.0, value=98.6)
        
        with col3:
            spo2 = st.number_input("SpO2 (%)", min_value=70, max_value=100, value=98)
            respiratory_rate = st.number_input("Respiratory Rate", min_value=8, max_value=40, value=16)
        
        st.markdown("### 🔍 Current Health Status")
        
        symptoms = st.text_area("Current Symptoms (if any)", 
            placeholder="Describe any symptoms you're experiencing...")
        
        medical_history = st.text_area("Medical History",
            placeholder="List any chronic conditions, surgeries, or medications...")
        
        # Emergency symptoms
        emergency_symptoms = st.multiselect(
            "⚠️ Select if experiencing any emergency symptoms:",
            ["Chest pain", "Difficulty breathing", "Severe headache", "Loss of consciousness", 
             "Severe bleeding", "High fever (>103°F)"]
        )
        
        submit_button = st.form_submit_button("🚀 Complete Health Analysis", use_container_width=True)
    
    # Process form submission
    if submit_button and name:
        # Store health data
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        
        vitals = {
            'bp_systolic': bp_systolic,
            'bp_diastolic': bp_diastolic,
            'pulse': pulse,
            'temperature': temperature,
            'spo2': spo2,
            'respiratory_rate': respiratory_rate
        }
        
        lifestyle = {
            'exercise_frequency': exercise_freq,
            'stress_level': stress_level,
            'sleep_hours': sleep_hours
        }
        
        st.session_state.health_data = {
            'patient': {
                'name': name,
                'age': age,
                'gender': gender,
                'weight': weight,
                'height': height,
                'blood_group': blood_group
            },
            'vitals': vitals,
            'bmi': bmi,
            'symptoms': symptoms,
            'medical_history': medical_history,
            'lifestyle': lifestyle,
            'emergency_symptoms': emergency_symptoms,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.session_state.form_submitted = True
        
        # Show emergency alert if needed
        if emergency_symptoms:
            st.markdown("""
            <div class="emergency-alert">
                <h2>🚨 MEDICAL EMERGENCY DETECTED</h2>
                <p>You have reported emergency symptoms. Please contact emergency services immediately!</p>
                <p>🚑 Call 108 (Emergency) or 112 (National Emergency Number)</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-message">✅ Health assessment completed successfully!</div>', 
                unsafe_allow_html=True)
    
    # Analysis Section
    if st.session_state.form_submitted and st.session_state.health_data:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 Health Analysis Dashboard</h2>', unsafe_allow_html=True)
        
        # Health Score
        health_data = st.session_state.health_data
        health_score = calculate_health_score(
            health_data['vitals'],
            health_data['bmi'],
            health_data['patient']['age'],
            health_data['lifestyle']
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Display health score
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = health_score['score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Health Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 55], 'color': "lightgray"},
                        {'range': [55, 70], 'color': "yellow"},
                        {'range': [70, 85], 'color': "lightgreen"},
                        {'range': [85, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"<h3 style='text-align: center;'>Status: {health_score['status']}</h3>", 
                unsafe_allow_html=True)
        
        # BMI Analysis
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("BMI", f"{health_data['bmi']:.1f}")
        with col2:
            bmi_category = "Underweight" if health_data['bmi'] < 18.5 else "Normal" if health_data['bmi'] < 23 else "Overweight" if health_data['bmi'] < 27 else "Obese"
            st.metric("Category", bmi_category)
        with col3:
            ideal_weight_min = 18.5 * (health_data['patient']['height']/100) ** 2
            ideal_weight_max = 22.9 * (health_data['patient']['height']/100) ** 2
            st.metric("Ideal Weight", f"{ideal_weight_min:.0f}-{ideal_weight_max:.0f} kg")
        with col4:
            st.metric("BP", f"{health_data['vitals']['bp_systolic']}/{health_data['vitals']['bp_diastolic']}")
        
        # Risk Factors and Recommendations
        if health_score['risk_factors']:
            st.markdown("### ⚠️ Risk Factors")
            for factor in health_score['risk_factors']:
                st.warning(f"• {factor}")
        
        if health_score['recommendations']:
            st.markdown("### 💡 Recommendations")
            for rec in health_score['recommendations']:
                st.info(f"• {rec}")
        
        # Tabs for different analyses
        tabs = st.tabs(["🤖 AI Analysis", "🏥 Nearby Hospitals", "💊 Pharmacies", 
                        "🧪 Lab Tests", "🥗 Diet Plan", "📰 Health News"])
        
        with tabs[0]:
            st.markdown("### 🤖 AI-Powered Health Analysis")
            
            analysis_type = st.selectbox(
                "Select Analysis Type:",
                ["Comprehensive Health Analysis", "Symptom Analysis", 
                 "Lifestyle Recommendations", "Risk Assessment"]
            )
            
            if st.button("🔍 Get AI Analysis", use_container_width=True):
                with st.spinner("Analyzing your health data..."):
                    ai_service = OpenAIService()
                    
                    # Create prompt based on analysis type
                    prompts = {
                        "Comprehensive Health Analysis": f"""
                        Provide a comprehensive health analysis for an Indian patient with the following data.
                        Consider Indian health standards, common conditions in Indian population, and 
                        provide culturally appropriate recommendations.
                        """,
                        "Symptom Analysis": f"""
                        Analyze the reported symptoms and provide detailed recommendations considering
                        common health issues in India, environmental factors, and appropriate treatment options.
                        """,
                        "Lifestyle Recommendations": f"""
                        Provide lifestyle recommendations tailored for Indian culture, including diet
                        (vegetarian/non-vegetarian options), exercise (yoga, walking), and stress management.
                        """,
                        "Risk Assessment": f"""
                        Assess health risks specific to Indian population including diabetes, heart disease,
                        and other conditions common in India. Consider genetic predispositions.
                        """
                    }
                    
                    prompt = prompts.get(analysis_type, prompts["Comprehensive Health Analysis"])
                    
                    response = ai_service.get_health_analysis(
                        prompt,
                        st.session_state.health_data,
                        analysis_type
                    )
                    
                    # Store and display response
                    st.session_state.analysis_results[analysis_type] = response
                    
                    st.markdown("### Analysis Results")
                    st.markdown(f'<div class="metric-card">{response}</div>', unsafe_allow_html=True)
            
            # Show previous analyses
            if analysis_type in st.session_state.analysis_results:
                st.markdown("### Previous Analysis")
                st.markdown(f'<div class="info-box">{st.session_state.analysis_results[analysis_type]}</div>', 
                    unsafe_allow_html=True)
        
        with tabs[1]:
            st.markdown("### 🏥 Nearby Hospitals")
            
            if st.session_state.coordinates:
                # Simulated hospital data
                hospitals = [
                    {"name": "AIIMS Delhi", "distance": 5.2, "type": "Government", "rating": 4.8, "beds": 2478},
                    {"name": "Apollo Hospital", "distance": 3.1, "type": "Private", "rating": 4.5, "beds": 500},
                    {"name": "Fortis Healthcare", "distance": 7.8, "type": "Private", "rating": 4.3, "beds": 400},
                    {"name": "Max Healthcare", "distance": 9.2, "type": "Private", "rating": 4.4, "beds": 350},
                    {"name": "Government District Hospital", "distance": 2.5, "type": "Government", "rating": 3.8, "beds": 300}
                ]
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    hospital_type = st.selectbox("Filter by Type:", ["All", "Government", "Private"])
                with col2:
                    max_distance = st.slider("Maximum Distance (km):", 1, 20, 10)
                
                # Filter hospitals
                filtered_hospitals = [h for h in hospitals if h['distance'] <= max_distance]
                if hospital_type != "All":
                    filtered_hospitals = [h for h in filtered_hospitals if h['type'] == hospital_type]
                
                # Display hospitals
                for hospital in sorted(filtered_hospitals, key=lambda x: x['distance']):
                    with st.expander(f"🏥 {hospital['name']} - {hospital['distance']} km"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Type", hospital['type'])
                        with col2:
                            st.metric("Rating", f"⭐ {hospital['rating']}/5")
                        with col3:
                            st.metric("Beds", hospital['beds'])
                        
                        st.info(f"📞 Emergency: 108 | 📍 Distance: {hospital['distance']} km")
            else:
                st.warning("Please set your location to find nearby hospitals.")
        
        with tabs[2]:
            st.markdown("### 💊 Nearby Pharmacies")
            
            if st.session_state.coordinates:
                # Simulated pharmacy data
                pharmacies = [
                    {"name": "Apollo Pharmacy", "distance": 0.8, "type": "Chain", "24x7": True},
                    {"name": "Jan Aushadhi Store", "distance": 1.2, "type": "Government", "24x7": False},
                    {"name": "MedPlus", "distance": 1.5, "type": "Chain", "24x7": True},
                    {"name": "Local Medical Store", "distance": 0.5, "type": "Independent", "24x7": False}
                ]
                
                for pharmacy in sorted(pharmacies, key=lambda x: x['distance']):
                    with st.expander(f"💊 {pharmacy['name']} - {pharmacy['distance']} km"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Type", pharmacy['type'])
                        with col2:
                            st.metric("24x7", "✅ Yes" if pharmacy['24x7'] else "❌ No")
                        
                        if pharmacy['type'] == "Government":
                            st.success("🏛️ Jan Aushadhi - Generic medicines at 90% less cost!")
            else:
                st.warning("Please set your location to find nearby pharmacies.")
        
        with tabs[3]:
            st.markdown("### 🧪 Recommended Lab Tests")
            
            age = health_data['patient']['age']
            gender = health_data['patient']['gender']
            
            # Basic tests for everyone
            tests = [
                {"name": "Complete Blood Count (CBC)", "frequency": "Annual", "cost": "₹200-400"},
                {"name": "Lipid Profile", "frequency": "Annual", "cost": "₹400-600"},
                {"name": "HbA1c (Diabetes)", "frequency": "Annual", "cost": "₹300-500"},
                {"name": "Thyroid Function Test", "frequency": "Annual", "cost": "₹600-1000"},
                {"name": "Vitamin D3", "frequency": "Annual", "cost": "₹800-1200"},
                {"name": "Vitamin B12", "frequency": "Annual", "cost": "₹600-1000"}
            ]
            
            # Age-specific tests
            if age > 40:
                tests.extend([
                    {"name": "ECG", "frequency": "Annual", "cost": "₹200-500"},
                    {"name": "Cardiac Markers", "frequency": "As needed", "cost": "₹2000-4000"}
                ])
            
            # Gender-specific tests
            if gender == "Female":
                tests.append({"name": "Iron Studies", "frequency": "Annual", "cost": "₹500-800"})
                if age > 21:
                    tests.append({"name": "Pap Smear", "frequency": "Every 3 years", "cost": "₹800-1500"})
            
            # Display tests
            for test in tests:
                with st.expander(f"🧪 {test['name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Frequency:** {test['frequency']}")
                    with col2:
                        st.write(f"**Cost Range:** {test['cost']}")
        
        with tabs[4]:
            st.markdown("### 🥗 Personalized Diet Plan")
            
            # Calculate calorie needs
            if gender == "Male":
                bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
            else:
                bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
            
            activity_multiplier = 1.2 if exercise_freq == "Never" else 1.375 if exercise_freq == "Rarely" else 1.55
            daily_calories = int(bmr * activity_multiplier)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Daily Calories", f"{daily_calories} kcal")
            with col2:
                st.metric("Protein", f"{int(daily_calories * 0.15 / 4)}g")
            with col3:
                st.metric("Carbs", f"{int(daily_calories * 0.60 / 4)}g")
            
            # Indian meal plan
            st.markdown("#### 🍽️ Traditional Indian Meal Plan")
            
            meal_plan = {
                "Breakfast (8 AM)": ["2 Idli/Dosa with sambar", "OR 2 Parathas with curd", "1 glass milk/tea"],
                "Mid-Morning (11 AM)": ["1 seasonal fruit", "Handful of nuts"],
                "Lunch (1 PM)": ["2 rotis", "1 bowl dal", "Vegetable curry", "Rice (1 bowl)", "Curd"],
                "Evening (5 PM)": ["Tea with 2 biscuits", "OR Roasted chana"],
                "Dinner (8 PM)": ["2 rotis", "Dal/Paneer curry", "Salad", "Vegetables"]
            }
            
            for meal, items in meal_plan.items():
                with st.expander(meal):
                    for item in items:
                        st.write(f"• {item}")
        
        with tabs[5]:
            st.markdown("### 📰 Latest Health News - India")
            
            # Simulated health news
            news_articles = [
                {
                    "title": "AIIMS Launches New Telemedicine Initiative",
                    "date": "June 20, 2024",
                    "summary": "AIIMS Delhi introduces AI-powered telemedicine for rural healthcare."
                },
                {
                    "title": "Ayushman Bharat Reaches 500 Million Indians",
                    "date": "June 18, 2024",
                    "summary": "Government health insurance scheme achieves major milestone."
                },
                {
                    "title": "New Study Links Air Pollution to Heart Disease",
                    "date": "June 15, 2024",
                    "summary": "Indian researchers find strong correlation between PM2.5 and cardiac issues."
                }
            ]
            
            for article in news_articles:
                with st.expander(f"📰 {article['title']}"):
                    st.write(f"**Date:** {article['date']}")
                    st.write(article['summary'])
                    st.button("Read More", key=f"news_{article['title'][:10]}")

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h3>🏥 MediAI Pro - Your Health, Our Priority</h3>
        <p><strong>Emergency Numbers:</strong> 🚑 108 (Medical) | 112 (National Emergency)</p>
        <p><strong>Disclaimer:</strong> This app provides health information for educational purposes only. 
        Always consult qualified healthcare professionals for medical advice.</p>
        <p>Made with ❤️ for India | Version 2.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
        show_footer()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")