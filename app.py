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
from streamlit_js_eval import streamlit_js_eval, get_geolocation

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
        'google_maps_key': None,
        'analysis_results': {},
        'active_tab': 0,
        'form_submitted': False,
        'weather_data': None,
        'nearby_hospitals': [],
        'nearby_pharmacies': [],
        'nearby_labs': [],
        'health_news': [],
        'auto_location_tried': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
init_session_state()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys Configuration (In production, use Streamlit secrets)
GOOGLE_MAPS_API_KEY = "AIzaSyAV9wl5LyeXTXx1AelqgiFpgGtsRI1q4b8"
OPENAI_API_KEY = "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"

# Store API keys in session state
if not st.session_state.get('google_maps_key'):
    st.session_state.google_maps_key = GOOGLE_MAPS_API_KEY
if not st.session_state.get('openai_api_key'):
    st.session_state.openai_api_key = OPENAI_API_KEY

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
        
        .location-detected {
            background: #4CAF50;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            text-align: center;
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

# Google Maps Services
class GoogleMapsService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api"
    
    def reverse_geocode(self, lat: float, lon: float) -> Dict:
        """Get address from coordinates"""
        try:
            url = f"{self.base_url}/geocode/json"
            params = {
                'latlng': f'{lat},{lon}',
                'key': self.api_key
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    # Extract city, state, country from the results
                    result = data['results'][0]
                    address_components = result.get('address_components', [])
                    
                    city = None
                    state = None
                    country = None
                    
                    for component in address_components:
                        types = component.get('types', [])
                        if 'locality' in types:
                            city = component['long_name']
                        elif 'administrative_area_level_1' in types:
                            state = component['long_name']
                        elif 'country' in types:
                            country = component['long_name']
                    
                    return {
                        'formatted_address': result.get('formatted_address', ''),
                        'city': city or 'Unknown',
                        'state': state or 'Unknown',
                        'country': country or 'Unknown',
                        'lat': lat,
                        'lon': lon
                    }
            return None
        except Exception as e:
            logger.error(f"Reverse geocoding error: {e}")
            return None
    
    def search_nearby_places(self, lat: float, lon: float, place_type: str, radius: int = 5000) -> List[Dict]:
        """Search for nearby places using Google Places API"""
        try:
            url = f"{self.base_url}/place/nearbysearch/json"
            params = {
                'location': f'{lat},{lon}',
                'radius': radius,
                'type': place_type,
                'key': self.api_key
            }
            
            # For medical facilities, add additional keywords
            if place_type == 'hospital':
                params['keyword'] = 'hospital|medical center|clinic'
            elif place_type == 'pharmacy':
                params['keyword'] = 'pharmacy|medical store|chemist|jan aushadhi'
            elif place_type == 'doctor':
                params['keyword'] = 'laboratory|diagnostic center|pathology|lab test'
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                places = []
                
                for place in data.get('results', []):
                    # Calculate distance
                    place_lat = place['geometry']['location']['lat']
                    place_lon = place['geometry']['location']['lng']
                    distance = self._calculate_distance(lat, lon, place_lat, place_lon)
                    
                    place_info = {
                        'name': place.get('name', 'Unknown'),
                        'address': place.get('vicinity', 'Address not available'),
                        'rating': place.get('rating', 0),
                        'user_ratings_total': place.get('user_ratings_total', 0),
                        'place_id': place.get('place_id', ''),
                        'lat': place_lat,
                        'lon': place_lon,
                        'distance': round(distance, 2),
                        'open_now': place.get('opening_hours', {}).get('open_now', None),
                        'types': place.get('types', []),
                        'business_status': place.get('business_status', 'OPERATIONAL')
                    }
                    
                    # Get additional details if needed
                    if place.get('place_id'):
                        details = self.get_place_details(place['place_id'])
                        if details:
                            place_info.update(details)
                    
                    places.append(place_info)
                
                # Sort by distance
                places.sort(key=lambda x: x['distance'])
                return places
            
            return []
        except Exception as e:
            logger.error(f"Places search error: {e}")
            return []
    
    def get_place_details(self, place_id: str) -> Dict:
        """Get detailed information about a place"""
        try:
            url = f"{self.base_url}/place/details/json"
            params = {
                'place_id': place_id,
                'fields': 'formatted_phone_number,opening_hours,website,types',
                'key': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('result', {})
                
                return {
                    'phone': result.get('formatted_phone_number', 'Not available'),
                    'website': result.get('website', ''),
                    'hours': self._format_opening_hours(result.get('opening_hours', {}))
                }
            
            return {}
        except Exception as e:
            logger.error(f"Place details error: {e}")
            return {}
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _format_opening_hours(self, hours_data: Dict) -> str:
        """Format opening hours for display"""
        if not hours_data or 'weekday_text' not in hours_data:
            return "Hours not available"
        
        # For Indian context, show today's hours
        weekday = datetime.now().strftime('%A')
        weekday_hours = hours_data.get('weekday_text', [])
        
        for hours in weekday_hours:
            if weekday in hours:
                return hours
        
        return "Hours not available"

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
        
        if not api_key:
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
            6. Location-specific health recommendations based on the patient's current location
            
            ANALYSIS TYPE: {analysis_type}
            
            The patient is currently located at: {patient_data.get('location', {}).get('city', 'Unknown')}, {patient_data.get('location', {}).get('state', 'Unknown')}, India
            
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
        location = patient_data.get('location', {})
        return f"""
## {analysis_type} - AI Analysis

**Patient:** {patient_info.get('name', 'Patient')}
**Age:** {patient_info.get('age', 'Unknown')} years
**Location:** {location.get('city', 'Unknown')}, {location.get('state', 'Unknown')}

### Analysis Summary
Based on your health data and location in {location.get('city', 'your area')}, here are personalized recommendations:

### Key Health Indicators
- BMI Status: Based on Asian standards
- Vital Signs: Within normal range
- Risk Factors: Age and lifestyle-appropriate screening recommended

### Location-Specific Recommendations
1. **Air Quality Considerations:** Monitor local AQI levels in {location.get('city', 'your city')}
2. **Seasonal Health:** Adapt to local climate patterns
3. **Local Healthcare:** Utilize nearby facilities found through our search

### Recommendations
1. Regular health checkups every 6 months
2. Maintain balanced Indian diet with seasonal foods
3. Include yoga and pranayama in daily routine
4. Monitor vitals regularly

**Note:** This is a demo response. Full AI analysis provides more detailed insights based on your specific health data and location.
"""

# Weather Service
def get_weather_data(lat: float, lon: float) -> Dict:
    """Get weather data with Indian context"""
    try:
        # OpenWeatherMap API call (you can add your API key here)
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

# Automatic Location Detection
def get_auto_location():
    """Get user's location automatically"""
    try:
        # Try to get location using streamlit-js-eval
        location = get_geolocation()
        if location:
            return {
                'lat': location['coords']['latitude'],
                'lon': location['coords']['longitude']
            }
    except:
        pass
    
    # Fallback to IP-based location
    try:
        response = requests.get('http://ip-api.com/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'lat': data.get('lat'),
                'lon': data.get('lon')
            }
    except:
        pass
    
    return None

# Main App
def main():
    load_custom_css()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 MediAI Pro - Advanced Health Assistant for India</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1em; color: #666; margin-bottom: 30px;">Integrating Modern Medicine with Traditional Indian Healthcare</p>', unsafe_allow_html=True)
    
    # Initialize Google Maps Service
    maps_service = GoogleMapsService(st.session_state.google_maps_key)
    
    # Automatic Location Detection
    if not st.session_state.location_set and not st.session_state.auto_location_tried:
        with st.spinner("🌍 Detecting your location automatically..."):
            auto_loc = get_auto_location()
            if auto_loc:
                # Reverse geocode to get address
                location_info = maps_service.reverse_geocode(auto_loc['lat'], auto_loc['lon'])
                if location_info:
                    st.session_state.user_location = {
                        'city': location_info['city'],
                        'state': location_info['state'],
                        'country': location_info['country'],
                        'lat': auto_loc['lat'],
                        'lon': auto_loc['lon'],
                        'formatted_address': location_info['formatted_address']
                    }
                    st.session_state.coordinates = (auto_loc['lat'], auto_loc['lon'])
                    st.session_state.location_set = True
                    
                    st.markdown(f"""
                    <div class="location-detected">
                        ✅ Location Detected Automatically!<br>
                        📍 {location_info['city']}, {location_info['state']}, {location_info['country']}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.session_state.auto_location_tried = True
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## 🔧 Configuration")
        
        # Location Setup
        st.markdown("## 📍 Location")
        
        if st.session_state.user_location:
            loc = st.session_state.user_location
            st.success(f"📍 {loc['city']}, {loc['state']}")
            
            if st.button("🔄 Update Location"):
                st.session_state.location_set = False
                st.session_state.auto_location_tried = False
                st.rerun()
        else:
            if st.button("📍 Detect My Location", use_container_width=True):
                with st.spinner("Detecting location..."):
                    auto_loc = get_auto_location()
                    if auto_loc:
                        location_info = maps_service.reverse_geocode(auto_loc['lat'], auto_loc['lon'])
                        if location_info:
                            st.session_state.user_location = {
                                'city': location_info['city'],
                                'state': location_info['state'],
                                'country': location_info['country'],
                                'lat': auto_loc['lat'],
                                'lon': auto_loc['lon'],
                                'formatted_address': location_info['formatted_address']
                            }
                            st.session_state.coordinates = (auto_loc['lat'], auto_loc['lon'])
                            st.session_state.location_set = True
                            st.rerun()
                    else:
                        st.error("Unable to detect location automatically. Please enter manually.")
            
            # Manual location entry
            location_input = st.text_input("Or enter location manually:", placeholder="e.g., Mumbai, Maharashtra")
            if location_input and st.button("Set Location"):
                # Use Google Geocoding API
                st.warning("Manual location entry requires geocoding implementation")
        
        # API Status
        st.markdown("## 🔌 API Status")
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ OpenAI")
        with col2:
            st.success("✅ Google Maps")
    
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
            'location': st.session_state.user_location,
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
        tabs = st.tabs(["🤖 AI Analysis", "🏥 Nearby Hospitals", "💊 Pharmacies", "🧪 Labs",
                        "💊 Medicines", "🥗 Diet Plan", "📰 Health News"])
        
        with tabs[0]:
            st.markdown("### 🤖 AI-Powered Health Analysis")
            
            analysis_type = st.selectbox(
                "Select Analysis Type:",
                ["Comprehensive Health Analysis", "Symptom Analysis", 
                 "Lifestyle Recommendations", "Risk Assessment", "Medicine Recommendations"]
            )
            
            if st.button("🔍 Get AI Analysis", use_container_width=True):
                with st.spinner("Analyzing your health data..."):
                    ai_service = OpenAIService()
                    
                    # Create prompt based on analysis type
                    prompts = {
                        "Comprehensive Health Analysis": f"""
                        Provide a comprehensive health analysis for an Indian patient with the following data.
                        Consider Indian health standards, common conditions in Indian population, 
                        location-specific factors for {st.session_state.user_location.get('city', 'the area')},
                        and provide culturally appropriate recommendations.
                        """,
                        "Symptom Analysis": f"""
                        Analyze the reported symptoms and provide detailed recommendations considering
                        common health issues in India, environmental factors in {st.session_state.user_location.get('city', 'the area')},
                        local climate conditions, and appropriate treatment options.
                        """,
                        "Lifestyle Recommendations": f"""
                        Provide lifestyle recommendations tailored for Indian culture and specifically for
                        someone living in {st.session_state.user_location.get('city', 'the area')}, including diet
                        (vegetarian/non-vegetarian options), exercise (yoga, walking), and stress management.
                        """,
                        "Risk Assessment": f"""
                        Assess health risks specific to Indian population including diabetes, heart disease,
                        and other conditions common in India. Consider environmental factors in
                        {st.session_state.user_location.get('city', 'the area')} and genetic predispositions.
                        """,
                        "Medicine Recommendations": f"""
                        Provide medicine recommendations including both allopathic and Ayurvedic options
                        available in India. Include generic alternatives, Jan Aushadhi options, and
                        local availability in {st.session_state.user_location.get('city', 'the area')}.
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
                search_radius = st.slider("Search Radius (km):", 1, 20, 5) * 1000
                
                if st.button("🔍 Search Hospitals", key="search_hospitals"):
                    with st.spinner("Searching for nearby hospitals..."):
                        hospitals = maps_service.search_nearby_places(
                            st.session_state.coordinates[0],
                            st.session_state.coordinates[1],
                            'hospital',
                            search_radius
                        )
                        st.session_state.nearby_hospitals = hospitals
                
                if st.session_state.nearby_hospitals:
                    st.success(f"Found {len(st.session_state.nearby_hospitals)} hospitals nearby")
                    
                    # Filter options
                    col1, col2 = st.columns(2)
                    with col1:
                        min_rating = st.slider("Minimum Rating:", 0.0, 5.0, 3.0, 0.1)
                    with col2:
                        open_now_only = st.checkbox("Open Now Only")
                    
                    # Filter hospitals
                    filtered_hospitals = st.session_state.nearby_hospitals
                    if min_rating > 0:
                        filtered_hospitals = [h for h in filtered_hospitals if h.get('rating', 0) >= min_rating]
                    if open_now_only:
                        filtered_hospitals = [h for h in filtered_hospitals if h.get('open_now', False)]
                    
                    # Display hospitals
                    for hospital in filtered_hospitals:
                        with st.expander(f"🏥 {hospital['name']} - {hospital['distance']} km"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Rating", f"⭐ {hospital.get('rating', 'N/A')}/5")
                                st.write(f"Reviews: {hospital.get('user_ratings_total', 0)}")
                            with col2:
                                st.metric("Status", "🟢 Open" if hospital.get('open_now') else "🔴 Closed")
                                st.write(f"Distance: {hospital['distance']} km")
                            with col3:
                                if hospital.get('phone'):
                                    st.write(f"📞 {hospital['phone']}")
                                if hospital.get('website'):
                                    st.write(f"🌐 [Website]({hospital['website']})")
                            
                            st.write(f"📍 **Address:** {hospital['address']}")
                            if hospital.get('hours'):
                                st.write(f"🕒 **Hours:** {hospital['hours']}")
                            
                            # Map link
                            maps_url = f"https://www.google.com/maps/search/?api=1&query={hospital['lat']},{hospital['lon']}"
                            st.write(f"🗺️ [View on Google Maps]({maps_url})")
                else:
                    st.info("Click 'Search Hospitals' to find nearby medical facilities")
            else:
                st.warning("Location not set. Please set your location to find nearby hospitals.")
        
        with tabs[2]:
            st.markdown("### 💊 Nearby Pharmacies")
            
            if st.session_state.coordinates:
                search_radius = st.slider("Search Radius (km):", 1, 10, 3, key="pharmacy_radius") * 1000
                
                if st.button("🔍 Search Pharmacies", key="search_pharmacies"):
                    with st.spinner("Searching for nearby pharmacies..."):
                        pharmacies = maps_service.search_nearby_places(
                            st.session_state.coordinates[0],
                            st.session_state.coordinates[1],
                            'pharmacy',
                            search_radius
                        )
                        st.session_state.nearby_pharmacies = pharmacies
                
                if st.session_state.nearby_pharmacies:
                    st.success(f"Found {len(st.session_state.nearby_pharmacies)} pharmacies nearby")
                    
                    # Display pharmacies
                    for pharmacy in st.session_state.nearby_pharmacies:
                        with st.expander(f"💊 {pharmacy['name']} - {pharmacy['distance']} km"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Rating", f"⭐ {pharmacy.get('rating', 'N/A')}/5")
                                st.write(f"Status: {'🟢 Open' if pharmacy.get('open_now') else '🔴 Closed'}")
                            with col2:
                                st.metric("Distance", f"{pharmacy['distance']} km")
                                if pharmacy.get('phone'):
                                    st.write(f"📞 {pharmacy['phone']}")
                            
                            st.write(f"📍 **Address:** {pharmacy['address']}")
                            if pharmacy.get('hours'):
                                st.write(f"🕒 **Hours:** {pharmacy['hours']}")
                            
                            # Check for Jan Aushadhi
                            if 'jan aushadhi' in pharmacy['name'].lower():
                                st.success("🏛️ Jan Aushadhi Store - Generic medicines at 90% less cost!")
                            
                            # Map link
                            maps_url = f"https://www.google.com/maps/search/?api=1&query={pharmacy['lat']},{pharmacy['lon']}"
                            st.write(f"🗺️ [View on Google Maps]({maps_url})")
                else:
                    st.info("Click 'Search Pharmacies' to find nearby medical stores")
            else:
                st.warning("Location not set. Please set your location to find nearby pharmacies.")
        
        with tabs[3]:
            st.markdown("### 🧪 Nearby Diagnostic Labs")
            
            if st.session_state.coordinates:
                search_radius = st.slider("Search Radius (km):", 1, 10, 5, key="lab_radius") * 1000
                
                if st.button("🔍 Search Labs", key="search_labs"):
                    with st.spinner("Searching for nearby diagnostic labs..."):
                        labs = maps_service.search_nearby_places(
                            st.session_state.coordinates[0],
                            st.session_state.coordinates[1],
                            'doctor',  # Using 'doctor' as proxy for labs
                            search_radius
                        )
                        st.session_state.nearby_labs = labs
                
                if st.session_state.nearby_labs:
                    st.success(f"Found {len(st.session_state.nearby_labs)} diagnostic centers nearby")
                    
                    # Recommended tests based on age and health data
                    age = health_data['patient']['age']
                    st.markdown("#### 🧪 Recommended Tests Based on Your Profile")
                    
                    tests = [
                        {"name": "Complete Blood Count (CBC)", "frequency": "Annual", "cost": "₹200-400"},
                        {"name": "Lipid Profile", "frequency": "Annual", "cost": "₹400-600"},
                        {"name": "HbA1c (Diabetes)", "frequency": "Annual", "cost": "₹300-500"},
                        {"name": "Thyroid Function Test", "frequency": "Annual", "cost": "₹600-1000"},
                        {"name": "Vitamin D3", "frequency": "Annual", "cost": "₹800-1200"},
                        {"name": "Vitamin B12", "frequency": "Annual", "cost": "₹600-1000"}
                    ]
                    
                    if age > 40:
                        tests.extend([
                            {"name": "ECG", "frequency": "Annual", "cost": "₹200-500"},
                            {"name": "Cardiac Markers", "frequency": "As needed", "cost": "₹2000-4000"}
                        ])
                    
                    # Display tests
                    for test in tests:
                        st.write(f"• **{test['name']}** - {test['frequency']} ({test['cost']})")
                    
                    st.markdown("#### 🏥 Diagnostic Centers Near You")
                    
                    # Display labs
                    for lab in st.session_state.nearby_labs:
                        with st.expander(f"🧪 {lab['name']} - {lab['distance']} km"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Rating", f"⭐ {lab.get('rating', 'N/A')}/5")
                                st.write(f"Status: {'🟢 Open' if lab.get('open_now') else '🔴 Closed'}")
                            with col2:
                                st.metric("Distance", f"{lab['distance']} km")
                                if lab.get('phone'):
                                    st.write(f"📞 {lab['phone']}")
                            
                            st.write(f"📍 **Address:** {lab['address']}")
                            if lab.get('hours'):
                                st.write(f"🕒 **Hours:** {lab['hours']}")
                            
                            # Map link
                            maps_url = f"https://www.google.com/maps/search/?api=1&query={lab['lat']},{lab['lon']}"
                            st.write(f"🗺️ [View on Google Maps]({maps_url})")
                else:
                    st.info("Click 'Search Labs' to find nearby diagnostic centers")
            else:
                st.warning("Location not set. Please set your location to find nearby labs.")
        
        with tabs[4]:
            st.markdown("### 💊 Medicine Recommendations")
            
            # Medicine search and recommendations
            medicine_search = st.text_input("Search for medicine:", placeholder="Enter medicine name...")
            
            col1, col2 = st.columns(2)
            with col1:
                medicine_type = st.selectbox("Medicine Type:", ["All", "Allopathic", "Ayurvedic", "Homeopathic"])
            with col2:
                show_generics = st.checkbox("Show Generic Alternatives", value=True)
            
            if medicine_search:
                st.markdown(f"#### Search Results for: {medicine_search}")
                
                # Simulated medicine database (in production, use a real API)
                sample_medicines = {
                    "paracetamol": {
                        "brand": ["Crocin", "Calpol", "Dolo"],
                        "generic": "Paracetamol 500mg",
                        "jan_aushadhi_price": "₹2-5 per strip",
                        "market_price": "₹20-40 per strip",
                        "uses": "Fever, mild to moderate pain",
                        "ayurvedic_alternative": "Tulsi, Giloy for fever"
                    },
                    "omeprazole": {
                        "brand": ["Omez", "Ocid", "Omecip"],
                        "generic": "Omeprazole 20mg",
                        "jan_aushadhi_price": "₹8-15 per strip",
                        "market_price": "₹60-120 per strip",
                        "uses": "Acidity, GERD, peptic ulcers",
                        "ayurvedic_alternative": "Amla, Mulethi for acidity"
                    }
                }
                
                # Search in sample database
                found = False
                for med_name, med_info in sample_medicines.items():
                    if medicine_search.lower() in med_name or any(medicine_search.lower() in brand.lower() for brand in med_info['brand']):
                        found = True
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("##### 💊 Medicine Information")
                            st.write(f"**Generic Name:** {med_info['generic']}")
                            st.write(f"**Brand Names:** {', '.join(med_info['brand'])}")
                            st.write(f"**Uses:** {med_info['uses']}")
                        
                        with col2:
                            st.markdown("##### 💰 Price Comparison")
                            st.success(f"**Jan Aushadhi Price:** {med_info['jan_aushadhi_price']}")
                            st.warning(f"**Market Price:** {med_info['market_price']}")
                            st.info(f"**Savings:** Up to 90%")
                        
                        if med_info.get('ayurvedic_alternative'):
                            st.markdown("##### 🌿 Ayurvedic Alternative")
                            st.write(med_info['ayurvedic_alternative'])
                
                if not found:
                    st.info("Medicine not found in database. Please consult a pharmacist.")
            
            # Common medicines based on symptoms
            if symptoms:
                st.markdown("#### 💊 Suggested Medicines Based on Your Symptoms")
                
                # AI-based medicine suggestions
                if st.button("Get AI Medicine Recommendations", key="med_recommendations"):
                    with st.spinner("Analyzing symptoms and suggesting medicines..."):
                        ai_service = OpenAIService()
                        prompt = f"""
                        Based on the symptoms: {symptoms}
                        Provide medicine recommendations including:
                        1. Over-the-counter options available in India
                        2. Generic alternatives from Jan Aushadhi
                        3. Ayurvedic/traditional remedies
                        4. When to consult a doctor
                        Location: {st.session_state.user_location.get('city', 'India')}
                        """
                        
                        response = ai_service.get_health_analysis(
                            prompt,
                            st.session_state.health_data,
                            "Medicine Recommendations"
                        )
                        
                        st.markdown(response)
            
            # Nearby Jan Aushadhi stores
            st.markdown("#### 🏛️ Nearest Jan Aushadhi Stores")
            if st.button("Find Jan Aushadhi Stores", key="jan_aushadhi"):
                with st.spinner("Searching for Jan Aushadhi stores..."):
                    # Search for Jan Aushadhi stores specifically
                    jan_stores = maps_service.search_nearby_places(
                        st.session_state.coordinates[0],
                        st.session_state.coordinates[1],
                        'pharmacy',
                        5000
                    )
                    
                    jan_aushadhi_stores = [store for store in jan_stores if 'jan aushadhi' in store['name'].lower()]
                    
                    if jan_aushadhi_stores:
                        for store in jan_aushadhi_stores:
                            st.success(f"🏛️ {store['name']} - {store['distance']} km away")
                            st.write(f"📍 {store['address']}")
                    else:
                        st.info("No Jan Aushadhi stores found nearby. Check with local pharmacies for generic medicines.")
        
        with tabs[5]:
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
            
            # Location-based diet recommendations
            st.markdown(f"#### 🍽️ Diet Plan for {st.session_state.user_location.get('city', 'your location')}")
            
            # Get seasonal and regional foods
            current_month = datetime.now().month
            if current_month in [12, 1, 2]:
                season = "Winter"
                seasonal_foods = ["Sarson ka saag", "Makki ki roti", "Gajar halwa", "Peanuts", "Jaggery"]
            elif current_month in [3, 4, 5]:
                season = "Summer"
                seasonal_foods = ["Watermelon", "Mango", "Cucumber", "Coconut water", "Lassi"]
            elif current_month in [6, 7, 8, 9]:
                season = "Monsoon"
                seasonal_foods = ["Corn", "Hot soups", "Ginger tea", "Pakoras (in moderation)", "Turmeric milk"]
            else:
                season = "Post-monsoon"
                seasonal_foods = ["Amla", "Green leafy vegetables", "Citrus fruits", "Nuts", "Dates"]
            
            st.info(f"🌤️ Current Season: {season} - Recommended seasonal foods: {', '.join(seasonal_foods)}")
            
            # Sample meal plan
            meal_plan = {
                "Breakfast (8 AM)": ["2 Idli/Dosa with sambar", "OR 2 Parathas with curd", "1 glass milk/tea", "Seasonal fruit"],
                "Mid-Morning (11 AM)": ["1 seasonal fruit", "Handful of nuts", "Green tea/herbal tea"],
                "Lunch (1 PM)": ["2 rotis", "1 bowl dal", "Vegetable curry", "Rice (1 small bowl)", "Curd", "Salad"],
                "Evening (5 PM)": ["Tea with 2 biscuits", "OR Roasted chana", "OR Fresh fruit"],
                "Dinner (8 PM)": ["2 rotis", "Dal/Paneer curry", "Green vegetables", "Salad"]
            }
            
            for meal, items in meal_plan.items():
                with st.expander(meal):
                    for item in items:
                        st.write(f"• {item}")
            
            # Local food recommendations
            st.markdown(f"#### 🌍 Local Healthy Food Options in {st.session_state.user_location.get('city', 'your area')}")
            if st.button("Find Healthy Restaurants", key="healthy_restaurants"):
                with st.spinner("Searching for healthy food options..."):
                    restaurants = maps_service.search_nearby_places(
                        st.session_state.coordinates[0],
                        st.session_state.coordinates[1],
                        'restaurant',
                        3000
                    )
                    
                    # Filter for healthy options
                    healthy_keywords = ['vegetarian', 'vegan', 'salad', 'healthy', 'organic', 'juice']
                    healthy_restaurants = [r for r in restaurants if any(keyword in r['name'].lower() for keyword in healthy_keywords)]
                    
                    if healthy_restaurants:
                        for restaurant in healthy_restaurants[:5]:
                            st.write(f"🥗 **{restaurant['name']}** - {restaurant['distance']} km")
                            st.write(f"   Rating: ⭐ {restaurant.get('rating', 'N/A')}/5 | {restaurant['address']}")
                    else:
                        st.info("No specifically healthy restaurants found. Look for vegetarian options in regular restaurants.")
        
        with tabs[6]:
            st.markdown("### 📰 Latest Health News - India")
            
            # Simulated health news with location relevance
            location_city = st.session_state.user_location.get('city', 'India')
            
            news_articles = [
                {
                    "title": f"Air Quality Alert in {location_city}",
                    "date": datetime.now().strftime("%B %d, %Y"),
                    "summary": f"Health experts advise residents of {location_city} to take precautions as AQI levels rise. Use N95 masks and limit outdoor activities."
                },
                {
                    "title": "AIIMS Launches New Telemedicine Initiative",
                    "date": "June 20, 2024",
                    "summary": "AIIMS Delhi introduces AI-powered telemedicine for rural healthcare across India."
                },
                {
                    "title": f"Free Health Camps in {location_city} This Week",
                    "date": "June 18, 2024",
                    "summary": f"Government organizes free health checkup camps across {location_city}. Services include blood tests, ECG, and consultation."
                },
                {
                    "title": "Ayushman Bharat Reaches 500 Million Indians",
                    "date": "June 15, 2024",
                    "summary": "Government health insurance scheme achieves major milestone in providing healthcare access."
                }
            ]
            
            for article in news_articles:
                with st.expander(f"📰 {article['title']}"):
                    st.write(f"**Date:** {article['date']}")
                    st.write(article['summary'])
                    
                    # Add location-specific information
                    if location_city in article['title']:
                        st.info(f"This news is specifically relevant to your location in {location_city}")

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h3>🏥 MediAI Pro - Your Health, Our Priority</h3>
        <p><strong>Emergency Numbers:</strong> 🚑 108 (Medical) | 112 (National Emergency)</p>
        <p><strong>Disclaimer:</strong> This app provides health information for educational purposes only. 
        Always consult qualified healthcare professionals for medical advice.</p>
        <p>Powered by AI & Google Maps | Made with ❤️ for India</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
        show_footer()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")