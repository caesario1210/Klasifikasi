'''
Materi Dr. Eng. Farrikh Alzami, M.Kom - Universitas Dian Nuswantoro
'''
import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Client Subscribed Prediction App ",
    page_icon="🔔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import io

# Load model components
@st.cache_resource
def load_model():
    """Load the trained model components"""
    try:
        components = joblib.load('_034411_dt_tuned_comps.joblib')
        return components
    except FileNotFoundError:
        st.error("Model file '_034411_dt_tuned_comps.joblib' not found!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def predict_y(data, model_components):
    """Make y predictions using the trained model"""
    # Convert to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()
    
    # Get components
    model = model_components['model']
    encoding_maps = model_components['encoding_maps']
    feature_names = model_components['feature_names']
    
    # Apply encodings to categorical columns
    for column in df.columns:
        if column in encoding_maps and column != 'y':
            df[column] = df[column].map(encoding_maps[column])
        if df[column].isnull().any():
            st.error(f"Unknown category found in column '{column}'. Please check your input.")
            st.stop()
    
    # Ensure we only use features that the model was trained on
    df_for_pred = df[feature_names].copy()
    
    # Make prediction
    prediction = model.predict(df_for_pred)[0]
    probabilities = model.predict_proba(df_for_pred)[0]
    
    # Get y label
    y_map_inverse = {v: k for k, v in encoding_maps['y'].items()}
    prediction_label = y_map_inverse[prediction]
    
    return {
        'prediction': int(prediction),
        'prediction_label': prediction_label,
        'probability': float(probabilities[prediction]),
        'probabilities': probabilities.tolist()
    }

def validate_inputs(data):
    """Validate input data"""
    errors = []
    
    # Age validation
    if data['age'] < 19 or data['age'] > 95:
        errors.append("Age should be between 19 and 95")
    
    # Education number validation
    if data['balance'] < -8019 or data['balance'] > 102127:
        errors.append("Average yearly balance should be between -8019 and 102127")

    # Capital gain/loss validation
    if data['duration'] < 0 or data['duration'] > 4918:
        errors.append("Capital gain should be between 0 and 4918")
    
    if data['campaign'] < 1 or data['campaign'] > 63:
        errors.append("Capital loss should be between 1 and 63")
    
    # Final weight validation
    if data['pdays'] < -1 or data['pdays'] > 871:
        errors.append("Final weight should be between -1 and 871")

    if input_data['previous'] < 0 or input_data['previous'] > 275:
        st.warning("⚠️ Invalid 'previous' value. It should be between 0 and 275.")
    
    return errors

def export_prediction(data, result):
    """Export prediction result to JSON"""
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'input_data': data,
        'prediction': {
            'class': result['prediction_label'],
            'confidence': result['probability'],
            'raw_prediction': result['prediction']
        }
    }
    return json.dumps(export_data, indent=2)

def export_prediction_csv(data, result):
    """Export prediction result to CSV"""
    output = io.StringIO()
    # Gabungkan input dan hasil prediksi ke satu dictionary
    export_dict = {**data, **{
        'prediction_class': result['prediction_label'],
        'confidence': result['probability'],
        'raw_prediction': result['prediction']
    }}
    # Buat DataFrame satu baris
    df = pd.DataFrame([export_dict])
    df.to_csv(output, index=False)
    return output.getvalue()

def reset_session_state():
    """Reset all input values to default"""
    keys_to_reset = [
    'age', 'job', 'marital', 'education', 'default', 'balance', 
    'housing', 'loan', 'contact', 'day', 'month', 'duration', 
    'campaign', 'pdays', 'previous', 'poutcome' 
    # 'y' adalah target, bukan input, jadi tidak perlu direset
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

# Load model
model_components = load_model()

# Mappings for categorical columns based on image_8993d9.png

job_options = ['management', 'technician', 'entrepreneur', 'blue-collar', 'unknown',
 'retired', 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid',
 'student']


marital_options = ['married', 'single', 'divorced']

education_options = ['tertiary', 'secondary', 'unknown', 'primary']

default_options = ['no', 'yes'] # Column 'default' unique values

housing_options = ['yes', 'no'] # Column 'housing' unique values

loan_options = ['no', 'yes'] # Column 'loan' unique values

contact_options = ['unknown', 'cellular', 'telephone'] # Column 'contact' unique values

day_options = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'
]

month_options = [ 'jan', 'feb', 'mar', 'apr','may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'] # Column 'month' unique values

poutcome_options = ['unknown', 'failure', 'other', 'success'] # Column 'poutcome' unique values

y_options = ['no', 'yes'] # Column 'y' unique values

# Main app
st.title("🔔 Client Subscribed Prediction App - Kelompok 10")
st.markdown("📢 Predict Client Subscription or Not Based on Provided Data")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Input Features")
    
    # Create form for inputs
    with st.form("prediction_form"):
        # Demographic Information
        st.markdown("**Demographic Information**")
        col_demo1, col_demo2 = st.columns(2)
        
        with col_demo1:
            age = st.number_input("Age", min_value=17, max_value=95, value=39, key="age")
            marital = st.selectbox("Marital", marital_options, key="marital")
            education = st.selectbox("Education", education_options, key="education")
        
        # with col_demo2:
        #     marital_status = st.selectbox("Marital Status", marital_status_options, key="marital_status")
        #     relationship = st.selectbox("Relationship", relationship_options, key="relationship")
        #     native_country = st.selectbox("Native Country", native_country_options, key="native_country")
        
        st.divider()
        
        # Work Information
        st.markdown("**Work Information**")
        col_work1, col_work2 = st.columns(2)
        
        with col_work1:
            job = st.selectbox("Job", job_options, key="job")
            default = st.selectbox("Default", default_options, key="default")
            balance = st.number_input("Balance", min_value=-8019, max_value=102127, value=1000, key="balance")
        
        with col_work2:
            housing = st.selectbox("Housing", housing_options, key="housing")
            loan = st.selectbox("Loan", loan_options, key="loan")
            
        
        st.divider()
        
        # Financial Information
        st.markdown("**Financial Information**")
        col_fin1, col_fin2, col_fin3, col_fin4 = st.columns(4)
        
        with col_fin1:
            contact = st.selectbox("Contact Type", contact_options, key="contact") # Key corrected
            day = st.selectbox("Last Contact Day (of week)", day_options, key="day") # Key corrected
        
        with col_fin2:
            month = st.selectbox("Last Contact Month", month_options, key="month") # Key corrected
            duration = st.number_input("Last Contact Duration (seconds)", min_value=0, max_value=5000, value=100, key="duration", 
                                      help="Beware: Duration is known after the call ends, so it's a strong predictor.") # Key corrected, help added
        
        with col_fin3:
            campaign = st.number_input("Contacts during this campaign", min_value=1, max_value=65, value=1, key="campaign") # Key corrected
            pdays = st.number_input("Days since previous contact)", min_value=-1, max_value=999, value=-1, key="pdays") # Key corrected
            
        with col_fin4:
            
            previous = st.number_input("Contacts before this campaign", min_value=0, max_value=275, value=0, key="previous") # Key corrected
            poutcome = st.selectbox("Outcome of previous campaign", poutcome_options, key="poutcome") # Key corrected
        
        # Buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            predict_button = st.form_submit_button("🔮 Predict", type="primary")
        with col_btn2:
            reset_button = st.form_submit_button("🔄 Reset")
        with col_btn3:
            export_button = st.form_submit_button("📤 Export Last Result")

# Handle reset button
if reset_button:
    reset_session_state()
    st.rerun()

# Handle prediction
if predict_button:
    # Collect input data
    input_data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day, # Pastikan ini konsisten dengan nama kolom di dataset Anda
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }
    
    # Validate inputs
    validation_errors = validate_inputs(input_data)
    
    if validation_errors:
        with col2:
            st.error("❌ Validation Errors:")
            for error in validation_errors:
                st.error(f"• {error}")
    else:
        # Make prediction
        try:
            result = predict_y(input_data, model_components)
            
            # Store result in session state for export
            st.session_state['last_prediction'] = {
                'input_data': input_data,
                'result': result
            }
            
            with col2:
                st.subheader("🎯 Prediction Results")
                
                # Display prediction
                prediction_color = "green" if result['prediction_label'] == '1' else "orange"
                st.markdown(f"**Predicted Subscribe:** :{prediction_color}[{result['prediction_label']}]")
                
                # Confidence level with gauge
                confidence = result['probability'] * 100
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = confidence,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Level (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': prediction_color},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Probability breakdown
                prob_df = pd.DataFrame({
                    'Class': ['0', '1'],
                    'Probability': result['probabilities']
                })
                
                fig_bar = px.bar(
                    prob_df, 
                    x='Class', 
                    y='Probability',
                    title='Probability Distribution',
                    color='Probability',
                    color_continuous_scale=['orange', 'green']
                )
                fig_bar.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)
                
        except Exception as e:
            with col2:
                st.error(f"❌ Prediction Error: {str(e)}")

# Feature Importance section
st.subheader("📊 Feature Importance")

if 'model' in model_components:
    try:
        feature_names = model_components['feature_names']
        feature_importance = model_components['model'].feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Feature Importance in Decision Tree Model',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_importance, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying feature importance: {str(e)}")

# Handle export
if export_button:
    if 'last_prediction' in st.session_state:
        export_csv = export_prediction_csv(
            st.session_state['last_prediction']['input_data'],
            st.session_state['last_prediction']['result']
        )
        st.success("Hasil prediksi siap diunduh sebagai CSV.")
        st.download_button(
            label="📥 Download Prediction Results (CSV)",
            data=export_csv,
            file_name=f"y_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No prediction results to export. Please make a prediction first.")

# Jangan tampilkan seluruh session state ke user
# st.write(st.session_state)

# Footer
st.markdown("---")
st.markdown("<p style='text-align;'><em>Built with Streamlit • GRUB 10</em></p>", unsafe_allow_html=True)

st.markdown("""
1. **Ridho Farizqi (A12.2022.06867)**
2. **Dhea Maharani (A12.2022.06900)**
3. **Caesario Gumilang F (A12.2022.06910)**  
""")

st.markdown("<p style='text-align:center;'><a href='https://www.linkedin.com/in/caesariofirdaus' target='_blank'>LinkedIn</a></p>", unsafe_allow_html=True)