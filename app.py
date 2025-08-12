import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
import json
from datetime import datetime
from openai import OpenAI
import sys

# Add model_code to path for importing the transformer predictor
sys.path.append('../model_code')

try:
    from simple_transformer_replay_predictor import get_simple_transformer_replay_predictor
    TRANSFORMER_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Cannot import Transformer model: {e}")
    TRANSFORMER_AVAILABLE = False

APP_DIR = Path(__file__).parent.resolve()

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None
if 'probabilities' not in st.session_state:
    st.session_state['probabilities'] = None
if 'feature_data' not in st.session_state:
    st.session_state['feature_data'] = None
if 'ecg_record_column' not in st.session_state:
    st.session_state['ecg_record_column'] = None
if 'show_risk_analysis' not in st.session_state:
    st.session_state['show_risk_analysis'] = False
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'current_question' not in st.session_state:
    st.session_state['current_question'] = ""
if 'show_ai_answer' not in st.session_state:
    st.session_state['show_ai_answer'] = False

# DeepSeek API Configuration
def setup_deepseek():
    """Setup DeepSeek API"""
    api_key = "sk-e0d5b89c33ec4cd2a8abbe398c9f85e1"
    base_url = "https://api.deepseek.com"
    return OpenAI(api_key=api_key, base_url=base_url)

def get_deepseek_response(user_question, ecg_context, chat_history):
    """Get intelligent response using DeepSeek API"""
    try:
        client = setup_deepseek()
        
        # Build system prompt
        system_prompt = f"""You are a professional Type 2 Diabetes and ECG health advisor named "ecg_advisor". Based on the following ECG risk assessment results, provide users with professional and accurate health advice:
        
ECG Assessment Results:
{ecg_context}

User Question: {user_question}

Please provide:
1. Personalized recommendations based on ECG results
2. Specific action steps
3. Preventive measures
4. Important considerations

Response Requirements:
- Professional, accurate, and easy to understand
- Based on user's specific situation
- Include specific action recommendations
- Use English to answer
- Clear structure, highlight key points
- Answer as "ecg_advisor" identity"""
        
        # Build conversation history
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add recent conversation history (limit length)
        if chat_history:
            recent_history = chat_history[-4:]  # Last 2 rounds of conversation
            for msg in recent_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Call DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Sorry, AI response generation failed: {str(e)}. Please try again later or contact technical support."

# OpenAI API Configuration (keep original functionality)
def setup_openai():
    """Setup OpenAI API"""
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if api_key:
        openai.api_key = api_key
        return True
    return False

def get_ai_response(user_question, ecg_context, chat_history):
    """Get intelligent response using OpenAI API"""
    try:
        if not setup_openai():
            return "Sorry, OpenAI API is not configured, cannot provide AI response. Please check API key settings."
        
        # Build system prompt
        system_prompt = f"""You are a professional Type 2 Diabetes and ECG health advisor. Based on the following ECG risk assessment results, provide users with professional and accurate health advice:

ECG Assessment Results:
{ecg_context}

User Question: {user_question}

Please provide:
1. Personalized recommendations based on ECG results
2. Specific action steps
3. Preventive measures
4. Important considerations

Response Requirements:
- Professional, accurate, and easy to understand
- Based on user's specific situation
- Include specific action recommendations
- Use English to answer
- Clear structure, highlight key points"""
        
        # Build conversation history
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add recent conversation history (limit length)
        if chat_history:
            recent_history = chat_history[-4:]  # Last 2 rounds of conversation
            for msg in recent_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Sorry, AI response generation failed: {str(e)}. Please try again later or contact technical support."

@st.cache_resource(show_spinner=False)
def load_feature_names() -> list:
    feature_file = APP_DIR / 'feature_names.txt'
    with open(feature_file, 'r') as f:
        names = [line.strip() for line in f if line.strip()]
    return names


@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    model_path = APP_DIR / 'RandomForest.joblib'
    scaler_path = APP_DIR / 'scaler.joblib'
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def ensure_dataframe_has_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f'Input data missing columns: {missing}')
    # Reorder according to training order
    return df[feature_names]


def predict_with_model(model, scaler, X: pd.DataFrame, already_scaled: bool) -> tuple:
    if not already_scaled:
        X_scaled = scaler.transform(X.values)
    else:
        X_scaled = X.values
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)
    return y_pred, y_prob


# Transformer model prediction function (keeping the transformer logic)
def predict_with_transformer(feature_data):
    """
    Predict diabetes risk using the Simple Transformer Replay model
    
    Args:
        feature_data: DataFrame containing ECG features
        
    Returns:
        tuple: (predictions, probabilities)
    """
    if not TRANSFORMER_AVAILABLE:
        st.error("❌ Transformer model not available")
        return None, None
    
    try:
        # Initialize the transformer predictor
        predictor = get_simple_transformer_replay_predictor()
        
        if predictor is None:
            st.error("❌ Cannot initialize Transformer predictor")
            return None, None
        
        predictions = []
        probabilities = []
        
        # Process each row
        for idx, row in feature_data.iterrows():
            # Convert row to dictionary format expected by the predictor
            ecg_features = row.to_dict()
            
            # Make prediction
            result = predictor.predict_diabetes_risk({
                'ecg_features': ecg_features
            })
            
            if 'error' in result:
                st.warning(f"Row {idx+1} prediction failed: {result['error']}")
                predictions.append(0)  # Default to normal
                probabilities.append(0.0)
            else:
                # Convert percentage to probability (0-1)
                prob = result['diabetes_risk_probability'] / 100.0
                probabilities.append(prob)
                
                # Determine prediction (threshold at 0.5)
                pred = 1 if prob > 0.5 else 0
                predictions.append(pred)
        
        return np.array(predictions), np.array(probabilities)
        
    except Exception as e:
        st.error(f"❌ Transformer prediction failed: {e}")
        return None, None

def create_risk_dashboard(predictions, probabilities, feature_data):
    """Create risk dashboard"""
    
    # 1. Prediction result statistics
    diabetes_count = sum(predictions)
    normal_count = len(predictions) - diabetes_count
    total_records = len(predictions)
    
    # 2. Risk distribution pie chart - using blue-green color scheme
    fig_pie = px.pie(
        values=[diabetes_count, normal_count],
        names=['Type 2 Diabetes Risk', 'Normal'],
        title='ECG Record Risk Distribution',
        color_discrete_map={'Type 2 Diabetes Risk': '#ff6b6b', 'Normal': '#20b2aa'}
    )
    
    # 3. Risk probability distribution histogram - using blue-green color scheme
    if probabilities is not None:
        fig_hist = px.histogram(
            x=probabilities,
            nbins=20,
            title='Type 2 Diabetes Risk Probability Distribution',
            labels={'x': 'Type 2 Diabetes Probability', 'y': 'Record Count'},
            color_discrete_sequence=['#20b2aa']
        )
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#ff6b6b", annotation_text="Risk Threshold")
    
    # 4. Key feature analysis (keep only specified 9 features)
    key_features = [
        'RR_Interval', 'QT_Interval', 'T_Wave_Peak', 'T_R_ratio',
        'QT_RR_ratio', 'anchor_age', 'HRV_SDNN', 'R_P_ratio', 'QTc_variability'
    ]
    available_features = [f for f in key_features if f in feature_data.columns]
    
    if available_features:
        # Calculate subplot layout
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig_features = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f.replace('anchor_age', 'age') for f in available_features],
            specs=[[{"secondary_y": False}] * n_cols for _ in range(n_rows)]
        )
        
        for i, feature in enumerate(available_features):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            # Display grouped by prediction results
            normal_values = feature_data[feature][predictions == 0]
            risk_values = feature_data[feature][predictions == 1]
            
            if len(normal_values) > 0:
                fig_features.add_trace(
                    go.Box(y=normal_values, name=f'{feature.replace("anchor_age", "age")}-Normal', marker_color='#20b2aa'),
                    row=row, col=col
                )
            if len(risk_values) > 0:
                fig_features.add_trace(
                    go.Box(y=risk_values, name=f'{feature.replace("anchor_age", "age")}-Risk', marker_color='#ff6b6b'),
                    row=row, col=col
                )
        
        fig_features.update_layout(
            height=200 * n_rows, 
            title_text="Key ECG Feature Analysis",
            showlegend=False
        )
    
    return fig_pie, fig_hist if probabilities is not None else None, fig_features if available_features else None


def create_mini_dashboard(predictions, probabilities):
    """Create mini dashboard"""
    diabetes_count = sum(predictions)
    normal_count = len(predictions) - diabetes_count
    total_records = len(predictions)
    
    # Calculate percentages
    diabetes_pct = (diabetes_count / total_records) * 100
    normal_pct = (normal_count / total_records) * 100
    
    # Risk level calculation
    if probabilities is not None:
        high_risk = sum(probabilities > 0.7)
        medium_risk = sum((probabilities > 0.3) & (probabilities <= 0.7))
        low_risk = sum(probabilities <= 0.3)
        
        high_risk_pct = (high_risk / total_records) * 100
        medium_risk_pct = (medium_risk / total_records) * 100
        low_risk_pct = (low_risk / total_records) * 100
    
    # Create dashboard chart
    fig = go.Figure()
    
    # Add pie chart showing main distribution - using blue-green color scheme
    fig.add_trace(go.Pie(
        labels=['Normal', 'Type 2 Diabetes Risk'],
        values=[normal_count, diabetes_count],
        hole=0.6,
        marker_colors=['#20b2aa', '#48cae4'],  # Sea green and light blue
        textinfo='label+percent',
        textposition='inside',
        textfont_size=14
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Risk Assessment Overview',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=300,
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    
    return fig, diabetes_count, normal_count, total_records

def main():
    st.set_page_config(page_title='Type 2 Diabetes ECG Risk Prediction System', layout='wide')
    
    # Center display title and subtitle
    st.markdown("""
    <div class="custom-title-container">
        <h1>Type 2 Diabetes ECG Risk Prediction System</h1>
        <p>ECG Signal Type 2 Diabetes Risk Assessment and Monitoring System Based on Transformer & Replay</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Global status indicator
    if 'predictions' in st.session_state:
        st.markdown("""
        <div style="
            background: linear-gradient(90deg, #adb5bd 0%, #ced4da 100%);
            padding: 10px 20px;
            border-radius: 5px;
            color: #495057;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        ">
            <span style="font-size: 16px; font-weight: bold;">Risk Assessment Results Available</span>
            <span style="font-size: 14px; margin-left: 15px;">Click the tabs above to view detailed analysis</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background: linear-gradient(90deg, #1565c0 0%, #1976d2 100%);
            padding: 10px 20px;
            border-radius: 5px;
            color: white;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        ">
            <span style="font-size: 16px; font-weight: bold;">Please Upload ECG Data First</span>
            <span style="font-size: 14px; margin-left: 15px;">Upload your data in the "Data Upload" tab</span>
        </div>
        """, unsafe_allow_html=True)

    # Check model availability
    if not TRANSFORMER_AVAILABLE:
        st.error("""
        ❌ **Transformer model not available**
        
        Please ensure:
        1. `model_code` directory exists and contains necessary model files
        2. All dependencies are installed
        3. Model file paths are correct
        """)
        return

    feature_names = load_feature_names()
    model, scaler = load_model_and_scaler()

    # Check if need to automatically jump to risk analysis page
    if st.session_state.get('show_risk_analysis', False):
        # Show success message and prompt user to view risk analysis page
        st.success("Risk assessment completed! Please view the 'Risk Analysis' tab above for detailed results.")
        
        # Reset flag to avoid repeated display
        st.session_state['show_risk_analysis'] = False

    # Create four pages
    tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Risk Analysis", "Feature Analysis", "Risk Assessment Advice"])
    
    with tab1:
        st.subheader('Upload Patient ECG Data')
        st.caption('Upload CSV file containing 22 ECG features for diabetes risk assessment')
        
        uploaded_file = st.file_uploader(
            'Select ECG data file (CSV format)', 
            type=['csv'],
            help='File should contain the following feature columns: ' + ', '.join(feature_names)
        )
        
        if uploaded_file is not None:
            try:
                # 读取上传的数据
                patient_data = pd.read_csv(uploaded_file)
                
                # 保留ECG_Record列用于时间序列分析，但确保它不在特征列表中
                ecg_record_column = None
                if 'ECG_Record' in patient_data.columns:
                    ecg_record_column = patient_data['ECG_Record'].copy()
                    # 只删除用于预测的特征列，保留ECG_Record用于显示
                    feature_data_for_prediction = patient_data.drop('ECG_Record', axis=1)
                else:
                    feature_data_for_prediction = patient_data
                
                st.write('**Uploaded Data Preview (showing first 10 records):**')
                
                # Display first 10 records, set height and scroll
                preview_data = patient_data.head(10)
                st.dataframe(
                    preview_data, 
                    use_container_width=True,
                    height=400,  # Set fixed height
                    hide_index=False  # Show row index
                )
                
                # Display data statistics
                st.info(f"Data Overview: Total {len(patient_data)} records, {len(patient_data.columns)} feature columns")
                
                # Check feature columns (excluding ECG_Record column)
                expected_columns = len(feature_names)
                actual_columns = len(feature_data_for_prediction.columns)
                
                if actual_columns != expected_columns:
                    st.warning(f'Data column count mismatch: Expected {expected_columns} columns, Actual {actual_columns} columns')
                    st.write('**Current column names:**', list(feature_data_for_prediction.columns))
                    st.write('**Expected column names:**', feature_names)
                else:
                    st.success(f'Data column count matches: {actual_columns} columns')
                
                # Ensure feature column order is correct
                feature_data_for_prediction = ensure_dataframe_has_features(feature_data_for_prediction, feature_names)
                
                if st.button('Start Risk Assessment', type='primary'):
                    try:
                        with st.spinner('Analyzing ECG data...'):
                            # Use transformer model for prediction
                            y_pred, y_prob = predict_with_transformer(feature_data_for_prediction)
                        
                        # Store results to session state
                        st.session_state['predictions'] = y_pred
                        st.session_state['probabilities'] = y_prob
                        st.session_state['feature_data'] = patient_data  # Keep complete data including ECG_Record
                        st.session_state['ecg_record_column'] = ecg_record_column  # Store ECG record identifier
                        st.session_state['show_risk_analysis'] = True
                        
                        # Show success message
                        st.success("Risk assessment completed!")
                        
                        # Force page rerun
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f'Prediction failed: {e}')
                        
            except Exception as e:
                st.error(f'File reading failed: {e}')

    with tab2:
        st.subheader('Risk Analysis Results')
        
        # Check if risk assessment was just completed
        if st.session_state.get('show_risk_analysis', False):
            st.success("Welcome to view risk assessment results!")
            st.info("Below are the detailed analysis results of your uploaded ECG data.")
            
            # Add quick navigation hint
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #20b2aa 0%, #48cae4 100%);
                    padding: 15px;
                    border-radius: 8px;
                    color: white;
                    text-align: center;
                    margin: 15px 0;
                ">
                    <p style="margin: 0; font-size: 14px;"><strong>Quick Navigation Hint:</strong></p>
                    <p style="margin: 5px 0 0 0; font-size: 12px;">Feature Analysis | Risk Assessment Advice</p>
                </div>
                """, unsafe_allow_html=True)
        
        if 'predictions' in st.session_state and st.session_state['predictions'] is not None:
            predictions = st.session_state['predictions']
            probabilities = st.session_state['probabilities']
            feature_data = st.session_state['feature_data']
            
            # Calculate statistics
            diabetes_count = sum(predictions)
            normal_count = len(predictions) - diabetes_count
            total_records = len(predictions)
            
            # Display mini dashboard
            dashboard_fig, _, _, _ = create_mini_dashboard(predictions, probabilities)
            st.plotly_chart(dashboard_fig, use_container_width=True)
            
            # Display metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Records", total_records)
            with col_b:
                st.metric("High Risk Records", diabetes_count, delta=f"{diabetes_count/total_records*100:.1f}%")
            with col_c:
                st.metric("Normal Records", normal_count, delta=f"{normal_count/total_records*100:.1f}%")
            
            # Risk distribution pie chart
            fig_pie, fig_hist, _ = create_risk_dashboard(predictions, probabilities, feature_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                if fig_hist:
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            # Detailed prediction results table
            st.subheader('Detailed Prediction Results')
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities if probabilities is not None else [None] * len(predictions))):
                risk_level = "Very High Risk" if prob > 0.7 else "High Risk" if prob > 0.5 else "Medium Risk" if prob > 0.3 else "Low Risk" if prob is not None else "N/A"
                results.append({
                    'Record ID': f'Record_{i+1:02d}',
                    'Prediction Result': 'Type 2 Diabetes Risk' if pred == 1 else 'Normal',
                    'Type 2 Diabetes Probability': f'{float(prob):.2%}' if prob is not None else 'N/A',
                    'Risk Level': risk_level,
                    'Recommendation': 'Recommend further examination' if pred == 1 else 'Regular monitoring'
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Add probability calculation explanation
            st.subheader('📊 Probability Calculation Explanation')
            with st.expander("🔍 Detailed explanation of probability calculation", expanded=True):
                st.markdown("""
                **🎯Probability Calculation Principle:**
                
                **1. Model Architecture**
                - This application uses **Transformer + Replay** model for ECG signal analysis
                - Transformer model excels at capturing temporal features and long-term dependencies in ECG signals
                - Replay mechanism enhances the model's memory capacity for historical data
                
                **2. Sigmoid Activation Function**
                - Neural network output is transformed through **Sigmoid function** σ(x) = 1/(1+e^(-x))
                - Converts arbitrary real number output to **probability values between 0-1**
                - 0.0 = 100% certain to be normal, 1.0 = 100% certain to be Type 2 diabetes risk
                
                **3. Risk Level Classification**
                - **Very High Risk**: probability > 0.7 (70%)
                - **High Risk**: probability > 0.5 (50%)
                - **Medium Risk**: probability > 0.3 (30%)
                - **Low Risk**: probability ≤ 0.3 (30%)
                
                **4. Threshold Adjustment**
                - Default risk threshold: **0.5 (50%)**
                - Can be adjusted according to clinical needs
                - Lowering threshold increases sensitivity, raising threshold increases specificity
                
                **5. Model Advantages**
                - **Sequential Modeling**: Captures temporal change patterns in ECG signals
                - **Memory Mechanism**: Replay enhances understanding of historical data
                - **Feature Extraction**: Automatically learns key features of ECG signals
                """)
            
            # Download results
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                'Download Risk Assessment Report',
                csv_data,
                file_name='type2_diabetes_risk_assessment.csv',
                mime='text/csv'
            )
        else:
            st.info("Please upload data and perform risk assessment first")

    with tab3:
        st.subheader('ECG Feature Analysis')
        
        if 'predictions' in st.session_state and st.session_state['predictions'] is not None:
            predictions = st.session_state['predictions']
            probabilities = st.session_state['probabilities']
            feature_data = st.session_state['feature_data']
            
            diabetes_count = sum(predictions)
            normal_count = len(predictions) - diabetes_count
            

            
            # Key ECG feature analysis box plots
            fig_pie, fig_hist, fig_features = create_risk_dashboard(predictions, probabilities, feature_data)
            
            if fig_features:
                st.plotly_chart(fig_features, use_container_width=True)
            
            # Feature change trend chart
            if diabetes_count > 0:  # Has high-risk records, determined as non-normal person
                st.subheader('ECG Feature Change Trends for Type 2 Diabetes Patients')
                
                # Select key features for time series analysis
                time_series_features = [
                    'RR_Interval', 'QT_Interval', 'T_Wave_Peak', 'T_R_ratio',
                    'QT_RR_ratio', 'HRV_SDNN', 'R_P_ratio', 'QTc_variability'
                ]
                available_time_features = [f for f in time_series_features if f in feature_data.columns]
                
                # Set different colors for different features
                feature_colors = {
                    'RR_Interval': '#ff6b6b',      # Red
                    'QT_Interval': '#4ecdc4',      # Cyan
                    'T_Wave_Peak': '#45b7d1',     # Blue
                    'T_R_ratio': '#96ceb4',        # Green
                    'QT_RR_ratio': '#feca57',      # Yellow
                    'HRV_SDNN': '#ff9ff3',         # Pink
                    'R_P_ratio': '#54a0ff',        # Dark Blue
                    'QTc_variability': '#5f27cd'   # Purple
                }
                
                if available_time_features:
                    # Prepare indices of high-risk records
                    diabetes_indices = np.where(predictions == 1)[0]
                    
                    if len(diabetes_indices) == 1:
                        # Single record: radar + bar charts
                        st.subheader('🔍 Single High-Risk ECG Record Feature Analysis')

                        fig_radar = go.Figure()
                        current_record = diabetes_indices[0]
                        feature_values = []
                        feature_names_clean = []
                        for feature in available_time_features:
                            if feature in feature_data.columns:
                                value = feature_data[feature].iloc[current_record]
                                feature_values.append(value)
                                feature_names_clean.append(feature)

                        fig_radar.add_trace(go.Scatterpolar(
                            r=feature_values,
                            theta=feature_names_clean,
                            fill='toself',
                            name='Current Record',
                            line_color='#ff6b6b',
                            fillcolor='rgba(255, 107, 107, 0.3)'
                        ))

                        if normal_count > 0:
                            normal_values = []
                            for feature in available_time_features:
                                if feature in feature_data.columns:
                                    normal_mean = feature_data[feature][predictions == 0].mean()
                                    normal_values.append(normal_mean)

                            fig_radar.add_trace(go.Scatterpolar(
                                r=normal_values,
                                theta=feature_names_clean,
                                fill='toself',
                                name='Normal Mean',
                                line_color='#20b2aa',
                                fillcolor='rgba(32, 178, 170, 0.3)'
                            ))

                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[min(feature_values) - 0.5, max(feature_values) + 0.5])),
                            showlegend=True,
                            title='ECG Feature Radar Chart Comparison'
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                        st.subheader('📈 Feature Value Bar Chart Comparison')
                        fig_bar = go.Figure()
                        fig_bar.add_trace(go.Bar(x=feature_names_clean, y=feature_values, name='Current Record', marker_color='#ff6b6b', opacity=0.8))
                        if normal_count > 0:
                            normal_values = []
                            for feature in available_time_features:
                                if feature in feature_data.columns:
                                    normal_mean = feature_data[feature][predictions == 0].mean()
                                    normal_values.append(normal_mean)
                            fig_bar.add_trace(go.Bar(x=feature_names_clean, y=normal_values, name='Normal Mean', marker_color='#20b2aa', opacity=0.6))
                        fig_bar.update_layout(title='Feature Value Comparison Bar Chart', xaxis_title='ECG Features', yaxis_title='Feature Values', barmode='group', height=500)
                        st.plotly_chart(fig_bar, use_container_width=True)
                        st.info(f'🔍 Detected {len(diabetes_indices)} high-risk records, using radar chart and bar chart for intuitive analysis')

                    elif len(diabetes_indices) > 1:
                        # Multiple records: time series
                        st.subheader('📈 Multiple High-Risk ECG Record Feature Time Series Changes')
                        
                        n_features = len(available_time_features)
                        n_cols = 3
                        n_rows = (n_features + n_cols - 1) // n_cols
                        fig_time_series = make_subplots(
                            rows=n_rows,
                            cols=n_cols,
                            subplot_titles=available_time_features,
                            specs=[[{"secondary_y": False}] * n_cols for _ in range(n_rows)]
                        )
                        
                        for i, feature in enumerate(available_time_features):
                            row = (i // n_cols) + 1
                            col = (i % n_cols) + 1
                            feature_values = feature_data[feature].iloc[diabetes_indices].values
                            if len(feature_values) > 0:
                                ecg_labels = st.session_state.get('ecg_record_column', None)
                                if ecg_labels is not None and len(ecg_labels) == len(feature_values):
                                    x_labels = ecg_labels.iloc[diabetes_indices].tolist()
                                else:
                                    x_labels = list(range(1, len(feature_values) + 1))
                                fig_time_series.add_trace(
                                    go.Scatter(
                                        x=x_labels,
                                        y=feature_values,
                                        mode='lines+markers',
                                        name=f'{feature} (Current Sample)',
                                        line=dict(color=feature_colors.get(feature, '#ff6b6b'), width=2),
                                        marker=dict(size=4),
                                        showlegend=False
                                    ),
                                    row=row, col=col
                                )
                                normal_mean = feature_data[feature][predictions == 0].mean() if normal_count > 0 else 0
                                normal_std = feature_data[feature][predictions == 0].std() if normal_count > 0 else 1
                                fig_time_series.add_hline(
                                    y=normal_mean, 
                                    line_dash='dash',
                                    line_color='#666666',
                                    row=row, col=col,
                                    annotation_text='Normal Mean'
                                )
                                fig_time_series.add_trace(
                                    go.Scatter(
                                        x=[1, len(feature_values)],
                                        y=[normal_mean + normal_std, normal_mean + normal_std],
                                        mode='lines',
                                        line=dict(color='#666666', width=1, dash='dot'),
                                        showlegend=False
                                    ),
                                    row=row, col=col
                                )
                                fig_time_series.add_trace(
                                    go.Scatter(
                                        x=[1, len(feature_values)],
                                        y=[normal_mean - normal_std, normal_mean - normal_std],
                                        mode='lines',
                                        line=dict(color='#666666', width=1, dash='dot'),
                                        showlegend=False
                                    ),
                                    row=row, col=col
                                )
                        
                        fig_time_series.update_layout(
                            title=f'{len(diabetes_indices)} High-Risk ECG Record Feature Time Series Changes vs Normal Value Reference',
                            height=200 * n_rows,
                            showlegend=False
                        )
                        st.plotly_chart(fig_time_series, use_container_width=True)
                        st.info(f'Detected {len(diabetes_indices)} high-risk records, showing feature time series change trends')
                else:
                    st.info('No high-risk records detected')
            else:
                # No high-risk records, determined as normal person
                st.success('No high-risk records, determined as normal person')
                st.info('All ECG records are within normal range, no need to display feature change trend chart')
        else:
            st.info("Please upload data and perform risk assessment first")

    with tab4:
        st.subheader('Risk Assessment Advice - ecg_advisor Intelligent Assistant')
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Initialize risk assessment status
        if 'predictions' in st.session_state and st.session_state['predictions'] is not None:
            predictions = st.session_state['predictions']
            diabetes_count = sum(predictions)
            
            # Display risk assessment summary
            if diabetes_count > 0:
                st.warning(f'Detected {diabetes_count} high-risk ECG records')
                risk_summary = f"Your ECG data shows {diabetes_count} records with Type 2 diabetes risk."
            else:
                st.success('All ECG records show normal')
                risk_summary = "Your ECG data shows all records are within normal range."
            
            # Chat interface
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 15px 25px;
                        border-radius: 15px;
                border: 1px solid #dee2e6;
                margin: 20px 0;
                max-width: 1000px;
                        margin-left: auto;
                margin-right: auto;
            ">
                <h4 style="color: #1565c0; margin-bottom: 10px;">💬 ecg_advisor Intelligent Health Assistant</h4>
                <p style="color: #6c757d; margin-bottom: 10px;">Based on your ECG risk assessment results, I can provide you with personalized health advice. Please tell me what concerns you:</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Preset question buttons - using Streamlit native buttons, centered layout
            st.markdown("""
            <style>
            /* Beautify preset question buttons */
            [data-testid="stButton"] button {
                background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                font-size: 14px !important;
                font-weight: 500 !important;
                padding: 10px 20px !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
                margin: 5px !important;
            }
            
            [data-testid="stButton"] button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
                background: linear-gradient(135deg, #1565d2 0%, #0d47a1 100%) !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("What should I pay attention to?", key="q1",
                              help="Click to get personalized health advice",
                              use_container_width=True):
                    st.session_state.current_question = "What should I pay attention to?"
                    st.session_state.show_ai_answer = True

            with col2:
                if st.button("How to prevent?", key="q2",
                              help="Click to get prevention advice",
                              use_container_width=True):
                    st.session_state.current_question = "How to prevent?"
                    st.session_state.show_ai_answer = True

            with col3:
                if st.button("How often should I recheck?", key="q3",
                              help="Click to get recheck advice",
                              use_container_width=True):
                    st.session_state.current_question = "How often should I recheck?"
                    st.session_state.show_ai_answer = True
            
            # Display AI answer
            if st.session_state.get('show_ai_answer', False) and st.session_state.get('current_question'):
                with st.spinner("ecg_advisor is thinking..."):
                    # Build ECG context information
                    ecg_context = f"""
                    Risk Assessment Results:
                    - Total Records: {len(predictions)}
                    - High Risk Records: {diabetes_count}
                    - Normal Records: {len(predictions) - diabetes_count}
                    - Risk Status: {'High Risk' if diabetes_count > 0 else 'Normal'}
                    """
                    
                    # Call DeepSeek API
                    ai_answer = get_deepseek_response(
                        st.session_state.current_question,
                        ecg_context,
                        st.session_state.chat_history
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "user", "content": st.session_state.current_question})
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})
                    
                    # Display answer
                    st.markdown(f"""
                    <div style="
                        background: #f8f9fa;
                        padding: 15px 25px;
                        border-radius: 15px;
                        border-left: 4px solid #6c757d;
                        margin: 20px 0;
                        max-width: 1000px;
                        margin-left: auto;
                        margin-right: auto;
                    ">
                        <h5 style="color: #495057; margin-bottom: 10px;">🤖 ecg_advisor's answer:</h5>
                        <div style="color: #495057; line-height:1.6;">
                            {ai_answer}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
                    # Reset state
                    st.session_state.show_ai_answer = False
                    st.session_state.current_question = ""
            
            # Custom question input
            st.markdown("---")
            st.markdown("**💭 Or, you can ask your own question:**")
            
            user_question = st.text_input("Please enter your question:", placeholder="For example: What is my risk level? How often should I recheck?")
            
            if st.button("Send Question", type="primary"):
                if user_question.strip():
                    with st.spinner("ecg_advisor is thinking..."):
                        # Build ECG context information
                        ecg_context = f"""
                        Risk Assessment Results:
                        - Total Records: {len(predictions)}
                        - High Risk Records: {diabetes_count}
                        - Normal Records: {len(predictions) - diabetes_count}
                        - Risk Status: {'High Risk' if diabetes_count > 0 else 'Normal'}
                        """
                        
                        # Call DeepSeek API
                        ai_answer = get_deepseek_response(
                            user_question,
                            ecg_context,
                            st.session_state.chat_history
                        )
                        
                        # Add to chat history
                        st.session_state.chat_history.append({"role": "user", "content": user_question})
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})
                        
                        # Display answer
                        st.markdown(f"""
                        <div style="
                            background: #f8f9fa;
                            padding: 15px 25px;
                            border-radius: 15px;
                            border-left: 4px solid #6c757d;
                            margin: 20px 0;
                            max-width: 1000px;
                            margin-left: auto;
                            margin-right: auto;
                        ">
                            <h5 style="color: #495057; margin-bottom: 10px;">🤖 ecg_advisor's answer:</h5>
                            <div style="color: #495057; line-height: 1.6;">
                                {ai_answer}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
                    # Clear input box
                    st.rerun()
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("---")
                st.markdown("**📝 Chat History:**")
                
                for i, message in enumerate(st.session_state.chat_history):
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div style="
                            background: #e3f2fd;
                            padding: 12px 20px;
                            border-radius: 15px;
                            margin: 10px 0;
                            border-left: 4px solid #1565c0;
                            max-width: 1000px;
                            margin-left: auto;
                            margin-right: auto;
                        ">
                            <strong>You:</strong> {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="
                            background: #f8f9fa;
                            padding: 12px 20px;
                            border-radius: 15px;
                            margin: 10px 0;
                            border-left: 4px solid #6c757d;
                            max-width: 1000px;
                            margin-left: auto;
                            margin-right: auto;
                        ">
                            <strong>ecg_advisor:</strong> {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Clear chat history button
                if st.button("Clear Chat History", key="clear_chat"):
                    st.session_state.chat_history = []
                    st.rerun()
        else:
            st.info("Please upload data and perform risk assessment first")

    # Add CSS styling
    st.markdown("""
    <style>
    /* Global background setting */
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 50%, #90caf9 100%);
    }
    
    /* Main container background */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 25px;
        margin: 15px;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Tab container background */
    [data-testid="stTabs"] {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    /* Tab button styles */
    [data-testid="stTabs"] button {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
        margin: 3px !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stTabs"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Active tab style */
    [data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #1565c0 !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(21, 101, 192, 0.4) !important;
        border-color: #1565c0 !important;
    }
    
    /* Custom title container style */
    .custom-title-container {
        text-align: center !important;
        margin: 15px 0 25px 0 !important;
        padding: 15px !important;
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .custom-title-container h1 {
        margin: 0 0 12px 0 !important;
        font-size: 2.2em !important;
        font-weight: bold !important;
        background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    
    .custom-title-container p {
        margin: 0 !important;
        font-size: 1.1em !important;
        color: #1976d2 !important;
        font-weight: 500 !important;
        opacity: 0.9 !important;
    }
    
    /* Button style optimization */
    .stButton > button {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(25, 118, 210, 0.4) !important;
    }
    
    /* File uploader style */
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
        padding: 15px !important;
        border: 2px dashed #1976d2 !important;
    }
    
    /* Data frame style */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Scrollbar style */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(25, 118, 210, 0.6);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(25, 118, 210, 0.8);
    }
    </style>
    """, unsafe_allow_html=True)

# Main program entry point
if __name__ == '__main__':
    main()
