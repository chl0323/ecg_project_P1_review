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


APP_DIR = Path(__file__).parent.resolve()

# DeepSeek API配置
def setup_deepseek():
    """设置DeepSeek API"""
    api_key = "sk-e0d5b89c33ec4cd2a8abbe398c9f85e1"
    base_url = "https://api.deepseek.com"
    return OpenAI(api_key=api_key, base_url=base_url)

def get_deepseek_response(user_question, ecg_context, chat_history):
    """使用DeepSeek API获取智能回答"""
    try:
        client = setup_deepseek()
        
        # 构建系统提示词
        system_prompt = f"""你是一位专业的糖尿病和ECG健康顾问，名为"ecg_advisor"。基于以下ECG风险评估结果，为用户提供专业、准确的健康建议：

ECG评估结果：
{ecg_context}

用户问题：{user_question}

请提供：
1. 基于ECG结果的个性化建议
2. 具体的行动步骤
3. 预防措施
4. 注意事项

回答要求：
- 专业、准确、易懂
- 基于用户的具体情况
- 包含具体的行动建议
- 使用中文回答
- 结构清晰，重点突出
- 以"ecg_advisor"的身份回答"""
        
        # 构建对话历史
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 添加最近的对话历史（限制长度）
        if chat_history:
            recent_history = chat_history[-4:]  # 最近2轮对话
            for msg in recent_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 调用DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"抱歉，AI回答生成失败：{str(e)}。请稍后重试或联系技术支持。"

# OpenAI API配置（保留原有功能）
def setup_openai():
    """设置OpenAI API"""
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if api_key:
        openai.api_key = api_key
        return True
    return False

def get_ai_response(user_question, ecg_context, chat_history):
    """使用OpenAI API获取智能回答"""
    try:
        if not setup_openai():
            return "抱歉，OpenAI API未配置，无法提供AI回答。请检查API密钥设置。"
        
        # 构建系统提示词
        system_prompt = f"""你是一位专业的糖尿病和ECG健康顾问。基于以下ECG风险评估结果，为用户提供专业、准确的健康建议：

ECG评估结果：
{ecg_context}

用户问题：{user_question}

请提供：
1. 基于ECG结果的个性化建议
2. 具体的行动步骤
3. 预防措施
4. 注意事项

回答要求：
- 专业、准确、易懂
- 基于用户的具体情况
- 包含具体的行动建议
- 使用中文回答
- 结构清晰，重点突出"""
        
        # 构建对话历史
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 添加最近的对话历史（限制长度）
        if chat_history:
            recent_history = chat_history[-4:]  # 最近2轮对话
            for msg in recent_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 调用OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"抱歉，AI回答生成失败：{str(e)}。请稍后重试或联系技术支持。"


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
        raise ValueError(f'输入数据缺少列: {missing}')
    # 按训练时顺序重排
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


def create_risk_dashboard(predictions, probabilities, feature_data):
    """创建风险dashboard"""
    
    # 1. 预测结果统计
    diabetes_count = sum(predictions)
    normal_count = len(predictions) - diabetes_count
    total_records = len(predictions)
    
    # 2. 风险分布饼图 - 使用蓝绿色系
    fig_pie = px.pie(
        values=[diabetes_count, normal_count],
        names=['糖尿病风险', '正常'],
        title='ECG记录风险分布',
        color_discrete_map={'糖尿病风险': '#ff6b6b', '正常': '#20b2aa'}
    )
    
    # 3. 风险概率分布直方图 - 使用蓝绿色系
    if probabilities is not None:
        fig_hist = px.histogram(
            x=probabilities,
            nbins=20,
            title='糖尿病风险概率分布',
            labels={'x': '糖尿病概率', 'y': '记录数量'},
            color_discrete_sequence=['#20b2aa']
        )
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#ff6b6b", annotation_text="风险阈值")
    
    # 4. 关键特征分析（只保留指定的9个特征）
    key_features = [
        'RR_Interval', 'QT_Interval', 'T_Wave_Peak', 'T_R_ratio',
        'QT_RR_ratio', 'anchor_age', 'HRV_SDNN', 'R_P_ratio', 'QTc_variability'
    ]
    available_features = [f for f in key_features if f in feature_data.columns]
    
    if available_features:
        # 计算子图布局
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
            
            # 按预测结果分组显示
            normal_values = feature_data[feature][predictions == 0]
            risk_values = feature_data[feature][predictions == 1]
            
            if len(normal_values) > 0:
                fig_features.add_trace(
                    go.Box(y=normal_values, name=f'{feature.replace("anchor_age", "age")}-正常', marker_color='#20b2aa'),
                    row=row, col=col
                )
            if len(risk_values) > 0:
                fig_features.add_trace(
                    go.Box(y=risk_values, name=f'{feature.replace("anchor_age", "age")}-风险', marker_color='#ff6b6b'),
                    row=row, col=col
                )
        
        fig_features.update_layout(
            height=200 * n_rows, 
            title_text="关键ECG特征分析",
            showlegend=False
        )
    
    return fig_pie, fig_hist if probabilities is not None else None, fig_features if available_features else None


def create_mini_dashboard(predictions, probabilities):
    """创建迷你仪表盘"""
    diabetes_count = sum(predictions)
    normal_count = len(predictions) - diabetes_count
    total_records = len(predictions)
    
    # 计算百分比
    diabetes_pct = (diabetes_count / total_records) * 100
    normal_pct = (normal_count / total_records) * 100
    
    # 风险等级计算
    if probabilities is not None:
        high_risk = sum(probabilities > 0.7)
        medium_risk = sum((probabilities > 0.3) & (probabilities <= 0.7))
        low_risk = sum(probabilities <= 0.3)
        
        high_risk_pct = (high_risk / total_records) * 100
        medium_risk_pct = (medium_risk / total_records) * 100
        low_risk_pct = (low_risk / total_records) * 100
    
    # 创建仪表盘图表
    fig = go.Figure()
    
    # 添加环形图显示主要分布 - 使用蓝绿色系
    fig.add_trace(go.Pie(
        labels=['正常', '糖尿病风险'],
        values=[normal_count, diabetes_count],
        hole=0.6,
        marker_colors=['#20b2aa', '#48cae4'],  # 海绿色和浅蓝色
        textinfo='label+percent',
        textposition='inside',
        textfont_size=14
    ))
    
    # 更新布局
    fig.update_layout(
        title={
            'text': '风险评估概览',
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
    st.set_page_config(page_title='糖尿病ECG风险预测系统', layout='wide')
    
    # 居中显示标题和副标题
    st.markdown("""
    <div class="custom-title-container">
        <h1>糖尿病ECG风险预测系统</h1>
        <p>基于transformer&replay的ECG信号糖尿病风险评估与监控系统</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 全局状态指示器
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
            <span style="font-size: 16px; font-weight: bold;">风险评估结果可用</span>
            <span style="font-size: 14px; margin-left: 15px;">点击上方标签页查看详细分析</span>
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
            <span style="font-size: 16px; font-weight: bold;">请先上传ECG数据</span>
            <span style="font-size: 14px; margin-left: 15px;">在"数据上传"标签页中上传您的数据</span>
        </div>
        """, unsafe_allow_html=True)

    feature_names = load_feature_names()
    model, scaler = load_model_and_scaler()

    # 创建三个页面
    tab1, tab2, tab3, tab4 = st.tabs(["数据上传", "风险分析", "特征分析", "风险评估建议"])
    
    # 设置每个标签页的浅蓝色背景
    st.markdown("""
    <style>
    /* 全局背景设置 - 使用更可靠的选择器 */
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 50%, #90caf9 100%);
        background-attachment: fixed;
        background-size: cover;
    }
    
    /* 主容器背景 */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 25px;
        margin: 15px;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* 标签页容器背景 */
    [data-testid="stTabs"] {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    /* 标签页按钮样式 */
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
    
    /* 激活的标签页样式 */
    [data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #1565c0 !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(21, 101, 192, 0.4) !important;
        border-color: #1565c0 !important;
    }
    
    /* 标签页下划线样式 - 深蓝色 */
    [data-testid="stTabs"] button[aria-selected="true"]::after {
        content: '' !important;
        position: absolute !important;
        bottom: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 3px !important;
        background-color: #0d47a1 !important;
        border-radius: 2px 2px 0 0 !important;
        z-index: 1000 !important;
    }
    
    /* 强制覆盖Streamlit默认样式 */
    [data-testid="stTabs"] button[aria-selected="true"] {
        border-bottom: 3px solid #0d47a1 !important;
    }
    
    /* 标题和副标题样式 */
    h1, h2, h3 {
        color: #1565c0 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    /* 自定义标题容器样式 */
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
    
    /* 按钮样式优化 */
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
    
    /* 文件上传器样式 */
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
        padding: 15px !important;
        border: 2px dashed #1976d2 !important;
    }
    
    /* 数据框样式 */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* 数据表格滚动条样式 */
    .stDataFrame .stDataFrame {
        max-height: 400px !important;
        overflow-y: auto !important;
    }
    
    /* 表格行悬停效果 */
    .stDataFrame tbody tr:hover {
        background-color: rgba(25, 118, 210, 0.1) !important;
    }
    
    /* 表格头部样式 */
    .stDataFrame thead th {
        background-color: rgba(25, 118, 210, 0.1) !important;
        color: #1565c0 !important;
        font-weight: bold !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 10 !important;
    }
    
    /* 成功和错误消息样式 */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* 滚动条样式 */
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
    
    # 检查是否需要自动跳转到风险分析页面
    if st.session_state.get('show_risk_analysis', False):
        # 显示成功消息并提示用户查看风险分析页面
        st.success("风险评估完成！请查看上方'风险分析'标签页查看详细结果。")
        
        # 重置标志，避免重复显示
        st.session_state['show_risk_analysis'] = False
    
    with tab1:
        st.subheader('上传病人ECG数据')
        st.caption('上传包含22个ECG特征的CSV文件进行糖尿病风险评估')
        
        uploaded_file = st.file_uploader(
            '选择ECG数据文件 (CSV格式)', 
            type=['csv'],
            help='文件应包含以下特征列: ' + ', '.join(feature_names)
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
                
                st.write('**上传数据预览 (显示前10条记录):**')
                
                # 显示前10条记录，设置高度和滚动
                preview_data = patient_data.head(10)
                st.dataframe(
                    preview_data, 
                    use_container_width=True,
                    height=400,  # 设置固定高度
                    hide_index=False  # 显示行索引
                )
                
                # 显示数据统计信息
                st.info(f"数据概览: 总共 {len(patient_data)} 条记录，{len(patient_data.columns)} 个特征列")
                
                # 检查特征列（不包括ECG_Record列）
                expected_columns = len(feature_names)
                actual_columns = len(feature_data_for_prediction.columns)
                
                if actual_columns != expected_columns:
                    st.warning(f'数据列数不匹配: 期望 {expected_columns} 列, 实际 {actual_columns} 列')
                    st.write('**当前列名:**', list(feature_data_for_prediction.columns))
                    st.write('**期望列名:**', feature_names)
                else:
                    st.success(f'数据列数匹配: {actual_columns} 列')
                
                # 确保特征列顺序正确
                feature_data_for_prediction = ensure_dataframe_has_features(feature_data_for_prediction, feature_names)
                
                if st.button('开始风险评估', type='primary'):
                    try:
                        with st.spinner('正在分析ECG数据...'):
                            y_pred, y_prob = predict_with_model(model, scaler, feature_data_for_prediction, already_scaled=False)
                        
                        # 存储结果到session state
                        st.session_state['predictions'] = y_pred
                        st.session_state['probabilities'] = y_prob
                        st.session_state['feature_data'] = patient_data  # 保留完整数据包括ECG_Record
                        st.session_state['ecg_record_column'] = ecg_record_column  # 存储ECG记录标识
                        st.session_state['show_risk_analysis'] = True
                        
                        # 显示成功消息
                        st.success("风险评估完成！")
                        
                        # 强制重新运行页面
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f'预测失败: {e}')
                        
            except Exception as e:
                st.error(f'文件读取失败: {e}')
    
    with tab2:
        st.subheader('风险分析结果')
        
        # 检查是否刚完成风险评估
        if st.session_state.get('show_risk_analysis', False):
            st.success("欢迎查看风险评估结果！")
            st.info("以下是您上传的ECG数据的详细分析结果。")
            
            # 添加快速导航提示
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
                    <p style="margin: 0; font-size: 14px;"><strong>快速导航提示：</strong></p>
                    <p style="margin: 5px 0 0 0; font-size: 12px;">特征分析 | 风险评估建议</p>
                </div>
                """, unsafe_allow_html=True)
        
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            probabilities = st.session_state['probabilities']
            feature_data = st.session_state['feature_data']
            
            # 计算统计信息
            diabetes_count = sum(predictions)
            normal_count = len(predictions) - diabetes_count
            total_records = len(predictions)
            
            # 显示迷你仪表盘
            dashboard_fig, _, _, _ = create_mini_dashboard(predictions, probabilities)
            st.plotly_chart(dashboard_fig, use_container_width=True)
            
            # 显示指标
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("总记录数", total_records)
            with col_b:
                st.metric("高风险记录", diabetes_count, delta=f"{diabetes_count/total_records*100:.1f}%")
            with col_c:
                st.metric("正常记录", normal_count, delta=f"{normal_count/total_records*100:.1f}%")
            
            # 风险分布饼图
            fig_pie, fig_hist, _ = create_risk_dashboard(predictions, probabilities, feature_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                if fig_hist:
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            # 详细预测结果表格
            st.subheader('详细预测结果')
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities if probabilities is not None else [None] * len(predictions))):
                risk_level = "极高风险" if prob > 0.7 else "高风险" if prob > 0.5 else "中等风险" if prob > 0.3 else "低风险" if prob is not None else "N/A"
                results.append({
                    '记录编号': f'Record_{i+1:02d}',
                    '预测结果': '糖尿病风险' if pred == 1 else '正常',
                    '糖尿病概率': f'{float(prob):.2%}' if prob is not None else 'N/A',
                    '风险等级': risk_level,
                    '建议': '建议进一步检查' if pred == 1 else '定期监测'
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # 添加概率计算说明
            st.subheader('📊 概率计算说明')
            with st.expander("🔍 关于概率计算的详细说明", expanded=True):
                st.markdown("""
                **🎯 概率计算原理：**
                
                **1. 模型架构**
                - 本应用使用 **Transformer + Replay** 模型进行ECG信号分析
                - Transformer模型擅长捕捉ECG信号中的时序特征和长期依赖关系
                - Replay机制增强了模型对历史数据的记忆能力
                
                **2. Sigmoid激活函数**
                - 神经网络输出经过 **Sigmoid函数** σ(x) = 1/(1+e^(-x)) 转换
                - 将任意实数输出转换为 **0-1之间的概率值**
                - 0.0 = 100%确定是正常，1.0 = 100%确定是糖尿病风险
                
                **3. 风险等级划分**
                - **极高风险**：概率 > 0.7 (70%)
                - **高风险**：概率 > 0.5 (50%)
                - **中等风险**：概率 > 0.3 (30%)
                - **低风险**：概率 ≤ 0.3 (30%)
                
                **4. 阈值调整**
                - 默认风险阈值：**0.5 (50%)**
                - 可根据临床需求调整阈值
                - 降低阈值会增加敏感性，提高阈值会增加特异性
                
                **5. 模型优势**
                - **序列建模**：捕捉ECG信号的时间变化模式
                - **记忆机制**：Replay增强对历史数据的理解
                - **特征提取**：自动学习ECG信号的关键特征
                """)
            
            # 下载结果
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                '下载风险评估报告',
                csv_data,
                file_name='diabetes_risk_assessment.csv',
                mime='text/csv'
            )
        else:
            st.info("请先上传数据并进行风险评估")
    
    with tab3:
        st.subheader('ECG特征分析')
        
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            probabilities = st.session_state['probabilities']
            feature_data = st.session_state['feature_data']
            
            diabetes_count = sum(predictions)
            normal_count = len(predictions) - diabetes_count
            

            
            # 关键ECG特征分析箱线图
            fig_pie, fig_hist, fig_features = create_risk_dashboard(predictions, probabilities, feature_data)
            
            if fig_features:
                st.plotly_chart(fig_features, use_container_width=True)
            
            # 特征变化趋势图
            if diabetes_count > 0:  # 有高风险记录，判定为非正常人
                st.subheader('糖尿病患者每次ECG记录特征变化趋势')
                
                # 选择关键特征进行时间序列分析
                time_series_features = [
                    'RR_Interval', 'QT_Interval', 'T_Wave_Peak', 'T_R_ratio',
                    'QT_RR_ratio', 'HRV_SDNN', 'R_P_ratio', 'QTc_variability'
                ]
                available_time_features = [f for f in time_series_features if f in feature_data.columns]
                
                # 为不同特征设置不同颜色
                feature_colors = {
                    'RR_Interval': '#ff6b6b',      # 红色
                    'QT_Interval': '#4ecdc4',      # 青色
                    'T_Wave_Peak': '#45b7d1',     # 蓝色
                    'T_R_ratio': '#96ceb4',        # 绿色
                    'QT_RR_ratio': '#feca57',      # 黄色
                    'HRV_SDNN': '#ff9ff3',         # 粉色
                    'R_P_ratio': '#54a0ff',        # 深蓝色
                    'QTc_variability': '#5f27cd'   # 紫色
                }
                
                if available_time_features:
                    # 检查是否有足够的数据点来画折线图
                    diabetes_indices = np.where(predictions == 1)[0]
                    
                    # 只要有高风险记录就画图，不管是一条还是多条
                    if len(diabetes_indices) >= 1:
                        if len(diabetes_indices) == 1:
                            # 单条记录：使用雷达图和特征值对比表，更直观易懂
                            st.subheader('🔍 单条高风险ECG记录特征分析')
                            
                            # 1. 创建雷达图
                            fig_radar = go.Figure()
                            
                            # 获取当前记录的特征值
                            current_record = diabetes_indices[0]
                            feature_values = []
                            feature_names_clean = []
                            
                            for feature in available_time_features:
                                if feature in feature_data.columns:
                                    value = feature_data[feature].iloc[current_record]
                                    feature_values.append(value)
                                    feature_names_clean.append(feature)
                            
                            # 添加雷达图
                            fig_radar.add_trace(go.Scatterpolar(
                                r=feature_values,
                                theta=feature_names_clean,
                                fill='toself',
                                name='当前记录',
                                line_color='#ff6b6b',
                                fillcolor='rgba(255, 107, 107, 0.3)'
                            ))
                            
                            # 添加正常值参考（如果有正常记录）
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
                                    name='正常均值',
                                    line_color='#20b2aa',
                                    fillcolor='rgba(32, 178, 170, 0.3)'
                                ))
                            
                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[min(feature_values) - 0.5, max(feature_values) + 0.5]
                                    )),
                                showlegend=True,
                                title='ECG特征雷达图对比'
                            )
                            
                            st.plotly_chart(fig_radar, use_container_width=True)
                            

                            
                            # 3. 创建条形图对比
                            st.subheader('📈 特征值条形图对比')
                            
                            fig_bar = go.Figure()
                            
                            # 当前值条形图
                            fig_bar.add_trace(go.Bar(
                                x=feature_names_clean,
                                y=feature_values,
                                name='当前记录',
                                marker_color='#ff6b6b',
                                opacity=0.8
                            ))
                            
                            # 正常值条形图
                            if normal_count > 0:
                                normal_values = []
                                for feature in available_time_features:
                                    if feature in feature_data.columns:
                                        normal_mean = feature_data[feature][predictions == 0].mean()
                                        normal_values.append(normal_mean)
                                
                                fig_bar.add_trace(go.Bar(
                                    x=feature_names_clean,
                                    y=normal_values,
                                    name='正常均值',
                                    marker_color='#20b2aa',
                                    opacity=0.6
                                ))
                            
                            fig_bar.update_layout(
                                title='特征值对比条形图',
                                xaxis_title='ECG特征',
                                yaxis_title='特征值',
                                barmode='group',
                                height=500
                            )
                            
                            st.plotly_chart(fig_bar, use_container_width=True)
                            
                            st.info(f'🔍 检测到{len(diabetes_indices)}条高风险记录，使用雷达图、对比表和条形图进行直观分析')
                            
                        else:
                            # 多条记录：使用传统的时间序列折线图
                            st.subheader('📈 多条高风险ECG记录特征时间序列变化')
                            
                            # 创建子图布局
                            n_features = len(available_time_features)
                            n_cols = 3
                            n_rows = (n_features + n_cols - 1) // n_cols
                            
                            fig_time_series = make_subplots(
                                rows=n_rows, cols=n_cols,
                                subplot_titles=available_time_features,
                                specs=[[{"secondary_y": False}] * n_cols for _ in range(n_rows)]
                            )
                            
                            for i, feature in enumerate(available_time_features):
                                row = (i // n_cols) + 1
                                col = (i % n_cols) + 1
                                
                                feature_values = feature_data[feature].iloc[diabetes_indices].values
                                
                                if len(feature_values) > 0:
                                    # 获取ECG记录标识（如果存在）
                                    ecg_labels = st.session_state.get('ecg_record_column', None)
                                    if ecg_labels is not None and len(ecg_labels) == len(feature_values):
                                        x_labels = ecg_labels.iloc[diabetes_indices].tolist()
                                    else:
                                        x_labels = list(range(1, len(feature_values) + 1))
                                    
                                    # 多点显示折线
                                    fig_time_series.add_trace(
                                        go.Scatter(
                                            x=x_labels,
                                            y=feature_values,
                                            mode='lines+markers',
                                            name=f'{feature} (当前样本)',
                                            line=dict(color=feature_colors.get(feature, '#ff6b6b'), width=2),
                                            marker=dict(size=4),
                                            showlegend=False
                                        ),
                                        row=row, col=col
                                    )
                                    
                                    # 添加正常值参考线（假设正常值范围）
                                    normal_mean = feature_data[feature][predictions == 0].mean() if normal_count > 0 else 0
                                    normal_std = feature_data[feature][predictions == 0].std() if normal_count > 0 else 1
                                    
                                    # 正常值范围线 - 使用更深的颜色
                                    fig_time_series.add_hline(
                                        y=normal_mean, 
                                        line_dash="dash", 
                                        line_color="#666666",  # 更深的灰色
                                        row=row, col=col,
                                        annotation_text="正常均值"
                                    )
                                    
                                    # 正常值范围带
                                    fig_time_series.add_trace(
                                        go.Scatter(
                                            x=[1, len(feature_values)],
                                            y=[normal_mean + normal_std, normal_mean + normal_std],
                                            mode='lines',
                                            line=dict(color='#666666', width=1, dash='dot'),  # 更深的灰色
                                            showlegend=False
                                        ),
                                        row=row, col=col
                                    )
                                    
                                    fig_time_series.add_trace(
                                        go.Scatter(
                                            x=[1, len(feature_values)],
                                            y=[normal_mean - normal_std, normal_mean - normal_std],
                                            mode='lines',
                                            line=dict(color='#666666', width=1, dash='dot'),  # 更深的灰色
                                            showlegend=False
                                        ),
                                        row=row, col=col
                                    )
                            
                            fig_time_series.update_layout(
                                title=f'{len(diabetes_indices)}条高风险ECG记录特征时间序列变化 vs 正常值参考',
                                height=200 * n_rows,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_time_series, use_container_width=True)
                            st.info(f'📈 检测到{len(diabetes_indices)}条高风险记录，显示特征时间序列变化趋势')
                else:
                    st.info('未检测到高风险记录')
            else:
                # 无高风险记录，判定为正常人
                st.success('无高风险记录，判定为正常人')
                st.info('所有ECG记录均在正常范围内，无需显示特征变化趋势图')
        else:
            st.info("请先上传数据并进行风险评估")
    
    with tab4:
        st.subheader('风险评估建议 - ecg_advisor 智能助手')
        
        # 初始化聊天历史
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # 初始化风险评估状态
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            diabetes_count = sum(predictions)
            
            # 显示风险评估摘要
            if diabetes_count > 0:
                st.warning(f'检测到 {diabetes_count} 条高风险ECG记录')
                risk_summary = f"您的ECG数据显示有{diabetes_count}条记录存在糖尿病风险。"
            else:
                st.success('所有ECG记录均显示正常')
                risk_summary = "您的ECG数据显示所有记录均在正常范围内。"
            
            # 聊天界面
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
                <h4 style="color: #1565c0; margin-bottom: 10px;">💬 ecg_advisor 智能健康助手</h4>
                <p style="color: #6c757d; margin-bottom: 10px;">基于您的ECG风险评估结果，我可以为您提供个性化的健康建议。请告诉我您关心的问题：</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 预设问题按钮 - 使用Streamlit原生按钮，居中布局
            st.markdown("""
            <style>
            /* 美化预设问题按钮 */
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
                if st.button("我应该注意什么？", key="q1", 
                           help="点击获取个性化健康建议",
                           use_container_width=True):
                    st.session_state.current_question = "我应该注意什么？"
                    st.session_state.show_ai_answer = True
            
            with col2:
                if st.button("如何预防？", key="q2",
                           help="点击获取预防建议",
                           use_container_width=True):
                    st.session_state.current_question = "如何预防？"
                    st.session_state.show_ai_answer = True
            
            with col3:
                if st.button("需要多久复查？", key="q3",
                           help="点击获取复查建议",
                           use_container_width=True):
                    st.session_state.current_question = "需要多久复查？"
                    st.session_state.show_ai_answer = True
            
            # 显示AI回答
            if st.session_state.get('show_ai_answer', False) and st.session_state.get('current_question'):
                with st.spinner("ecg_advisor 正在思考中..."):
                    # 构建ECG上下文信息
                    ecg_context = f"""
                    风险评估结果：
                    - 总记录数：{len(predictions)}
                    - 高风险记录：{diabetes_count}
                    - 正常记录：{len(predictions) - diabetes_count}
                    - 风险状态：{'高风险' if diabetes_count > 0 else '正常'}
                    """
                    
                    # 调用DeepSeek API
                    ai_answer = get_deepseek_response(
                        st.session_state.current_question, 
                        ecg_context, 
                        st.session_state.chat_history
                    )
                    
                    # 添加到聊天历史
                    st.session_state.chat_history.append({"role": "user", "content": st.session_state.current_question})
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})
                    
                    # 显示回答
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
                        <h5 style="color: #495057; margin-bottom: 10px;">🤖 ecg_advisor 的回答：</h5>
                        <div style="color: #495057; line-height:1.6;">
                            {ai_answer}
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 重置状态
                    st.session_state.show_ai_answer = False
                    st.session_state.current_question = ""
            
            # 自定义问题输入
            st.markdown("---")
            st.markdown("**💭 或者，您可以提出自己的问题：**")
            
            user_question = st.text_input("请输入您的问题：", placeholder="例如：我的风险等级如何？需要多久复查一次？")
            
            if st.button("发送问题", type="primary"):
                if user_question.strip():
                    with st.spinner("ecg_advisor 正在思考中..."):
                        # 构建ECG上下文信息
                        ecg_context = f"""
                        风险评估结果：
                        - 总记录数：{len(predictions)}
                        - 高风险记录：{diabetes_count}
                        - 正常记录：{len(predictions) - diabetes_count}
                        - 风险状态：{'高风险' if diabetes_count > 0 else '正常'}
                        """
                        
                        # 调用DeepSeek API
                        ai_answer = get_deepseek_response(
                            user_question, 
                            ecg_context, 
                            st.session_state.chat_history
                        )
                        
                        # 添加到聊天历史
                        st.session_state.chat_history.append({"role": "user", "content": user_question})
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})
                        
                        # 显示回答
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
                            <h5 style="color: #495057; margin-bottom: 10px;">🤖 ecg_advisor 的回答：</h5>
                            <div style="color: #495057; line-height: 1.6;">
                                {ai_answer}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 清空输入框
                    st.rerun()
            
            # 显示聊天历史
            if st.session_state.chat_history:
                st.markdown("---")
                st.markdown("**📝 聊天记录：**")
                
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
                            <strong>您：</strong> {message["content"]}
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
                            <strong>ecg_advisor：</strong> {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                
                # 清空聊天记录按钮
                if st.button("清空聊天记录", key="clear_chat"):
                    st.session_state.chat_history = []
                    st.rerun()
        else:
            st.info("请先上传数据并进行风险评估")


if __name__ == '__main__':
    main()


