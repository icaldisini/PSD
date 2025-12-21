import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="PCG Signal Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

html, body, [class*="css"] {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
    color: #e5e7eb;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1e 0%, #050510 100%);
    border-right: 2px solid #7c3aed;
    box-shadow: 4px 0 20px rgba(124, 58, 237, 0.15);
}

section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

h1 {
    background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    font-size: 3rem;
    text-shadow: 0 0 30px rgba(167, 139, 250, 0.3);
    margin-bottom: 0.5rem;
}

h2, h3 {
    color: #f9fafb;
    font-weight: 600;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

h3 {
    border-left: 4px solid #7c3aed;
    padding-left: 1rem;
}

.css-1629p8f, [data-testid="stCaptionContainer"] {
    color: #9ca3af;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(145deg, #1e1b4b 0%, #0f172a 100%);
    border: 2px solid transparent;
    background-clip: padding-box;
    position: relative;
    padding: 28px 24px;
    border-radius: 20px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    border-radius: 20px;
    padding: 2px;
    background: linear-gradient(135deg, #7c3aed, #a78bfa, #c4b5fd);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    opacity: 0.6;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(124, 58, 237, 0.4);
}

.metric-card::after {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(167, 139, 250, 0.1), transparent);
    transform: rotate(45deg);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
}

.metric-title {
    font-size: 11px;
    color: #a78bfa;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 12px;
}

.metric-value {
    font-size: 36px;
    font-weight: 700;
    background: linear-gradient(135deg, #f9fafb 0%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}

.metric-subtitle {
    font-size: 12px;
    color: #6b7280;
    margin-top: 8px;
}

.info-card {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border-left: 4px solid #7c3aed;
    padding: 20px;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.info-card-title {
    font-size: 14px;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 8px;
}

.info-card-content {
    font-size: 13px;
    color: #d1d5db;
    line-height: 1.6;
}

[data-testid="stDataFrame"] {
    background-color: #0f172a;
    border-radius: 16px;
    border: 1px solid #1e293b;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
}

.stSelectbox > div > div, .stMultiSelect > div > div {
    background-color: #1e293b;
    border: 1px solid #374151;
    border-radius: 10px;
    color: #e5e7eb;
}

.stSelectbox > div > div:hover, .stMultiSelect > div > div:hover {
    border-color: #7c3aed;
    box-shadow: 0 0 10px rgba(124, 58, 237, 0.3);
}

hr {
    border: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #7c3aed, transparent);
    margin: 2rem 0;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #0f172a;
    border-radius: 10px;
    padding: 8px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1e293b;
    border-radius: 8px;
    color: #9ca3af;
    padding: 12px 24px;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #a78bfa);
    color: white;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #a78bfa);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-weight: 600;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.6);
}

.stSuccess, .stInfo {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border-radius: 12px;
    border-left: 4px solid #7c3aed;
    padding: 16px;
}

.streamlit-expanderHeader {
    background-color: #1e293b;
    border-radius: 10px;
    border: 1px solid #374151;
}

.streamlit-expanderHeader:hover {
    border-color: #7c3aed;
}

::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: #0f172a;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #7c3aed, #a78bfa);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #a78bfa, #c4b5fd);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.main .block-container > div {
    animation: fadeIn 0.6s ease-out;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("pcg_top_values_per_class.csv")

df = load_data()

st.title("PCG Signal Intelligence Dashboard")
st.caption(
    "Advanced feature-level analysis of phonocardiogram signals using "
    "frequency and time-frequency domain descriptors (FFT, PSD, STFT, CWT). "
    "Real-time cardiac acoustics visualization and pattern recognition."
)

st.divider()

st.sidebar.markdown("### Analysis Controls")
st.sidebar.markdown("---")

label_selected = st.sidebar.selectbox(
    "Heart Condition",
    ["All"] + sorted(df["Label"].unique()),
    help="Select a specific cardiac condition or view all"
)

st.sidebar.markdown("---")

feature_selected = st.sidebar.multiselect(
    "Displayed Features",
    [c for c in df.columns if c != "Label"],
    default=["FFT_max_Hz", "PSD_max_Hz", "STFT_max_Hz", "CWT_max_Hz"],
    help="Choose which features to display in visualizations"
)

st.sidebar.markdown("---")

show_advanced = st.sidebar.checkbox("Advanced Analytics", value=True)
show_correlations = st.sidebar.checkbox("Correlation Matrix", value=False)
show_3d = st.sidebar.checkbox("3D Visualization", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="info-card">
    <div class="info-card-title">Quick Guide</div>
    <div class="info-card-content">
    • <b>AS</b>: Aortic Stenosis<br>
    • <b>MR</b>: Mitral Regurgitation<br>
    • <b>MS</b>: Mitral Stenosis<br>
    • <b>MVP</b>: Mitral Valve Prolapse<br>
    • <b>N</b>: Normal
    </div>
</div>
""", unsafe_allow_html=True)

if label_selected != "All":
    df_view = df[df["Label"] == label_selected]
else:
    df_view = df.copy()

st.markdown("### Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Samples</div>
        <div class="metric-value">{len(df_view)}</div>
        <div class="metric-subtitle">Records analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Classes</div>
        <div class="metric-value">{df_view['Label'].nunique()}</div>
        <div class="metric-subtitle">Cardiac conditions</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">FFT Peak</div>
        <div class="metric-value">{df_view['FFT_max_Hz'].mean():.1f}</div>
        <div class="metric-subtitle">Hz (average)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">CWT Coeff</div>
        <div class="metric-value">{df_view['CWT_max_coeff'].mean():.2f}</div>
        <div class="metric-subtitle">Mean coefficient</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    max_magnitude = df_view['FFT_max_mag'].max()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Max Magnitude</div>
        <div class="metric-value">{max_magnitude:.1f}</div>
        <div class="metric-subtitle">FFT peak value</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Feature Analysis", "Distribution", "Advanced", "Data Table"])

with tab1:
    st.markdown("### Feature Comparison Across Classes")
    
    if feature_selected:
        mean_df = df.groupby("Label")[feature_selected].mean().reset_index()
        
        fig = go.Figure()
        
        colors = ['#7c3aed', '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe', '#f5f3ff', '#faf5ff', '#fdf4ff']
        
        for idx, feature in enumerate(feature_selected):
            fig.add_trace(go.Bar(
                name=feature,
                x=mean_df['Label'],
                y=mean_df[feature],
                marker_color=colors[idx % len(colors)],
                text=mean_df[feature].round(2),
                textposition='auto',
            ))
        
        fig.update_layout(
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb', size=12),
            xaxis=dict(
                title='Cardiac Condition',
                gridcolor='#1e293b',
                showgrid=True
            ),
            yaxis=dict(
                title='Mean Value',
                gridcolor='#1e293b',
                showgrid=True
            ),
            hovermode='x unified',
            legend=dict(
                bgcolor='rgba(30, 41, 59, 0.8)',
                bordercolor='#7c3aed',
                borderwidth=1
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Frequency Distribution Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scatter = px.scatter(
            df_view, 
            x='FFT_max_Hz', 
            y='PSD_max_Hz',
            color='Label',
            size='FFT_max_mag',
            hover_data=['CWT_max_Hz'],
            title='FFT vs PSD Frequency Distribution',
            color_discrete_sequence=px.colors.sequential.Purples_r
        )
        
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb'),
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b'),
            height=400
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        if label_selected != "All":
            categories = ['FFT_max_Hz', 'PSD_max_Hz', 'STFT_max_Hz', 'CWT_max_Hz']
            
            fig_radar = go.Figure()
            
            values = df_view[categories].mean().values.tolist()
            values += values[:1]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=label_selected,
                line_color='#7c3aed',
                fillcolor='rgba(124, 58, 237, 0.3)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        gridcolor='#1e293b',
                        color='#9ca3af'
                    ),
                    bgcolor='rgba(0,0,0,0)',
                    angularaxis=dict(gridcolor='#1e293b')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'),
                title=f'Frequency Profile: {label_selected}',
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Select a specific condition in the sidebar to view the radar chart.")

with tab2:
    st.markdown("### Feature Distribution Analysis")
    
    box_feature = st.selectbox(
        "Select Feature for Analysis",
        [c for c in df.columns if c != "Label"],
        key="box_feature"
    )
    
    fig_box = px.box(
        df, 
        x='Label', 
        y=box_feature,
        color='Label',
        title=f'Distribution of {box_feature} Across Conditions',
        color_discrete_sequence=px.colors.sequential.Purples_r
    )
    
    fig_box.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb', size=12),
        xaxis=dict(
            title='Cardiac Condition',
            gridcolor='#1e293b'
        ),
        yaxis=dict(
            title=box_feature,
            gridcolor='#1e293b'
        ),
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("### Density Distribution")
    
    fig_violin = px.violin(
        df,
        x='Label',
        y=box_feature,
        color='Label',
        box=True,
        points='all',
        title=f'Density Distribution of {box_feature}',
        color_discrete_sequence=px.colors.sequential.Purples_r
    )
    
    fig_violin.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb'),
        xaxis=dict(gridcolor='#1e293b'),
        yaxis=dict(gridcolor='#1e293b'),
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig_violin, use_container_width=True)

with tab3:
    if show_advanced:
        st.markdown("### Normalized Feature Signature")
        
        numeric_features = df.drop(columns=["Label"])
        normalized = (numeric_features - numeric_features.min()) / (
            numeric_features.max() - numeric_features.min()
        )
        
        norm_df = pd.concat([df["Label"], normalized], axis=1)
        
        fig_heatmap = px.imshow(
            norm_df.drop(columns=['Label']).T,
            labels=dict(x="Sample Index", y="Features", color="Normalized Value"),
            x=norm_df.index,
            y=norm_df.drop(columns=['Label']).columns,
            color_continuous_scale='Purples',
            title='Normalized Feature Heatmap'
        )
        
        fig_heatmap.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb'),
            height=500
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    if show_correlations:
        st.markdown("### Feature Correlation Matrix")
        
        corr_matrix = df.drop(columns=['Label']).corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title='Feature Correlation Heatmap'
        )
        
        fig_corr.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb'),
            height=600
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    if show_3d:
        st.markdown("### 3D Feature Space Visualization")
        
        fig_3d = px.scatter_3d(
            df,
            x='FFT_max_Hz',
            y='PSD_max_Hz',
            z='CWT_max_coeff',
            color='Label',
            size='FFT_max_mag',
            hover_data=['STFT_max_Hz'],
            title='3D Feature Space',
            color_discrete_sequence=px.colors.sequential.Purples_r
        )
        
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='#1e293b'),
                yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='#1e293b'),
                zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='#1e293b'),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb'),
            height=700
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)

with tab4:
    st.markdown("### Complete Feature Table")
    st.dataframe(
        df_view.style.background_gradient(cmap='Purples', subset=[c for c in df_view.columns if c != 'Label']),
        use_container_width=True,
        height=400
    )
    
    st.markdown("### Statistical Summary")
    
    stats = df_view.drop(columns=["Label"]).describe().T
    st.dataframe(
        stats.style.format("{:.4f}").background_gradient(cmap='Purples'),
        use_container_width=True
    )

st.divider()
st.markdown("### AI-Powered Signal Interpretation")

col1, col2 = st.columns([2, 1])

with col1:
    fft_mean = df_view["FFT_max_Hz"].mean()
    cwt_mean = df_view["CWT_max_coeff"].mean()
    psd_mean = df_view["PSD_max_val"].mean()
    
    if fft_mean > 95 and cwt_mean > 3:
        st.success(
            "**Pathological Pattern Detected**\n\n"
            "The signal exhibits dominant high-frequency components "
            f"(FFT: {fft_mean:.2f} Hz) and elevated time-frequency energy "
            f"(CWT: {cwt_mean:.2f}). Such characteristics are commonly associated with "
            "pathological cardiac murmurs, indicating potential valve dysfunction or "
            "structural abnormalities."
        )
    else:
        st.info(
            "**Normal Pattern Characteristics**\n\n"
            f"The signal is dominated by lower-frequency components (FFT: {fft_mean:.2f} Hz) "
            f"with moderate time-frequency energy (CWT: {cwt_mean:.2f}), "
            "suggesting normal or mildly altered cardiac conditions. "
            "These parameters fall within expected physiological ranges."
        )

with col2:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-card-title">Key Metrics</div>
        <div class="info-card-content">
        • FFT Mean: <b>{fft_mean:.2f} Hz</b><br>
        • CWT Mean: <b>{cwt_mean:.2f}</b><br>
        • PSD Mean: <b>{psd_mean:.6f}</b><br>
        • Samples: <b>{len(df_view)}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
    <p style='font-size: 14px; margin-bottom: 8px;'>
        <b>PCG Signal Intelligence Platform</b> | Biomedical Signal Processing Dashboard
    </p>
    <p style='font-size: 12px;'>
        Advanced Cardiac Acoustics Analysis • Feature-Based Classification • 
        Clinical Decision Support • Research-Grade Analytics
    </p>
    <p style='font-size: 11px; margin-top: 12px; color: #4b5563;'>
        Powered by Streamlit • Plotly • NumPy • Pandas
    </p>
</div>
""", unsafe_allow_html=True)