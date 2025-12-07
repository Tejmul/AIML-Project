import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import re
from datetime import datetime, timedelta

# Set page configuration with wide layout
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- HELPER FUNCTIONS ----------------

@st.cache_data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

@st.cache_data
def parse_video_datetime(video_name):
    """
    Parses date and hour from video name format: cctv052x2004080516x01638
    Format seems to be: cctv[ID]x[YYYYMMDDHH]x...
    """
    match = re.search(r'x(\d{10})x', video_name)
    if match:
        date_str = match.group(1)
        try:
            return datetime.strptime(date_str, "%Y%m%d%H")
        except ValueError:
            return None
    return None

@st.cache_data
def process_dataframe(frames_data):
    df = pd.DataFrame(frames_data)
    
    # Pre-parse all unique video times
    unique_videos = df['video'].unique()
    video_start_times = {v: parse_video_datetime(v) for v in unique_videos}
    
    timestamps = []
    
    # Vectorized approach would be better but iterating is fine for this scale
    for vid, offset in zip(df['video'], df['timestamp_seconds']):
        base_time = video_start_times.get(vid)
        if base_time:
            timestamps.append(base_time + timedelta(seconds=offset))
        else:
            timestamps.append(None)
            
    df['datetime'] = timestamps
    
    # Create derived columns for plotting
    df['Date'] = df['datetime'].apply(lambda x: x.date() if x else None)
    df['Time'] = df['datetime'].apply(lambda x: x.time() if x else None)
    
    # For Plotly "Time of Day" axis, we normalize to a single dummy date
    df['TimeOfDay'] = df['datetime'].apply(lambda x: x.replace(year=2000, month=1, day=1) if x else None)
    
    return df

# ---------------- SIDEBAR: SIMPLE CONTROLS ----------------

st.sidebar.header("🕹️ Controls")

# Data Source
default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'traffic_anomaly_detection_report.json')
uploaded_file = st.sidebar.file_uploader("Upload Report", type=['json'])

data = None
if uploaded_file is not None:
    data = json.load(uploaded_file)
elif os.path.exists(default_path):
    data = load_data(default_path)
else:
    st.error("Please upload a JSON report.")
    st.stop()

# Load and Process Data
raw_frames = data.get('frames_data', [])
if not raw_frames:
    st.warning("No data found.")
    st.stop()

df = process_dataframe(raw_frames)

# Simplified Filter
st.sidebar.subheader("Filters")
all_videos = df['video'].unique()
selected_videos = st.sidebar.multiselect("Select Videos", all_videos, default=all_videos, help="Filter the dashboard by specific video feeds.")

if selected_videos:
    df_filtered = df[df['video'].isin(selected_videos)]
else:
    df_filtered = df # Show all if none selected

# ---------------- MAIN DASHBOARD ----------------

st.title("🚦 Traffic Anomaly Overview")

# 1. METRICS ROW
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Analyzed Frames", f"{len(df_filtered):,}")
with col2:
    anom_count = df_filtered[df_filtered['is_anomaly']].shape[0]
    st.metric("Anomalies", f"{anom_count}", delta_color="inverse")
with col3:
    acc_count = df_filtered[df_filtered['is_accident']].shape[0]
    st.metric("Accidents", f"{acc_count}", delta_color="inverse")
with col4:
    high_cong = df_filtered[df_filtered['congestion_level'] == 'HIGH'].shape[0]
    st.metric("High Congestion", f"{high_cong}", delta_color="inverse")

st.markdown("---")

# 2. DETAILED TRENDS
col_trend, col_dist = st.columns([2, 1])

with col_trend:
    st.subheader("📈 Anomaly Score Trend")
    if not df_filtered.empty:
        # Use simple timestamp if DateTime failed
        x_axis = 'datetime' if not df_filtered['datetime'].isnull().all() else 'timestamp_seconds'
        
        fig_trend = px.line(
            df_filtered,
            x=x_axis,
            y="anomaly_score",
            color="video" if len(selected_videos) <= 5 else None, # Don't color by video if too many
            title="Anomaly Score Progression",
            labels={"anomaly_score": "Score", x_axis: "Time"}
        )
        fig_trend.update_layout(
            height=350,
            showlegend=False
        )
        # Add threshold line
        if 'thresholds' in data.get('summary', {}):
             thresh = data['summary']['thresholds'].get('accident_threshold')
             if thresh:
                 fig_trend.add_hline(y=thresh, line_dash="dash", line_color="red", annotation_text="Threshold")
        
        st.plotly_chart(fig_trend, width="stretch")

with col_dist:
    st.subheader("Congestion Share")
    if 'congestion_level' in df_filtered.columns:
        counts = df_filtered['congestion_level'].value_counts()
        fig_pie = px.pie(
            values=counts.values, 
            names=counts.index,
            color=counts.index,
            color_discrete_map={'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'},
            title="Congestion Distribution",
            hole=0.4
        )
        fig_pie.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig_pie, width="stretch")

# 3. DEEP DIVE ANALYTICS ROW
st.subheader("🔍 Deep Dive Analytics")
col_box, col_hist, col_bar = st.columns(3)

with col_box:
    fig_box = px.box(
        df_filtered,
        x="congestion_level",
        y="anomaly_score",
        color="congestion_level",
        title="Scores by Congestion",
        color_discrete_map={'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'}
    )
    fig_box.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_box, width="stretch")

with col_hist:
    fig_hist = px.histogram(
        df_filtered,
        x="anomaly_score",
        nbins=50,
        title="Score Distribution",
        color_discrete_sequence=['#6366f1']
    )
    fig_hist.update_layout(height=300, showlegend=False, xaxis_title="Score", yaxis_title="Count")
    st.plotly_chart(fig_hist, width="stretch")

with col_bar:
    # Aggregated stats per video
    if not df_filtered.empty:
        vid_stats = df_filtered.groupby('video')['is_anomaly'].sum().reset_index()
        vid_stats = vid_stats.sort_values('is_anomaly', ascending=False).head(10)
        
        fig_bar = px.bar(
            vid_stats,
            x='is_anomaly',
            y='video',
            orientation='h',
            title="Top Videos by Anomaly Count",
            color_discrete_sequence=['#ef4444']
        )
        fig_bar.update_layout(height=300, showlegend=False, xaxis_title="Anomalies", yaxis_title="")
        st.plotly_chart(fig_bar, width="stretch")

# 4. TEMPORAL CLUSTERS (The "Dot Plots")
st.subheader("📅 Temporal Patterns")
st.caption("Identify when anomalies and congestion occur most frequently.")

if df_filtered['datetime'].isnull().all():
    st.error("No timestamp data available in video filenames.")
else:
    tab1, tab2 = st.tabs(["Congestion Clusters", "Anomaly Clusters"])
    
    with tab1:
        fig_cong = px.scatter(
            df_filtered,
            x="TimeOfDay",
            y="Date",
            color="congestion_level",
            color_discrete_map={'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'},
            title="Congestion Intensity by Time & Date",
            labels={"TimeOfDay": "Time of Day", "Date": "Date", "congestion_level": "Level"},
            hover_data=['video', 'timestamp']
        )
        fig_cong.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
        fig_cong.update_layout(
            xaxis_tickformat="%H:%M",
            height=400,
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )
        st.plotly_chart(fig_cong, width="stretch")

    with tab2:
        anom_df = df_filtered[df_filtered['is_anomaly'] == True]
        
        if not anom_df.empty:
            fig_anom = px.scatter(
                anom_df,
                x="TimeOfDay",
                y="Date",
                color="is_accident", # Differentiate accidents
                color_discrete_map={True: '#ef4444', False: '#3b82f6'},
                symbol="is_accident",
                title="Detected Anomalies by Time & Date",
                labels={"TimeOfDay": "Time of Day", "Date": "Date", "is_accident": "Is Accident?"},
                hover_data=['video', 'timestamp', 'anomaly_score']
            )
            fig_anom.update_traces(marker=dict(size=12, opacity=0.9, line=dict(width=1, color='DarkSlateGrey')))
            fig_anom.update_layout(
                xaxis_tickformat="%H:%M",
                height=400,
                 xaxis=dict(showgrid=True),
                 yaxis=dict(showgrid=True)
            )
            st.plotly_chart(fig_anom, width="stretch")
        else:
            st.info("No anomalies detected in the selected dataset.")

# 5. DATA TABLE
with st.expander("📄 View Raw Data Table"):
    st.dataframe(
        df_filtered[['video', 'timestamp', 'congestion_level', 'anomaly_score', 'is_anomaly', 'is_accident']],
        width="stretch"
    )
    
st.caption("Dashboard v1.3 | AI/ML Project")
