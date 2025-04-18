import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Streamlit app for interactive benchmark visualization
st.set_page_config(page_title="Benchmark Dashboard", layout="wide")

# Load data
with open('benchmark_data.json', 'r') as f:
    data = json.load(f)

# Define expected correct tables per question (manually provided)
expected_map = {
    "Hva var befolkningstallet i Norge i 2022?": {
        "07459", "06913", "03027", "03031", "11342",
        "12871", "13536", "05803", "10211", "05810", "05328"
    },
    "Hvor mange heter Trygve til fornavn i dag?": {"10501"},
    "Hva var Norges BNP fra 2010-2020?": {"09189"}
}

# Build DataFrame including token usage and correctness
rows = []
for config in data.get('configurations', []):
    tech = config.get('navigationConfiguration', {}).get('navigationTechnique', 'unknown')
    for benchmark in config.get('benchmarkAnswers', []):
        question = benchmark.get('userPrompt', '')
        expected = expected_map.get(question, set())
        for idx, resp in enumerate(benchmark.get('responses', []), start=1):
            tu = resp.get('tokenUsage', {}) or {}
            table_id = resp.get('tableId', '')
            rows.append({
                "Technique": tech,
                "Question": question,
                "Run": idx,
                "Time_ms": resp.get('milliseconds', 0),
                "TableID": table_id,
                "Expected": table_id in expected,
                "PromptTokens": tu.get('promptTokens', 0),
                "CompletionTokens": tu.get('completionTokens', 0),
                "TotalTokens": tu.get('totalTokens', 0)
            })

# Create DataFrame
df = pd.DataFrame(rows)

# Sidebar filters
st.sidebar.header("Filters")
techniques = st.sidebar.multiselect(
    "Navigation Technique", df['Technique'].unique(), default=df['Technique'].unique()
)
questions = st.sidebar.multiselect(
    "Questions", df['Question'].unique(), default=df['Question'].unique()
)
expected_filter = st.sidebar.selectbox(
    "Show", ["All", "Expected", "Unexpected"]
)

# Apply filters
df_filtered = df.copy()
if techniques:
    df_filtered = df_filtered[df_filtered['Technique'].isin(techniques)]
if questions:
    df_filtered = df_filtered[df_filtered['Question'].isin(questions)]
if expected_filter == "Expected":
    df_filtered = df_filtered[df_filtered['Expected']]
elif expected_filter == "Unexpected":
    df_filtered = df_filtered[~df_filtered['Expected']]

# Main dashboard
st.title("Benchmark Dashboard")

if df_filtered.empty:
    st.info("No data available for the selected filters.")
else:
    # Raw Data Table
    st.subheader("Raw Data")
    st.dataframe(df_filtered)

    # Per-Question Detailed View: Time & Token Usage Side by Side
    st.subheader("Detailed View by Question")
    for question in df_filtered['Question'].unique():
        st.markdown(f"### {question}")
        df_q = df_filtered[df_filtered['Question'] == question]
        col_time, col_tokens = st.columns(2)

        # Time Usage Chart
        with col_time:
            st.write("**Response Time (ms)**")
            fig_time = go.Figure()
            tech_colors = {
                'folderNavigation': 'rgba(0, 123, 255, 0.7)',
                'keywordSearch': 'rgba(0, 82, 204, 0.7)'
            }
            for tech in df_q['Technique'].unique():
                df_t = df_q[df_q['Technique'] == tech].sort_values('Run')
                outline_colors = df_t['Expected'].map({True: 'green', False: 'red'}).tolist()
                fig_time.add_trace(go.Bar(
                    x=df_t['Run'],
                    y=df_t['Time_ms'],
                    name=tech,
                    marker=dict(
                        color=tech_colors.get(tech, 'grey'),
                        line=dict(color=outline_colors, width=3)
                    ),
                    offsetgroup=tech
                ))
            fig_time.update_layout(
                xaxis_title="Run Index",
                yaxis_title="Time (ms)",
                barmode='group',
                showlegend=False
            )
            st.plotly_chart(fig_time, use_container_width=True, key=f"time_{question}")

        # Token Usage Chart
        with col_tokens:
            st.write("**Total Tokens Used**")
            fig_tokens = go.Figure()
            for tech in df_q['Technique'].unique():
                df_t = df_q[df_q['Technique'] == tech].sort_values('Run')
                outline_colors = df_t['Expected'].map({True: 'green', False: 'red'}).tolist()
                fig_tokens.add_trace(go.Bar(
                    x=df_t['Run'],
                    y=df_t['TotalTokens'],
                    name=tech,
                    marker=dict(
                        color=tech_colors.get(tech, 'grey'),
                        line=dict(color=outline_colors, width=3)
                    ),
                    offsetgroup=tech
                ))
            fig_tokens.update_layout(
                xaxis_title="Run Index",
                yaxis_title="Total Tokens",
                barmode='group',
                showlegend=False
            )
            st.plotly_chart(fig_tokens, use_container_width=True, key=f"tokens_{question}")

    # Histogram: Response Time Distribution
    st.subheader("Histogram: Response Time Distribution")
    fig3 = px.histogram(
        df_filtered,
        x="Time_ms",
        color="Technique",
        marginal="box",
        nbins=20,
        labels={"Time_ms": "Response Time (ms)"}
    )
    st.plotly_chart(fig3, use_container_width=True, key="histogram")


