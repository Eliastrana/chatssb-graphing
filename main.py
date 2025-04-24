import streamlit as st
import pandas as pd
import altair as alt
import json

from networkx.algorithms.bipartite import color
from setuptools.command.rotate import rotate

# Set page layout to wide
st.set_page_config(layout="wide", page_title="Benchmark Comparison")

# Load benchmark data (save the provided JSON as 'benchmark_data.json' in the same directory)
with open('benchmark_data.json', 'r') as f:
    data = json.load(f)

# Extract available models for selection
all_models = [cfg['navigationConfiguration']['model'] for cfg in data['configurations']]
# Remove duplicates while preserving order
models = list(dict.fromkeys(all_models))

st.sidebar.header("Filter Models")
selected_models = st.sidebar.multiselect(
    "Select language models to show:",
    options=models,
    default=models
)

# Flatten nested structure into a DataFrame, including the model name
rows = []
for cfg in data['configurations']:
    model_name = cfg['navigationConfiguration']['model']
    tech = cfg['navigationConfiguration']['navigationTechnique']
    nav_val = cfg['navigationConfiguration']['navigationValue']
    for bm in cfg['benchmarkAnswers']:
        prompt_text = bm['userPrompt']
        for resp in bm['responses']:
            rows.append({
                'model': model_name,
                'navigationTechnique': tech,
                'navigationValue': str(nav_val),
                'userPrompt': prompt_text,
                'result': resp.get('result'),
                'milliseconds': resp.get('milliseconds'),
                'totalTokens': resp['tokenUsage'].get('totalTokens')
            })

df = pd.DataFrame(rows)



# --- Filter out any error responses ---
df_3d = df[df['result'] != 'error']


import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- Price-per-Token Inputs ---
st.sidebar.header("Token Pricing")

initial_prices = {
    "gemini-2.0-flash": 0.17e-6,
    "gemini-2.0-flash-lite": 0.13e-6,
    "gpt-4.1-2025-04-14": 0.7e-6,
    # "gpt-4.1-mini-2025-04-14": 0.02,
    # "gpt-4.1-nano-2025-04-14": 0.02,
    "llama-3.3-70b-versatile": 0.64e-6,
    "meta-llama/llama-4-maverick-17b-128e-instruct": 0.3e-6,
    "meta-llama/llama-4-scout-17b-16e-instruct": 0.17e-6,
    "deepseek-r1-distill-llama-70b": 0.81e-6,

}

price_per_token = {}
for model in models:
    price_per_token[model] = st.sidebar.number_input(
        f"Price per token for {model}",
        min_value=0.0,
        value=initial_prices.get(model, 0.001),
        format="%.8f"
    )
# --- Compute Model-Level Performance & Cost ---
model_perf = (
    df_3d
    .groupby('model', as_index=False)
    .agg(
        avg_speed_ms = ('milliseconds', 'mean'),
        avg_accuracy = ('result',     lambda x: x.isin(['correct','technicallyCorrect']).mean()*100),
        avg_tokens   = ('totalTokens','mean')
    )
)
# Add a cost column (tokens × price)
model_perf['cost'] = model_perf.apply(
    lambda row: row['avg_tokens'] * price_per_token[row['model']],
    axis=1
)

# --- 3D Performance-Cost Plane with True Gradient ---

# 1) Re-fit your best plane coefficients a,b,d
A = np.c_[
    model_perf['avg_speed_ms'],
    model_perf['avg_accuracy'],
    np.ones(len(model_perf))
]
C, *_ = np.linalg.lstsq(A, model_perf['cost'], rcond=None)
a, b, d = C

# 2) Create a fine grid over the actual data ranges
speed_lin = np.linspace(model_perf['avg_speed_ms'].min(), model_perf['avg_speed_ms'].max(), 30)
acc_lin   = np.linspace(model_perf['avg_accuracy'].min(),   model_perf['avg_accuracy'].max(),   30)
S, A_ = np.meshgrid(speed_lin, acc_lin)

# 3) Compute the plane Z and clamp it within your real cost bounds
Z_raw = a * S + b * A_ + d
min_cost, max_cost = model_perf['cost'].min(), model_perf['cost'].max()
Z = np.clip(Z_raw, min_cost, max_cost)

# 4) Build a true 2D green_intensity:
#    fast→1, accurate→1, cheap→1; slower/less accurate/more expensive → closer to 0
norm_s = 1 - (S - S.min()) / (S.max() - S.min())
norm_a =     (A_ - A_.min()) / (A_.max() - A_.min())
norm_z = 1 - (Z  - min_cost) / (max_cost - min_cost)
green_intensity = norm_s * norm_a * norm_z

# 5) Plot!
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=model_perf['avg_speed_ms'],
    y=model_perf['avg_accuracy'],
    z=model_perf['cost'],
    mode='markers+text',
    marker=dict(size=6, color='navy'),
    text=model_perf['model'],
    textposition='top center'
))

fig.update_layout(
    scene=dict(
        xaxis_title='Avg Speed (ms)',
        yaxis_title='Accuracy (%)',
        zaxis=dict(
            title='Cost',
            autorange='reversed'
        )
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)

# Compute the optimal corner
opt_speed    = model_perf['avg_speed_ms'].min()
opt_accuracy = model_perf['avg_accuracy'].max()
opt_cost     = model_perf['cost'].min()

# … then after your existing fig.add_trace for the models …
fig.add_trace(
    go.Scatter3d(
        x=[opt_speed],
        y=[opt_accuracy],
        z=[opt_cost],
        mode='markers',
        marker=dict(
            size=12,
            color='red',
            symbol='diamond'
        ),
        name='Optimal'
    )
)


st.header("3D Performance-Cost Plane")
st.plotly_chart(fig, use_container_width=True)



# --- MOST DESIRED PLANE: speed vs accuracy per model ---
# 1) Compute each model’s overall average speed & accuracy
model_perf = (
    df
    .groupby('model', as_index=False)
    .agg(
        avg_speed_ms=('milliseconds', 'mean'),
        avg_accuracy = ('result', lambda x: x.isin(['correct','technicallyCorrect']).mean()*100)
    )
)

st.header("Model Performance: Speed vs Accuracy")

# 2) Scatter plot with labels
points = alt.Chart(model_perf).mark_circle(size=100).encode(
    x=alt.X('avg_speed_ms:Q', title='Avg Speed (ms)'),
    y=alt.Y('avg_accuracy:Q', title='Accuracy (%)'),
    color=alt.Color('model:N', legend=None),
    tooltip=['model','avg_speed_ms','avg_accuracy']
)

labels = alt.Chart(model_perf).mark_text(dx=7, dy=-7).encode(
    x='avg_speed_ms:Q',
    y='avg_accuracy:Q',
    text='model:N',
    color=alt.Color('model:N', legend=None)
)

plane = (points + labels).properties(
    width=700,
    height=500
)

st.altair_chart(plane, use_container_width=True)


# Compute overall metrics aggregated by navigation technique
agg_overall = df.groupby('navigationTechnique', as_index=False).agg(
    avg_speed_ms=('milliseconds', 'mean'),
    avg_tokens=('totalTokens', 'mean'),
    avg_accuracy=('result', lambda x: x.isin(['correct', 'technicallyCorrect']).mean() * 100)
)

# --- TOP GRAPHS (unchanged) ---
st.title("ChatSSB – Performance")
st.header("Generell oversikt over navigeringstype")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Average Speed (ms)")
    chart_speed = alt.Chart(agg_overall).mark_bar().encode(
        x=alt.X('navigationTechnique:N', title='Technique'),
        y=alt.Y('avg_speed_ms:Q', title='Speed (ms)'),
        color=alt.Color('navigationTechnique:N', legend=None),
        tooltip=['navigationTechnique', 'avg_speed_ms']
    ).properties(height=300)
    st.altair_chart(chart_speed, use_container_width=True)

with col2:
    st.subheader("Average Token Usage")
    chart_tokens = alt.Chart(agg_overall).mark_bar().encode(
        x=alt.X('navigationTechnique:N', title='Technique'),
        y=alt.Y('avg_tokens:Q', title='Total Tokens'),
        color=alt.Color('navigationTechnique:N', legend=None),
        tooltip=['navigationTechnique', 'avg_tokens']
    ).properties(height=300)
    st.altair_chart(chart_tokens, use_container_width=True)

with col3:
    st.subheader("Average Accuracy (%)")
    chart_accuracy = alt.Chart(agg_overall).mark_bar().encode(
        x=alt.X('navigationTechnique:N', title='Technique'),
        y=alt.Y('avg_accuracy:Q', title='Accuracy (%)'),
        color=alt.Color('navigationTechnique:N', legend=None),
        tooltip=['navigationTechnique', alt.Tooltip('avg_accuracy:Q', format='.1f')]
    ).properties(height=300)
    st.altair_chart(chart_accuracy, use_container_width=True)

st.markdown("---")

# --- AGGREGATED PER CONFIGURATION ---
agg_config = (
    df
    .groupby(['model', 'navigationTechnique', 'navigationValue'], as_index=False)
    .agg(
        total_speed_ms=('milliseconds', 'sum'),
        total_tokens=('totalTokens', 'sum'),
        accuracy=('result', lambda x: x.isin(['correct', 'technicallyCorrect']).mean() * 100)
    )
)
# create a human‐readable label for each config
agg_config['config'] = (
        agg_config['model']
        + ' | '
        + agg_config['navigationTechnique']
        + ' ('
        + agg_config['navigationValue']
        + ')'
)

st.header("Aggregert per konfigurasjon")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Total Speed (ms)")
    chart1 = alt.Chart(agg_config).mark_bar().encode(
        x=alt.X(
            'config:N',
            title='Configuration',
            axis=alt.Axis(
                labelAngle=-90,  # tilt them so they don’t overlap
                labelAlign='right',
                labelLimit=500  # allow up to 500px before cutting off
            )
        ),        y=alt.Y('total_speed_ms:Q', title='Total Speed (ms)'),
        color=alt.Color('model:N', legend=None),
        tooltip=['config', 'total_speed_ms']
    ).properties(height=550)
    st.altair_chart(chart1, use_container_width=True)

with col2:
    st.subheader("Total Tokens")
    chart2 = alt.Chart(agg_config).mark_bar().encode(
        x=alt.X(
            'config:N',
            title='Configuration',
            axis=alt.Axis(
                labelAngle=-90,  # tilt them so they don’t overlap
                labelAlign='right',
                labelLimit=500  # allow up to 500px before cutting off
            )
        ),        y=alt.Y('total_tokens:Q', title='Total Tokens'),
        color=alt.Color('model:N', legend=None),
        tooltip=['config', 'total_tokens']
    ).properties(height=550)
    st.altair_chart(chart2, use_container_width=True)

with col3:
    st.subheader("Accuracy (%)")
    chart3 = alt.Chart(agg_config).mark_bar().encode(
        x=alt.X(
            'config:N',
            title='Configuration',
            axis=alt.Axis(
                labelAngle=-90,  # tilt them so they don’t overlap
                labelAlign='right',
                labelLimit=500  # allow up to 500px before cutting off
            )
        ),
        y=alt.Y('accuracy:Q', title='Accuracy (%)'),
        color=alt.Color('model:N', legend=None),
        tooltip=[
            'config',
            alt.Tooltip('accuracy:Q', format='.1f')
        ]
    ).properties(height=550)
    st.altair_chart(chart3, use_container_width=True)


# --- PER-PROMPT, PER-MODEL GRAPHS ---
prompts = df['userPrompt'].unique()
blue_shades = ['#c6dbef', '#6baed6', '#2171b5']
domain_values = ['1', '3', '5']

for prompt in prompts:
    st.header(f" Prompt: {prompt}")
    sub_df = df[df['userPrompt'] == prompt]

    # For each selected model, show its charts vertically
    for m in selected_models:
        m_df = sub_df[sub_df['model'] == m]
        if m_df.empty:
            continue

        st.subheader(f"{m}")

        # Speed chart for this model
        speed_chart = alt.Chart(m_df).mark_bar().encode(
            x=alt.X('milliseconds:Q', title='Speed (ms)'),
            y=alt.Y('navigationTechnique:N', title='Technique'),
            stroke=alt.Stroke('result:N', scale=alt.Scale(
                domain=['correct', 'technicallyCorrect', 'incorrect', 'error'],
                range=['#06bd36', 'yellow', 'red', 'purple']
            )),
            strokeWidth=alt.value(4),
            color=alt.Color('navigationValue:N', title='Nav Value',
                            scale=alt.Scale(domain=domain_values, range=blue_shades)),
            tooltip=['navigationTechnique', 'navigationValue', 'result', 'milliseconds', 'totalTokens']
        ).properties(width=800, height=200)
        st.altair_chart(speed_chart, use_container_width=True)

        # Token usage chart for this model
        token_chart = alt.Chart(m_df).mark_bar().encode(
            x=alt.X('totalTokens:Q', title='Total Tokens'),
            y=alt.Y('navigationTechnique:N', title='Technique'),
            stroke=alt.Stroke('result:N', scale=alt.Scale(
                domain=['correct', 'technicallyCorrect', 'incorrect', 'error'],
                range=['#06bd36', 'yellow', 'red', 'purple']
            )),
            strokeWidth=alt.value(4),
            color=alt.Color('navigationValue:N', title='Nav Value',
                            scale=alt.Scale(domain=domain_values, range=blue_shades)),
            tooltip=['navigationTechnique', 'navigationValue', 'result', 'totalTokens']
        ).properties(width=800, height=200)
        st.altair_chart(token_chart, use_container_width=True)

        # Optionally show detailed response data
        with st.expander(f"{m} Responses"):
            st.write(m_df[['navigationTechnique', 'navigationValue', 'result', 'milliseconds', 'totalTokens']])

    st.markdown("---")



# --- CONFIGURATION-FIRST SECTION: per-question graphs ---
# 1) Precompute per-question averages for each configuration
agg_q = (
    df
    .groupby(
        ['model', 'navigationTechnique', 'navigationValue', 'userPrompt'],
        as_index=False
    )
    .agg(
        avg_speed_ms=('milliseconds', 'mean'),
        avg_tokens=('totalTokens', 'mean'),
        avg_accuracy=('result', lambda x: x.isin(['correct','technicallyCorrect']).mean()*100)
    )
)

st.header("Per-Question Performance by Configuration")
for _, cfg in agg_config.iterrows():
    model = cfg['model']
    tech  = cfg['navigationTechnique']
    nav   = cfg['navigationValue']
    title = f"{model} | {tech} ({nav})"
    st.subheader(title)

    sub = agg_q[
        (agg_q['model'] == model) &
        (agg_q['navigationTechnique'] == tech) &
        (agg_q['navigationValue'] == nav)
    ]

    # Speed chart
    speed_chart = (
        alt.Chart(sub)
        .mark_bar()
        .encode(
            x=alt.X('avg_speed_ms:Q', title='Avg Speed (ms)'),
            y=alt.Y('userPrompt:N', title='Prompt', sort='-x'),
            color=alt.value('#6baed6'),
            tooltip=['userPrompt', alt.Tooltip('avg_speed_ms:Q', format='.1f')]
        )
        .properties(height=300)
    )
    st.altair_chart(speed_chart, use_container_width=True)

    # Token-usage chart
    token_chart = (
        alt.Chart(sub)
        .mark_bar()
        .encode(
            x=alt.X('avg_tokens:Q', title='Avg Tokens'),
            y=alt.Y('userPrompt:N', title='Prompt', sort='-x'),
            color=alt.value('#2171b5'),
            tooltip=['userPrompt', alt.Tooltip('avg_tokens:Q', format='.1f')]
        )
        .properties(height=300)
    )
    st.altair_chart(token_chart, use_container_width=True)

    st.markdown("---")
