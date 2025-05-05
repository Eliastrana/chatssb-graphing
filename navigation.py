import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import pyplot as plt

MODEL_NAME_MAP = {
    "gemini-2.0-flash": "Gemini Flash 2",
    "gemini-2.0-flash-lite": "Gemini Flash 2 Lite",
    "gpt-4.1-2025-04-14": "GPT-4.1",
    "gpt-4.1-mini-2025-04-14": "GPT-4.1 Mini",
    "gpt-4.1-nano-2025-04-14": "GPT-4.1 Nano",
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick",
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout",
    "deepseek-r1-distill-llama-70b": "DeepSeek R1 70B",
}

# IMPORTANT: INPUT PRICES directly from API provider
INPUT_PRICES = {
    "Gemini Flash 2": 0.1e-6,
    "Gemini Flash 2 Lite": 0.07e-6,
    "GPT-4.1": 2.00e-6,
    "GPT-4.1 Mini": 0.4e-6,
    "GPT-4.1 Nano": 0.1e-6,
    "Llama 3.3 70B": 0.59e-6,
    "Llama 4 Maverick": 0.2e-6,
    "Llama 4 Scout": 0.11e-6,
    "DeepSeek R1 70B": 0.75e-6,
}


def to_rgba_string(rgb_tuple):
    r, g, b, a = rgb_tuple
    return f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})'


cmap = plt.cm.get_cmap('tab10', len(MODEL_NAME_MAP))
color_map = {
    model: to_rgba_string(cmap(i)) for i, model in enumerate(MODEL_NAME_MAP.values())
}


# ---- Utility Functions ----
def load_data(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def flatten_data(data):
    """Flatten nested JSON data into a DataFrame."""
    records = []
    for config in data['configurations']:
        model_name = config['navigationConfiguration']['model']
        tech = config['navigationConfiguration']['navigationTechnique']
        nav_val = config['navigationConfiguration']['navigationValue']
        for bm in config['benchmarkAnswers']:
            prompt_text = bm['userPrompt']
            for resp in bm['responses']:
                records.append({
                    'model': model_name,
                    'navigationTechnique': tech,
                    'navigationValue': nav_val,
                    'userPrompt': prompt_text,
                    'result': resp.get('result'),
                    'timeSeconds': resp.get('milliseconds') / 1000 if resp.get('milliseconds') is 
                                                                     not None else None,
                    'promptTokens': resp['tokenUsage'].get('promptTokens')
                })
    return pd.DataFrame(records)


def calculate_and_rank_models(df, price_per_token):
    """Calculate performance metrics and rank models."""
    # Calculate performance metrics
    model_performance = (
        df
        .groupby(['model', 'navigationTechnique', 'navigationValue'], as_index=False)
        .agg(
            avg_speed=('timeSeconds', lambda x: x[df['result'] != 'error'].mean()),
            avg_accuracy=(
            'result', lambda x: x.isin(['correct', 'technicallyCorrect']).mean() * 100),
            avg_tokens=('promptTokens', lambda x: x[df['result'] != 'error'].mean())
        )
    )

    model_performance['cost'] = model_performance.apply(
        lambda row: row['avg_tokens'] * price_per_token[row['model']],
        axis=1
    )

    if model_performance.empty:
        return None

    # Normalize and calculate distance
    s_min, s_max = model_performance['avg_speed'].min(), model_performance['avg_speed'].max()
    a_min, a_max = model_performance['avg_accuracy'].min(), model_performance['avg_accuracy'].max()
    c_min, c_max = model_performance['cost'].min(), model_performance['cost'].max()

    model_performance['speed_n'] = (s_max - model_performance['avg_speed']) / (s_max - s_min)
    model_performance['accuracy_n'] = (model_performance['avg_accuracy'] - a_min) / (a_max - a_min)
    model_performance['cost_n'] = (c_max - model_performance['cost']) / (c_max - c_min)

    model_performance['distance'] = np.sqrt(
        (1 - model_performance['speed_n']) ** 2 +
        (1 - model_performance['accuracy_n']) ** 2 +
        (1 - model_performance['cost_n']) ** 2
    ) / 3

    return model_performance


def plot_3d_performance(model_perf):
    """Create a 3D plot of model performance."""
    fig = go.Figure()

    # Add model performance points
    fig.add_trace(go.Scatter3d(
        x=model_perf['avg_speed'],
        y=model_perf['cost'],
        z=model_perf['avg_accuracy'],
        mode='markers',
        marker=dict(
            size=6,
            color=model_perf['model'].map(color_map),
            symbol=model_perf['navigationTechnique'].apply(
                lambda x: 'circle' if x == 'keywordSearch' else 'square'),
        ),
        hovertext=model_perf.apply(
            lambda row: f"Model: {row['model']}<br>Technique: {row['navigationTechnique']}_"
                        f"{row['navigationValue']}<br>"
                        f"<br>Speed: {row['avg_speed']} s<br>Accuracy: "
                        f"{row['avg_accuracy']}%<br>Cost: ${row['cost']:.6f}<br>Normalized Distance: "
                        f"{row['distance']:.4f}",
            axis=1
        ),
        hoverinfo="text"
    ))

    # Beregn det optimale hjørnet
    opt_speed = model_perf['avg_speed'].min()
    opt_accuracy = model_perf['avg_accuracy'].max()
    opt_cost = model_perf['cost'].min()

    # Legg til punktet for optimal balanse
    fig.add_trace(
        go.Scatter3d(
            x=[opt_speed],
            y=[opt_cost],
            z=[opt_accuracy],
            mode='markers',
            marker=dict(
                size=8,
                color='green',
                symbol='diamond'
            ),
            name='Optimalt'
        )
    )

    # Add layout details
    fig.update_layout(
        scene=dict(
            xaxis_title='Avg. Speed (s)',
            yaxis_title='Cost ($)',
            zaxis_title='Accuracy (%)'
        )
    )

    return fig


def plot_2d_performance(model_perf):
    """Create a 2D plot of model performance."""
    fig = go.Figure()

    # Add model performance points
    fig.add_trace(go.Scatter(
        x=model_perf['avg_speed'],
        y=model_perf['avg_accuracy'],
        mode='markers',
        marker=dict(
            size=10,
            color=model_perf['model'].map(color_map),
            symbol=model_perf['navigationTechnique'].apply(
                lambda x: 'circle' if x == 'keywordSearch' else 'square'
            ),
        ),
        hovertext=model_perf.apply(
            lambda row: f"Model: {row['model']}<br>Technique: {row['navigationTechnique']}_"
                        f"{row['navigationValue']}<br>"
                        f"<br>Speed: {row['avg_speed']} s<br>Accuracy: "
                        f"{row['avg_accuracy']}%<br>Cost: ${row['cost']:.6f}<br>Normalized Distance: "
                        f"{row['distance']:.4f}",
            axis=1
        ),
        hoverinfo="text"
    ))

    fig.update_layout(
        xaxis=dict(
            title='Avg. Speed (s)',
            showgrid=True
        ),
        yaxis=dict(
            title='Accuracy (%)',
            showgrid=True
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig


def filter_models(model_perf, min_acc_pct, max_time):
    """Filter models based on user-defined criteria."""
    return model_perf[
        (model_perf['avg_accuracy'] >= min_acc_pct) &
        (model_perf['avg_speed'] <= max_time)
        ].copy()


def show_dataframe(df):
    if df is None:
        st.warning("No models meet the criteria.")
    else:
        st.dataframe(
            df[[
                'model', 'navigationTechnique', 'navigationValue', 'avg_speed', 'avg_accuracy',
                'cost', 'distance'
            ]]
            .assign(
                avg_speed=lambda df: (df['avg_speed']).round(2),
                avg_accuracy=lambda df: df['avg_accuracy'].round(1),
                cost=lambda df: df['cost'].round(6),
                distance=lambda df: df['distance'].round(4)
            )
            .rename(columns={
                'avg_speed': 'Avg. Time (s)',
                'avg_accuracy': 'Accuracy (%)',
                'cost': 'Cost ($)',
                'distance': 'Norm Distance'
            })
            .style.apply(
                lambda x: ['background-color: {}'.format(color_map.get(x['model'], 'white')) for i in range(len(x))],
                axis=1,
            )
        )


def plot_horizontal_barchart(df):
    """Create a horizontal stacked bar chart of result proportions per model."""
    # Define the categories in the order you want them to appear
    result_categories = ['error', 'incorrect', 'technicallyCorrect', 'correct']

    # 1) Count and 2) Pivot into a Model × Result table
    counts = pd.crosstab(df['model'], df['result']).reindex(columns=result_categories, fill_value=0)

    # 3) Normalize each row to sum to 1
    proportions = counts.div(counts.sum(axis=1), axis=0)

    colors = {
        'correct': '#06bd36',
        'technicallyCorrect': 'yellow',
        'incorrect': 'red',
        'error': 'purple'
    }

    # Build the stacked horizontal bars
    fig = go.Figure()
    models = proportions.index.tolist()

    for category in result_categories:
        fig.add_trace(go.Bar(
            y=models,
            x=proportions[category],
            name=category.replace('technicallyCorrect', 'Technically Correct').capitalize(),
            orientation='h',
            marker=dict(
                color=colors[category]
            ),
            hovertemplate='%{y}<br>' + category + ': %{x:.0%}<extra></extra>'
        ))

    fig.update_layout(
        barmode='stack',
        title="Result Distribution by Model, all techniques",
        xaxis=dict(title="Proportion", tickformat=".0%"),
        yaxis=dict(title="Model", automargin=True),
        legend=dict(title="Outcome"),
        margin=dict(l=120, r=20, t=50, b=50),
        height=400 + 30*len(models)
    )

    return fig




def plot_speed_floating_bar(df):
    # 1) Compute descriptive stats per model
    stats = (
        df
        .groupby('model')['timeSeconds']
        .agg(
            p10=lambda x: np.percentile(x, 10),
            p25=lambda x: np.percentile(x, 25),
            median=lambda x: np.percentile(x, 50),
            p75=lambda x: np.percentile(x, 75),
            p90=lambda x: np.percentile(x, 90),
            mean='mean'
        )
        .reset_index()
    )

    # 2) Build the floating bars
    fig = go.Figure()
    # – 10–90 percentile range
    fig.add_trace(go.Bar(
        y=stats['model'],
        x=stats['p90'] - stats['p10'],
        base=stats['p10'],
        orientation='h',
        name='10th–90th pctile',
        marker=dict(color='lightsteelblue'),
        hovertemplate='10th–90th pctile: %{x:.2f}<extra></extra>'
    ))
    # – 25–75 percentile range
    fig.add_trace(go.Bar(
        y=stats['model'],
        x=stats['p75'] - stats['p25'],
        base=stats['p25'],
        orientation='h',
        name='25th–75th pctile',
        marker=dict(color='steelblue'),
        hovertemplate='25th–75th pctile: %{x:.2f}<extra></extra>'
    ))
    # – Median marker
    fig.add_trace(go.Scatter(
        y=stats['model'],
        x=stats['median'],
        mode='markers',
        name='Median',
        marker_symbol='line-ns-open',
        marker=dict(color='black', size=12),
        hovertemplate='Median: %{x:.2f}<extra></extra>'
    ))
    # – Mean marker
    fig.add_trace(go.Scatter(
        y=stats['model'],
        x=stats['mean'],
        mode='markers',
        name='Mean',
        marker_symbol='circle-open-dot',
        marker=dict(color='firebrick', size=8),
        hovertemplate='Mean: %{x:.2f}<extra></extra>'
    ))

    # 3) Layout tweaks
    fig.update_layout(
        title='Speed Distribution by Model\n(Percentiles with Mean & Median)',
        xaxis=dict(
            title='time (s)',
            showgrid=True,  # Enable gridlines for the x-axis
            gridcolor='gray',  # Optional: Set gridline color
            gridwidth=0.5,  # Optional: Set gridline width
            nticks = 20,
        ),
        yaxis=dict(
            title='Model',
            showgrid=True,  # Enable gridlines for the y-axis
            gridcolor='gray',  # Optional: Set gridline color
            gridwidth=0.5  # Optional: Set gridline width
        ),
        barmode='overlay',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=120, t=80, b=40),
        height=400 + 40 * len(stats),
    )
    
    return fig


# ---- Streamlit App ----
def main():
    st.set_page_config(layout="wide", page_title="Benchmark Comparison")

    # Load data
    data = load_data('data/navigationAnswers.json')
    original_df = flatten_data(data)
    df = flatten_data(data)

    # Update model names
    original_df['model'] = original_df['model'].replace(MODEL_NAME_MAP)
    df['model'] = df['model'].replace(MODEL_NAME_MAP)

    original_df = original_df.assign(
        color=original_df['model'].map(color_map)
    )
    df = df.assign(
        color=df['model'].map(color_map)
    )

    # SIDEBAR CONFIGURATION
    # Sidebar: Navigation Technique selection
    all_techniques = list(dict.fromkeys(original_df['navigationTechnique'].unique()))
    st.sidebar.header("Filter Navigation Techniques")
    selected_techniques = st.sidebar.multiselect("Select navigation techniques to display:",
                                                 options=all_techniques, default=all_techniques)

    # Sidebar: Navigation Value selection
    all_values = list(dict.fromkeys(original_df['navigationValue'].unique()))
    st.sidebar.header("Filter Navigation Values")
    selected_values = st.sidebar.multiselect("Select navigation values to display:",
                                             options=all_values, default=all_values)
    df = df[df['navigationValue'].isin(selected_values)]

    # Filter DataFrame based on selected techniques
    df = df[df['navigationTechnique'].isin(selected_techniques)]

    # Sidebar: Model selection
    all_models = list(dict.fromkeys(original_df['model'].unique()))
    st.sidebar.header("Filter Models")
    selected_models = st.sidebar.multiselect("Select models to display:", options=all_models,
                                             default=all_models)

    # Filter DataFrame based on selected models
    df = df[df['model'].isin(selected_models)]

    # Sidebar: User-defined filters
    st.sidebar.header("Filter Criteria")
    min_acc_pct = st.sidebar.slider("Minimum Accuracy (%)", 0, 100, 0, step=1)
    max_time_sec = st.sidebar.slider("Maximum Avg. Response Time (s)", 0, 60, 60, step=1)

    # Sidebar: Token pricing
    st.sidebar.header("Token Pricing")
    price_per_token = {
        model: st.sidebar.number_input(
            f"Price per token for {model}",
            min_value=0.0,
            value=INPUT_PRICES.get(model, 0.001),
            format="%.8f"
        )
        for model in all_models
    }

    # Calculate performance
    df = calculate_and_rank_models(df, price_per_token)

    # Filter models based on user-defined criteria
    df = filter_models(df, min_acc_pct, max_time_sec)

    # 3D Plot
    st.plotly_chart(plot_3d_performance(df), use_container_width=True)

    show_dataframe(df)

    # 2D Plot
    st.plotly_chart(plot_2d_performance(df), use_container_width=True)

    # Bar chart of model accuracy aggregated by navigation technique
    st.plotly_chart(plot_horizontal_barchart(original_df), use_container_width=True)
    
    # Faceted bar charts of model accuracy aggregated by navigation technique and value
    st.plotly_chart(plot_faceted_barcharts(original_df), use_container_width=True)

    st.plotly_chart(
        plot_speed_floating_bar(original_df),
        use_container_width=True
    )


if __name__ == "__main__":
    main()
