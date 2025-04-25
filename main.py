import streamlit as st
import pandas as pd
import altair as alt
import json
import numpy as np
import plotly.graph_objects as go


# Sett sidelayout til bred
st.set_page_config(layout="wide", page_title="Benchmark-sammenligning")

# Last inn benchmark‑data (lagre JSON‑filen som 'benchmark_data.json' i samme mappe)
with open('benchmark_data.json', 'r') as f:
    data = json.load(f)

# Hent tilgjengelige modeller for valg
all_models = [cfg['navigationConfiguration']['model'] for cfg in data['configurations']]
# Fjern duplikater og behold rekkefølgen
models = list(dict.fromkeys(all_models))

# ---- Sidepanel: modellfilter ----
st.sidebar.header("Filtrer modeller")
selected_models = st.sidebar.multiselect(
    "Velg språkmodeller som skal vises:",
    options=models,
    default=models
)

# Flate ut den nøstede strukturen til en DataFrame, inkludert modellnavn
a = []
for cfg in data['configurations']:
    model_name = cfg['navigationConfiguration']['model']
    tech = cfg['navigationConfiguration']['navigationTechnique']
    nav_val = cfg['navigationConfiguration']['navigationValue']
    for bm in cfg['benchmarkAnswers']:
        prompt_text = bm['userPrompt']
        for resp in bm['responses']:
            a.append({
                'model': model_name,
                'navigationTechnique': tech,
                'navigationValue': str(nav_val),
                'userPrompt': prompt_text,
                'result': resp.get('result'),
                'milliseconds': resp.get('milliseconds'),
                'totalTokens': resp['tokenUsage'].get('totalTokens')
            })

df = pd.DataFrame(a)

# --- Filtrer bort feilresponser ---
df_3d = df[df['result'] != 'error']

import plotly.express as px

# ---- Prissetting per token ----


# ---- 3D‑plot av ytelse‑kostnad‑planet ----



# ---- Sidepanel: brukerdefinerte filtre ----
st.sidebar.header("Filterkriterier")
min_acc_pct = st.sidebar.slider(
    "Minimum nøyaktighet (%)", 0, 100, 70, step=1
)
max_time_sec = st.sidebar.slider(
    "Maksimum gjennomsnittlig svartid (s)", 0, 60, 20, step=1
)
max_time_ms = max_time_sec * 1000  # konverter til ms

# Valgfri sorteringsnøkkel
sort_key = st.sidebar.selectbox(
    "Sorter rangeringen etter",
    ["distance", "avg_accuracy", "avg_speed_ms", "cost"],
    index=0
)

# ---- Sidepanel: tokenpriser ----
st.sidebar.header("Token‑prising")

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
        f"Pris per token for {model}",
        min_value=0.0,
        value=initial_prices.get(model, 0.001),
        format="%.8f"
    )
# --- Beregn ytelse og kostnad pr modell ---
model_perf = (
    df_3d
    .groupby('model', as_index=False)
    .agg(
        avg_speed_ms=('milliseconds', 'mean'),
        avg_accuracy=('result', lambda x: x.isin(['correct', 'technicallyCorrect']).mean()*100),
        avg_tokens=('totalTokens', 'mean')
    )
)
# Legg til kostnadskolonne (tokens × pris)
model_perf['cost'] = model_perf.apply(
    lambda row: row['avg_tokens'] * price_per_token[row['model']],
    axis=1
)

# 1) Estimer planet (kostnad = a*speed + b*accuracy + d)
A = np.c_[
    model_perf['avg_speed_ms'],
    model_perf['avg_accuracy'],
    np.ones(len(model_perf))
]
C, *_ = np.linalg.lstsq(A, model_perf['cost'], rcond=None)
a, b, d = C

# 2) Finmasket rutenett innenfor dataområdene
speed_lin = np.linspace(model_perf['avg_speed_ms'].min(), model_perf['avg_speed_ms'].max(), 30)
acc_lin = np.linspace(model_perf['avg_accuracy'].min(), model_perf['avg_accuracy'].max(), 30)
S, A_ = np.meshgrid(speed_lin, acc_lin)

# 3) Beregn planet Z og avgrens innenfor faktiske kostnadsgrenser
Z_raw = a * S + b * A_ + d
min_cost, max_cost = model_perf['cost'].min(), model_perf['cost'].max()
Z = np.clip(Z_raw, min_cost, max_cost)

# 4) Lag et 2D‑felt for grønn intensitet (hurtig, nøyaktig, billig → grønn)
norm_s = 1 - (S - S.min()) / (S.max() - S.min())
norm_a = (A_ - A_.min()) / (A_.max() - A_.min())
norm_z = 1 - (Z - min_cost) / (max_cost - min_cost)
green_intensity = norm_s * norm_a * norm_z

# 5) Plot
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
        xaxis_title='Gj.sn. hastighet (ms)',
        yaxis_title='Nøyaktighet (%)',
        zaxis=dict(
            title='Kostnad',
            autorange='reversed'
        )
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)

# Beregn det optimale hjørnet
opt_speed = model_perf['avg_speed_ms'].min()
opt_accuracy = model_perf['avg_accuracy'].max()
opt_cost = model_perf['cost'].min()

# Legg til punktet for optimal balanse
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
        name='Optimalt'
    )
)

st.plotly_chart(fig, use_container_width=True)

# ---- Filtrer modellprestasjon etter brukerkrav ----
perf_filtered = model_perf[
    (model_perf['avg_accuracy'] >= min_acc_pct) &
    (model_perf['avg_speed_ms'] <= max_time_ms)
].copy()

if perf_filtered.empty:
    st.warning("Ingen modeller oppfyller kriteriene.")
else:
    # Normaliser og beregn avstand
    s_min, s_max = perf_filtered['avg_speed_ms'].min(), perf_filtered['avg_speed_ms'].max()
    a_min, a_max = perf_filtered['avg_accuracy'].min(), perf_filtered['avg_accuracy'].max()
    c_min, c_max = perf_filtered['cost'].min(), perf_filtered['cost'].max()

    perf_filtered['speed_n'] = (s_max - perf_filtered['avg_speed_ms']) / (s_max - s_min)
    perf_filtered['accuracy_n'] = (perf_filtered['avg_accuracy'] - a_min) / (a_max - a_min)
    perf_filtered['cost_n'] = (c_max - perf_filtered['cost']) / (c_max - c_min)

    perf_filtered['distance'] = np.sqrt(
        (1 - perf_filtered['speed_n'])**2 +
        (1 - perf_filtered['accuracy_n'])**2 +
        (1 - perf_filtered['cost_n'])**2
    )

    # Sorter etter valgt nøkkel
    perf_filtered = perf_filtered.sort_values(sort_key)

    best = perf_filtered.iloc[0]

    # st.subheader("Den beste modellen er:")
    # st.header(f"**{best['model']}**")

    st.table(
        perf_filtered[[
            'model', 'avg_speed_ms', 'avg_accuracy', 'cost', 'distance'
        ]]
        .assign(
            avg_speed_ms=lambda df: (df['avg_speed_ms']/1000).round(2).astype(str)+" s",
            avg_accuracy=lambda df: df['avg_accuracy'].round(1).astype(str)+" %",
            cost=lambda df: df['cost'].round(6),
            distance=lambda df: df['distance'].round(4)
        )
        .rename(columns={
            'avg_speed_ms': 'Gj.sn. tid',
            'avg_accuracy': 'Nøyaktighet',
            'cost': 'Kostnad',
        })
    )


# ---- Ønsket plan: hastighet vs nøyaktighet pr modell ----
model_perf = (
    df
    .groupby('model', as_index=False)
    .agg(
        avg_speed_ms=('milliseconds', 'mean'),
        avg_accuracy=('result', lambda x: x.isin(['correct','technicallyCorrect']).mean()*100)
    )
)

st.header("Modellprestasjon: Hastighet vs Nøyaktighet")

points = alt.Chart(model_perf).mark_circle(size=100).encode(
    x=alt.X('avg_speed_ms:Q', title='Gj.sn. hastighet (ms)'),
    y=alt.Y('avg_accuracy:Q', title='Nøyaktighet (%)'),
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

# ---- Aggregert statistikk pr navigeringstype ----
agg_overall = df.groupby('navigationTechnique', as_index=False).agg(
    avg_speed_ms=('milliseconds', 'mean'),
    avg_tokens=('totalTokens', 'mean'),
    avg_accuracy=('result', lambda x: x.isin(['correct', 'technicallyCorrect']).mean() * 100)
)

st.title("ChatSSB – Ytelse")
st.header("Generell oversikt over navigeringstype")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Gjennomsnittlig hastighet (ms)")
    chart_speed = alt.Chart(agg_overall).mark_bar().encode(
        x=alt.X('navigationTechnique:N', title='Teknikk'),
        y=alt.Y('avg_speed_ms:Q', title='Hastighet (ms)'),
        color=alt.Color('navigationTechnique:N', legend=None),
        tooltip=['navigationTechnique', 'avg_speed_ms']
    ).properties(height=300)
    st.altair_chart(chart_speed, use_container_width=True)

with col2:
    st.subheader("Gjennomsnittlig tokenforbruk")
    chart_tokens = alt.Chart(agg_overall).mark_bar().encode(
        x=alt.X('navigationTechnique:N', title='Teknikk'),
        y=alt.Y('avg_tokens:Q', title='Totalt antall tokens'),
        color=alt.Color('navigationTechnique:N', legend=None),
        tooltip=['navigationTechnique', 'avg_tokens']
    ).properties(height=300)
    st.altair_chart(chart_tokens, use_container_width=True)

with col3:
    st.subheader("Gjennomsnittlig nøyaktighet (%)")
    chart_accuracy = alt.Chart(agg_overall).mark_bar().encode(
        x=alt.X('navigationTechnique:N', title='Teknikk'),
        y=alt.Y('avg_accuracy:Q', title='Nøyaktighet (%)'),
        color=alt.Color('navigationTechnique:N', legend=None),
        tooltip=['navigationTechnique', alt.Tooltip('avg_accuracy:Q', format='.1f')]
    ).properties(height=300)
    st.altair_chart(chart_accuracy, use_container_width=True)

st.markdown("---")

# ---- Aggregert pr konfigurasjon ----
agg_config = (
    df
    .groupby(['model', 'navigationTechnique', 'navigationValue'], as_index=False)
    .agg(
        total_speed_ms=('milliseconds', 'sum'),
        total_tokens=('totalTokens', 'sum'),
        accuracy=('result', lambda x: x.isin(['correct', 'technicallyCorrect']).mean() * 100)
    )
)
agg_config['config'] = (
    agg_config['model'] + ' | ' + agg_config['navigationTechnique'] + ' (' + agg_config['navigationValue'] + ')'
)

st.header("Aggregert per konfigurasjon")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Total hastighet (ms)")
    chart1 = alt.Chart(agg_config).mark_bar().encode(
        x=alt.X('config:N', title='Konfigurasjon', axis=alt.Axis(labelAngle=-90, labelAlign='right', labelLimit=500)),
        y=alt.Y('total_speed_ms:Q', title='Total hastighet (ms)'),
        color=alt.Color('model:N', legend=None),
        tooltip=['config', 'total_speed_ms']
    ).properties(height=550)
    st.altair_chart(chart1, use_container_width=True)

with col2:
    st.subheader("Totalt antall tokens")
    chart2 = alt.Chart(agg_config).mark_bar().encode(
        x=alt.X('config:N', title='Konfigurasjon', axis=alt.Axis(labelAngle=-90, labelAlign='right', labelLimit=500)),
        y=alt.Y('total_tokens:Q', title='Totalt antall tokens'),
        color=alt.Color('model:N', legend=None),
        tooltip=['config', 'total_tokens']
    ).properties(height=550)
    st.altair_chart(chart2, use_container_width=True)

with col3:
    st.subheader("Nøyaktighet (%)")
    chart3 = alt.Chart(agg_config).mark_bar().encode(
        x=alt.X('config:N', title='Konfigurasjon', axis=alt.Axis(labelAngle=-90, labelAlign='right', labelLimit=500)),
        y=alt.Y('accuracy:Q', title='Nøyaktighet (%)'),
        color=alt.Color('model:N', legend=None),
        tooltip=['config', alt.Tooltip('accuracy:Q', format='.1f')]
    ).properties(height=550)
    st.altair_chart(chart3, use_container_width=True)

# ---- GRAF PER PROMPT OG MODELL ----
prompts = df['userPrompt'].unique()
blue_shades = ['#c6dbef', '#6baed6', '#2171b5']
domain_values = ['1', '3', '5']

for prompt in prompts:
    st.header(f" Spørsmål: {prompt}")
    sub_df = df[df['userPrompt'] == prompt]

    for m in selected_models:
        m_df = sub_df[sub_df['model'] == m]
        if m_df.empty:
            continue

        st.subheader(f"{m}")

        # Hastighetsdiagram
        speed_chart = alt.Chart(m_df).mark_bar().encode(
            x=alt.X('milliseconds:Q', title='Hastighet (ms)'),
            y=alt.Y('navigationTechnique:N', title='Teknikk'),
            stroke=alt.Stroke('result:N', scale=alt.Scale(domain=['correct', 'technicallyCorrect', 'incorrect', 'error'], range=['#06bd36', 'yellow', 'red', 'purple'])),
            strokeWidth=alt.value(4),
            color=alt.Color('navigationValue:N', title='Nav‑verdi', scale=alt.Scale(domain=domain_values, range=blue_shades)),
            tooltip=['navigationTechnique', 'navigationValue', 'result', 'milliseconds', 'totalTokens']
        ).properties(width=800, height=200)
        st.altair_chart(speed_chart, use_container_width=True)

        # Tokenforbruksdiagram
        token_chart = alt.Chart(m_df).mark_bar().encode(
            x=alt.X('totalTokens:Q', title='Totalt antall tokens'),
            y=alt.Y('navigationTechnique:N', title='Teknikk'),
            stroke=alt.Stroke('result:N', scale=alt.Scale(domain=['correct', 'technicallyCorrect', 'incorrect', 'error'], range=['#06bd36', 'yellow', 'red', 'purple'])),
            strokeWidth=alt.value(4),
            color=alt.Color('navigationValue:N', title='Nav‑verdi', scale=alt.Scale(domain=domain_values, range=blue_shades)),
            tooltip=['navigationTechnique', 'navigationValue', 'result', 'totalTokens']
        ).properties(width=800, height=200)
        st.altair_chart(token_chart, use_container_width=True)

        # Detaljerte responser
        with st.expander(f"{m} svar"):
            st.write(m_df[['navigationTechnique', 'navigationValue', 'result', 'milliseconds', 'totalTokens']])

    st.markdown("---")

# ---- KONFIGURASJON-FØRST: per spørsmål‑grafer ----
agg_q = (
    df
    .groupby(['model', 'navigationTechnique', 'navigationValue', 'userPrompt'], as_index=False)
    .agg(
        avg_speed_ms=('milliseconds', 'mean'),
        avg_tokens=('totalTokens', 'mean'),
        avg_accuracy=('result', lambda x: x.isin(['correct','technicallyCorrect']).mean()*100)
    )
)

st.header("Ytelse per spørsmål per konfigurasjon")
for _, cfg in agg_config.iterrows():
    model = cfg['model']
    tech = cfg['navigationTechnique']
    nav = cfg['navigationValue']
    title = f"{model} | {tech} ({nav})"
    st.subheader(title)

    sub = agg_q[(agg_q['model'] == model) & (agg_q['navigationTechnique'] == tech) & (agg_q['navigationValue'] == nav)]

    # Hastighet
    speed_chart = (
        alt.Chart(sub).mark_bar().encode(
            x=alt.X('avg_speed_ms:Q', title='Gj.sn. hastighet (ms)'),
            y=alt.Y('userPrompt:N', title='Spørsmål', sort='-x'),
            color=alt.value('#6baed6'),
            tooltip=['userPrompt', alt.Tooltip('avg_speed_ms:Q', format='.1f')]
        ).properties(height=300)
    )
    st.altair_chart(speed_chart, use_container_width=True)

    # Tokenforbruk
    token_chart = (
        alt.Chart(sub).mark_bar().encode(
            x=alt.X('avg_tokens:Q', title='Gj.sn. tokens'),
            y=alt.Y('userPrompt:N', title='Spørsmål', sort='-x'),
            color=alt.value('#2171b5'),
            tooltip=['userPrompt', alt.Tooltip('avg_tokens:Q', format='.1f')]
        ).properties(height=300)
    )
    st.altair_chart(token_chart, use_container_width=True)

    st.markdown("---")
