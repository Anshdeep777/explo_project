import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Lunar Tribology AI", layout="wide")
st.title("🚀 Lunar Spacecraft Material Predictor (AI)")
st.write("Select your operating conditions. The AI will predict **friction (COF)** and **wear rate** in real-time.")

# ─────────────────────────────────────────────
# 2. TRAIN MODELS (cached – only runs once)
# ─────────────────────────────────────────────
@st.cache_resource
def build_ai_brain():
    df = pd.read_csv('ML_Ready_Lunar_Data.csv')
    df.columns = df.columns.str.strip()

    wear_col = next(c for c in df.columns if 'Wear Rate' in c)
    df['Lunar Contact'] = df['Lunar Contact'].str.strip().map({'Yes': 1, 'No': 0})

    df['Log_Wear_Rate'] = np.log10(
        pd.to_numeric(df[wear_col], errors='coerce').clip(lower=1e-12)
    )

    phys_cols = ['Density (g/cm³)',
                 'Thermal Conductivity (W/m·K)',
                 'Thermal Expansion (×10⁻⁶/°C)',
                 'Hardness (HV)']
    mat_props   = df.groupby('Material')[phys_cols].mean().to_dict('index')
    coat_props  = df.groupby('Coating Material')[phys_cols].mean().to_dict('index')
    global_mean = df[phys_cols].mean().to_dict()

    df_enc    = pd.get_dummies(df, columns=['Material', 'Coating Material'])
    drop_cols = ['COF', wear_col, 'Log_Wear_Rate']
    X         = df_enc.drop(columns=drop_cols)
    y_cof     = df_enc['COF']
    y_wear    = df_enc['Log_Wear_Rate']

    params = dict(n_estimators=300, max_depth=6, learning_rate=0.08,
                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                  tree_method='hist')
    xgb_cof  = xgb.XGBRegressor(**params)
    xgb_wear = xgb.XGBRegressor(**params)
    xgb_cof.fit(X,  y_cof)
    xgb_wear.fit(X, y_wear)

    materials = sorted(df['Material'].unique())
    coatings  = sorted(df['Coating Material'].unique())

    return (xgb_cof, xgb_wear, X.columns.tolist(),
            materials, coatings, mat_props, coat_props, global_mean)


with st.spinner("🔬 Training AI on 64 000+ lunar tribology data points…"):
    (model_cof, model_wear, expected_columns,
     all_materials, all_coatings,
     mat_props, coat_props, global_mean) = build_ai_brain()

st.success("✅ AI models ready!")

# ─────────────────────────────────────────────
# 3. DASHBOARD CONTROLS
# ─────────────────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🌕 Environment")
    env  = st.radio("Operating environment", ["Lunar Vacuum (Yes)", "Earth Ambient Air (No)"])
    temp = st.slider("Temperature (°C)", min_value=-173, max_value=127,
                     value=-50, step=5,
                     help="Lunar surface: −173 °C (night) to +127 °C (day)")

with col2:
    st.subheader("⚙️ Mechanical Conditions")
    load  = st.slider("Normal Load (N)", min_value=1, max_value=50, value=10, step=1)
    speed = st.slider("Sliding Speed (m/s)", min_value=0.001, max_value=1.0,
                      value=0.1, step=0.001, format="%.3f")

with col3:
    st.subheader("🔩 Material Selection")
    mat  = st.selectbox("Substrate Material", all_materials)
    coat = st.selectbox("Coating", all_coatings)

# ─────────────────────────────────────────────
# 4. BUILD INPUT VECTOR
# ─────────────────────────────────────────────
props    = mat_props.get(mat, global_mean)
input_df = pd.DataFrame(0.0, index=[0], columns=expected_columns)

input_df['Lunar Contact']       = 1 if "Yes" in env else 0
input_df['Temperature (°C)']    = float(temp)
input_df['Normal Load (N)']     = float(load)
input_df['Sliding Speed (m/s)'] = float(speed)

for prop_col, val in props.items():
    if prop_col in input_df.columns:
        input_df[prop_col] = val

mat_col  = f'Material_{mat}'
coat_col = f'Coating Material_{coat}'
if mat_col  in input_df.columns: input_df[mat_col]  = 1.0
if coat_col in input_df.columns: input_df[coat_col] = 1.0

# ─────────────────────────────────────────────
# 5. PREDICT
# ─────────────────────────────────────────────
pred_cof       = float(model_cof.predict(input_df)[0])
pred_wear_log  = float(model_wear.predict(input_df)[0])
pred_wear_real = 10 ** pred_wear_log
pred_cof       = max(pred_cof, 0.0)

# ─────────────────────────────────────────────
# 6. DISPLAY RESULTS
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🤖 AI Predictions")

m1, m2 = st.columns(2)
m1.metric(label="⚡ Coefficient of Friction (COF)", value=f"{pred_cof:.4f}",
          help="Dimensionless — lower means less friction")
m2.metric(label="🔬 Wear Rate (mm³/N·m)", value=f"{pred_wear_real:.3e}",
          help="Volumetric material loss per unit load and sliding distance")

# ─────────────────────────────────────────────
# 7. PERFORMANCE INTERPRETATION
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Performance Rating")

def rate_cof(v):
    if v < 0.05:  return "🟢 Excellent  — Ultra-low friction (DLC-class)"
    if v < 0.15:  return "🟡 Good       — Low friction (space-grade)"
    if v < 0.30:  return "🟠 Moderate   — Acceptable for non-critical joints"
    return              "🔴 High       — Investigate different coating"

def rate_wear(v):
    if v < 1e-9:  return "🟢 Excellent  — Ultra-low wear (< 1×10⁻⁹)"
    if v < 1e-7:  return "🟡 Good       — Low wear (space-grade)"
    if v < 1e-5:  return "🟠 Moderate   — Monitor during mission"
    return              "🔴 High       — Not recommended for long missions"

r1, r2 = st.columns(2)
r1.info(f"**Friction:**  {rate_cof(pred_cof)}")
r2.info(f"**Wear:**      {rate_wear(pred_wear_real)}")

if "Yes" in env:
    st.caption("ℹ️ Lunar vacuum significantly reduces oxidation — especially for MoS₂ and DLC coatings.")
else:
    st.caption("ℹ️ Earth ambient air introduces oxidation and moisture, increasing friction and wear vs vacuum.")

# ─────────────────────────────────────────────
# 8. MATERIAL PHYSICAL PROPERTIES
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("🧪 Material Physical Properties (dataset averages)")

phys_display = {
    "Density (g/cm³)":               round(props.get('Density (g/cm³)', 0), 3),
    "Thermal Conductivity (W/m·K)":  round(props.get('Thermal Conductivity (W/m·K)', 0), 3),
    "Thermal Expansion (×10⁻⁶/°C)": round(props.get('Thermal Expansion (×10⁻⁶/°C)', 0), 3),
    "Hardness (HV)":                 round(props.get('Hardness (HV)', 0), 1),
}
st.table(pd.DataFrame(phys_display.items(), columns=["Property", "Value"]))

# ─────────────────────────────────────────────
# 9. CONFIGURATION SUMMARY
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Full Configuration Summary")
summary = {
    "Environment":         env,
    "Temperature (°C)":    temp,
    "Normal Load (N)":     load,
    "Sliding Speed (m/s)": speed,
    "Substrate Material":  mat,
    "Coating":             coat,
    "Predicted COF":       f"{pred_cof:.4f}",
    "Predicted Wear Rate": f"{pred_wear_real:.3e} mm³/N·m",
}
st.table(pd.DataFrame(summary.items(), columns=["Parameter", "Value"]))

# ─────────────────────────────────────────────
# 10. RESEARCH-GRADE INTERACTIVE VISUALIZATIONS
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📈 Research-Grade Interactive Visualizations")
st.caption("💡 Hover over any point or bar to see exact values. Click legend items to show/hide. Drag to zoom, double-click to reset.")

@st.cache_data
def load_plot_data():
    df = pd.read_csv('ML_Ready_Lunar_Data.csv')
    df.columns = df.columns.str.strip()
    df['Lunar Contact'] = df['Lunar Contact'].str.strip()
    for col in ['Hardness (HV)', 'Temperature (°C)', 'COF',
                'Wear Rate (mm³/N·m)', 'Normal Load (N)', 'Sliding Speed (m/s)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df

raw_df = load_plot_data()
vac_df = raw_df[raw_df['Lunar Contact'] == 'Yes']
air_df = raw_df[raw_df['Lunar Contact'] == 'No']

MATERIALS = sorted(raw_df['Material'].unique())
COATINGS  = sorted(raw_df['Coating Material'].unique())
PALETTE   = ['#1f4e79','#c0392b','#1a7a4a','#7d3c98',
             '#b7770d','#117a8b','#5d4037','#2e86ab','#d35400']
TEMPS     = sorted(raw_df['Temperature (°C)'].unique())

tab1, tab2, tab3, tab4 = st.tabs([
    "🌡️ Hardness vs Temperature",
    "📊 COF & Wear Rate by Coating",
    "🔬 Hardness–COF Correlation",
    "📉 COF vs Temperature (All Materials)",
])

# ── TAB 1: Hardness vs Temperature ──────────────────────────────
with tab1:
    env_choice = st.radio("Show environment:", ["Lunar Vacuum", "Earth Air", "Both"],
                          horizontal=True, key="tab1_env")

    fig = go.Figure()

    for i, material in enumerate(MATERIALS):
        c = PALETTE[i % len(PALETTE)]
        mv = (vac_df[vac_df['Material'] == material]
              .groupby('Temperature (°C)')['Hardness (HV)']
              .agg(['mean', 'std'])
              .reindex(TEMPS)
              .reset_index())
        ma = (air_df[air_df['Material'] == material]
              .groupby('Temperature (°C)')['Hardness (HV)']
              .agg(['mean', 'std'])
              .reindex(TEMPS)
              .reset_index())

        if env_choice in ("Lunar Vacuum", "Both"):
            fig.add_trace(go.Scatter(
                x=mv['Temperature (°C)'],
                y=mv['mean'],
                name=f"{material} (Vac)" if env_choice == "Both" else material,
                legendgroup=material,
                mode='lines+markers',
                line=dict(color=c, width=2),
                marker=dict(size=6),
                error_y=dict(type='data', array=mv['std'],
                             visible=True, thickness=1, width=3),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Temp: %{x}°C<br>"
                    "Hardness: %{y:.1f} HV<br>"
                    "±1σ: %{error_y.array:.1f} HV"
                    "<extra></extra>"
                )
            ))
            # shaded ±1σ band
            fig.add_trace(go.Scatter(
                x=pd.concat([mv['Temperature (°C)'], mv['Temperature (°C)'][::-1]]),
                y=pd.concat([mv['mean'] + mv['std'], (mv['mean'] - mv['std'])[::-1]]),
                fill='toself',
                fillcolor=c,
                opacity=0.10,
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip',
                legendgroup=material,
            ))

        if env_choice in ("Earth Air", "Both"):
            fig.add_trace(go.Scatter(
                x=ma['Temperature (°C)'],
                y=ma['mean'],
                name=f"{material} (Air)" if env_choice == "Both" else material,
                legendgroup=material,
                mode='lines+markers',
                line=dict(color=c, width=1.5, dash='dash'),
                marker=dict(symbol='square', size=5),
                opacity=0.75 if env_choice == "Both" else 1.0,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Temp: %{x}°C<br>"
                    "Hardness: %{y:.1f} HV<br>"
                    "±1σ: %{error_y.array:.1f} HV"
                    "<extra></extra>"
                ),
                error_y=dict(type='data', array=ma['std'],
                             visible=True, thickness=1, width=3),
            ))

    # cryogenic / high-T shading via shapes
    fig.add_vrect(x0=-173, x1=0, fillcolor='steelblue', opacity=0.04,
                  layer='below', line_width=0,
                  annotation_text='Cryogenic', annotation_position='top left',
                  annotation_font_size=10, annotation_font_color='steelblue')
    fig.add_vrect(x0=0, x1=127, fillcolor='tomato', opacity=0.04,
                  layer='below', line_width=0,
                  annotation_text='High-T', annotation_position='top right',
                  annotation_font_size=10, annotation_font_color='tomato')

    fig.update_layout(
        title=dict(
            text='Temperature-Dependent Hardness of Spacecraft Structural Materials<br>'
                 '<sup>Solid ● = Lunar Vacuum | Dashed ■ = Earth Ambient | Shaded band = ±1σ</sup>',
            font=dict(size=14)
        ),
        xaxis_title='Temperature (°C)',
        yaxis_title='Vickers Hardness (HV)',
        hovermode='x unified',
        legend_title='Substrate Material',
        height=560,
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: COF & Wear Rate by Coating ───────────────────────────
with tab2:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('COF by Coating — Vacuum vs Air',
                        'Wear Rate by Coating — log scale'),
        horizontal_spacing=0.12
    )

    cof_vac  = (vac_df.groupby('Coating Material')['COF']
                .agg(['mean', 'sem']).reindex(COATINGS))
    cof_air  = (air_df.groupby('Coating Material')['COF']
                .agg(['mean', 'sem']).reindex(COATINGS))
    wear_vac = (vac_df.groupby('Coating Material')['Wear Rate (mm³/N·m)']
                .agg(['mean', 'sem']).reindex(COATINGS))
    wear_air = (air_df.groupby('Coating Material')['Wear Rate (mm³/N·m)']
                .agg(['mean', 'sem']).reindex(COATINGS))

    # COF bars
    fig.add_trace(go.Bar(
        x=COATINGS, y=cof_vac['mean'],
        name='Lunar Vacuum',
        error_y=dict(type='data', array=cof_vac['sem'], visible=True),
        marker_color='#1f4e79', opacity=0.88,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Environment: Lunar Vacuum<br>"
            "Mean COF: %{y:.4f}<br>"
            "±SEM: %{error_y.array:.4f}"
            "<extra></extra>"
        ),
        legendgroup='vac', showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=COATINGS, y=cof_air['mean'],
        name='Earth Ambient Air',
        error_y=dict(type='data', array=cof_air['sem'], visible=True),
        marker_color='#c0392b', opacity=0.88,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Environment: Earth Air<br>"
            "Mean COF: %{y:.4f}<br>"
            "±SEM: %{error_y.array:.4f}"
            "<extra></extra>"
        ),
        legendgroup='air', showlegend=True
    ), row=1, col=1)

    # Wear Rate bars
    fig.add_trace(go.Bar(
        x=COATINGS, y=wear_vac['mean'],
        name='Lunar Vacuum',
        error_y=dict(type='data', array=wear_vac['sem'], visible=True),
        marker_color='#1f4e79', opacity=0.88,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Environment: Lunar Vacuum<br>"
            "Wear Rate: %{y:.3e} mm³/N·m<br>"
            "±SEM: %{error_y.array:.3e}"
            "<extra></extra>"
        ),
        legendgroup='vac', showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=COATINGS, y=wear_air['mean'],
        name='Earth Ambient Air',
        error_y=dict(type='data', array=wear_air['sem'], visible=True),
        marker_color='#c0392b', opacity=0.88,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Environment: Earth Air<br>"
            "Wear Rate: %{y:.3e} mm³/N·m<br>"
            "±SEM: %{error_y.array:.3e}"
            "<extra></extra>"
        ),
        legendgroup='air', showlegend=False
    ), row=1, col=2)

    fig.update_yaxes(title_text='Mean COF', row=1, col=1)
    fig.update_yaxes(title_text='Mean Wear Rate (mm³ N⁻¹ m⁻¹)', type='log', row=1, col=2)
    fig.update_xaxes(tickangle=40)
    fig.update_layout(
        barmode='group',
        height=520,
        title='Tribological Performance of Aerospace Coatings: Environment Comparison<br>'
              '<sup>Error bars = ±SEM | n ≈ 64 783</sup>',
        legend=dict(font=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optional: % reduction annotation table
    pct_reductions = {}
    for coat in COATINGS:
        mv_val = cof_vac.loc[coat, 'mean'] if coat in cof_vac.index else np.nan
        ma_val = cof_air.loc[coat, 'mean'] if coat in cof_air.index else np.nan
        if ma_val and ma_val > 0:
            pct_reductions[coat] = f"{(ma_val - mv_val) / ma_val * 100:.1f}%"
        else:
            pct_reductions[coat] = "N/A"
    with st.expander("📋 COF reduction in vacuum vs air (per coating)"):
        st.table(pd.DataFrame(
            list(pct_reductions.items()),
            columns=["Coating", "COF Reduction (Air→Vacuum)"]
        ))

# ── TAB 3: Hardness–COF Scatter ─────────────────────────────────
with tab3:
    env3 = st.radio("Environment:", ["Lunar Vacuum", "Earth Ambient Air"],
                    horizontal=True, key="tab3_env")
    subset3 = vac_df if env3 == "Lunar Vacuum" else air_df

    fig = go.Figure()

    for i, material in enumerate(MATERIALS):
        sub = (subset3[subset3['Material'] == material]
               .sample(min(300, len(subset3[subset3['Material'] == material])),
                       random_state=42))
        fig.add_trace(go.Scatter(
            x=sub['Hardness (HV)'],
            y=sub['COF'],
            mode='markers',
            name=material,
            marker=dict(color=PALETTE[i % len(PALETTE)], size=6, opacity=0.5),
            customdata=sub[['Temperature (°C)', 'Normal Load (N)', 'Sliding Speed (m/s)']].values,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Hardness: %{x:.0f} HV<br>"
                "COF: %{y:.4f}<br>"
                "Temp: %{customdata[0]:.0f}°C<br>"
                "Load: %{customdata[1]:.0f} N<br>"
                "Speed: %{customdata[2]:.3f} m/s"
                "<extra></extra>"
            )
        ))

    # OLS trendline
    slope, intercept, r, p, _ = linregress(subset3['Hardness (HV)'], subset3['COF'])
    xline = np.linspace(subset3['Hardness (HV)'].min(), subset3['Hardness (HV)'].max(), 300)
    fig.add_trace(go.Scatter(
        x=xline,
        y=slope * xline + intercept,
        mode='lines',
        name=f'OLS fit  R²={r**2:.4f}',
        line=dict(color='black', width=2, dash='dash'),
        hovertemplate="OLS: y = %.5fx + %.4f<br>R² = %.4f<extra></extra>" % (slope, intercept, r**2)
    ))

    fig.add_annotation(
        x=0.04, y=0.96, xref='paper', yref='paper',
        text=f"R² = {r**2:.4f}<br>y = {slope:.5f}x + {intercept:.4f}<br>p < 0.001",
        showarrow=False,
        bgcolor='white', bordercolor='#aaa', borderwidth=1,
        font=dict(size=11), align='left'
    )

    fig.update_layout(
        title=f'Hardness–Friction Correlation  [{env3}]<br>'
              f'<sup>Archard Contact Theory Framework | n = 300 samples/material</sup>',
        xaxis_title='Vickers Hardness (HV)',
        yaxis_title='Coefficient of Friction (COF)',
        legend_title='Material',
        height=560,
        hovermode='closest',
    )
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: COF vs Temperature — 3×3 facet grid ──────────────────
with tab4:
    n_cols = 3
    n_rows = int(np.ceil(len(MATERIALS) / n_cols))

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=MATERIALS,
        shared_xaxes=True,
        vertical_spacing=0.10,
        horizontal_spacing=0.07
    )

    for i, material in enumerate(MATERIALS):
        row = i // n_cols + 1
        col = i % n_cols + 1
        show_legend = (i == 0)

        mv = (vac_df[vac_df['Material'] == material]
              .groupby('Temperature (°C)')['COF']
              .agg(['mean', 'std', 'sem'])
              .reindex(TEMPS)
              .reset_index())
        ma = (air_df[air_df['Material'] == material]
              .groupby('Temperature (°C)')['COF']
              .agg(['mean', 'std', 'sem'])
              .reindex(TEMPS)
              .reset_index())

        # Vacuum line
        fig.add_trace(go.Scatter(
            x=mv['Temperature (°C)'],
            y=mv['mean'],
            name='Lunar Vacuum',
            legendgroup='vac',
            showlegend=show_legend,
            mode='lines+markers',
            line=dict(color='#1f4e79', width=2),
            marker=dict(size=5),
            error_y=dict(type='data', array=mv['sem'],
                         visible=True, thickness=1, width=3),
            hovertemplate=(
                "<b>Lunar Vacuum</b><br>"
                f"Material: {material}<br>"
                "Temp: %{x}°C<br>"
                "Mean COF: %{y:.4f}<br>"
                "±SEM: %{error_y.array:.4f}"
                "<extra></extra>"
            )
        ), row=row, col=col)

        # Vacuum ±1σ band
        fig.add_trace(go.Scatter(
            x=pd.concat([mv['Temperature (°C)'], mv['Temperature (°C)'][::-1]]),
            y=pd.concat([mv['mean'] + mv['std'], (mv['mean'] - mv['std'])[::-1]]),
            fill='toself', fillcolor='#1f4e79',
            opacity=0.10, line=dict(color='rgba(255,255,255,0)'),
            showlegend=False, hoverinfo='skip', legendgroup='vac',
        ), row=row, col=col)

        # Air line
        fig.add_trace(go.Scatter(
            x=ma['Temperature (°C)'],
            y=ma['mean'],
            name='Earth Ambient Air',
            legendgroup='air',
            showlegend=show_legend,
            mode='lines+markers',
            line=dict(color='#c0392b', width=1.8, dash='dash'),
            marker=dict(symbol='square', size=4.5),
            error_y=dict(type='data', array=ma['sem'],
                         visible=True, thickness=1, width=3),
            hovertemplate=(
                "<b>Earth Ambient Air</b><br>"
                f"Material: {material}<br>"
                "Temp: %{x}°C<br>"
                "Mean COF: %{y:.4f}<br>"
                "±SEM: %{error_y.array:.4f}"
                "<extra></extra>"
            )
        ), row=row, col=col)

        # Air ±1σ band
        fig.add_trace(go.Scatter(
            x=pd.concat([ma['Temperature (°C)'], ma['Temperature (°C)'][::-1]]),
            y=pd.concat([ma['mean'] + ma['std'], (ma['mean'] - ma['std'])[::-1]]),
            fill='toself', fillcolor='#c0392b',
            opacity=0.08, line=dict(color='rgba(255,255,255,0)'),
            showlegend=False, hoverinfo='skip', legendgroup='air',
        ), row=row, col=col)

        # 0°C vertical line per panel
        fig.add_vline(x=0, line=dict(color='gray', dash='dot', width=0.8),
                      row=row, col=col)

    # Bottom row x-axis labels
    for c in range(1, n_cols + 1):
        fig.update_xaxes(title_text='Temperature (°C)', row=n_rows, col=c)

    # All y-axes
    for r in range(1, n_rows + 1):
        fig.update_yaxes(title_text='COF', row=r, col=1)

    fig.update_layout(
        height=950,
        title=dict(
            text='Coefficient of Friction vs Temperature — All Substrate Materials<br>'
                 '<sup>Lunar Vacuum (—●) vs Earth Ambient Air (- -■) | Shaded: ±1σ | Error bars: ±SEM | Dotted line: 0°C</sup>',
            font=dict(size=13)
        ),
        showlegend=True,
        legend=dict(font=dict(size=11), x=1.01, y=1.0),
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)