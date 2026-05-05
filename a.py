import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
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
    df = pd.read_csv('Lunar_Tribology_ML_Dataset__1_.csv')
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

    temp_mat_props = (
        df.groupby(['Material', 'Temperature (°C)'])[phys_cols].mean()
    )
    temp_mat_std = (
        df.groupby(['Material', 'Temperature (°C)'])[phys_cols].std()
    )

    # Raw COF grouped by material+speed for overlay
    raw_cof_speed = (
        df.groupby(['Material', 'Sliding Speed (m/s)'])['COF']
        .agg(['mean', 'std']).reset_index()
    )

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
            materials, coatings, mat_props, coat_props,
            global_mean, temp_mat_props, temp_mat_std,
            raw_cof_speed)


with st.spinner("🔬 Training AI on 64 000+ lunar tribology data points…"):
    (model_cof, model_wear, expected_columns,
     all_materials, all_coatings,
     mat_props, coat_props, global_mean,
     temp_mat_props, temp_mat_std,
     raw_cof_speed) = build_ai_brain()

st.success("✅ AI models ready!")

# ─────────────────────────────────────────────
# HELPER: get interpolated props for material+temp
# ─────────────────────────────────────────────
def get_props_at_temp(material, temperature, mat_props, temp_mat_props, global_mean):
    phys_cols = ['Density (g/cm³)',
                 'Thermal Conductivity (W/m·K)',
                 'Thermal Expansion (×10⁻⁶/°C)',
                 'Hardness (HV)']

    if material in temp_mat_props.index.get_level_values(0):
        mat_temp_df = temp_mat_props.loc[material]
        known_temps = sorted(mat_temp_df.index.tolist())

        if temperature in known_temps:
            return mat_temp_df.loc[temperature].to_dict()

        lower = [t for t in known_temps if t <= temperature]
        upper = [t for t in known_temps if t >= temperature]

        if lower and upper:
            t_lo, t_hi = max(lower), min(upper)
            if t_lo == t_hi:
                return mat_temp_df.loc[t_lo].to_dict()
            frac = (temperature - t_lo) / (t_hi - t_lo)
            props_lo = mat_temp_df.loc[t_lo]
            props_hi = mat_temp_df.loc[t_hi]
            return {col: float(props_lo[col] + frac * (props_hi[col] - props_lo[col]))
                    for col in phys_cols}
        elif lower:
            return mat_temp_df.loc[max(lower)].to_dict()
        elif upper:
            return mat_temp_df.loc[min(upper)].to_dict()

    return mat_props.get(material, global_mean)


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
# 4. GET TEMPERATURE-DEPENDENT PROPERTIES
# ─────────────────────────────────────────────
props = get_props_at_temp(mat, temp, mat_props, temp_mat_props, global_mean)

# ─────────────────────────────────────────────
# 5. BUILD INPUT VECTOR
# ─────────────────────────────────────────────
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
# 6. PREDICT
# ─────────────────────────────────────────────
pred_cof       = float(model_cof.predict(input_df)[0])
pred_wear_log  = float(model_wear.predict(input_df)[0])
pred_wear_real = 10 ** pred_wear_log
pred_cof       = max(pred_cof, 0.0)

# ─────────────────────────────────────────────
# 7. DISPLAY RESULTS
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🤖 AI Predictions")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric(label="⚡ COF",                  value=f"{pred_cof:.4f}",
          help="Coefficient of Friction")
m2.metric(label="🔬 Wear Rate (mm³/N·m)", value=f"{pred_wear_real:.3e}",
          help="Volumetric material loss per unit load and sliding distance")
m3.metric(label="⚖️ Density",             value=f"{props.get('Density (g/cm³)', 0):.3f} g/cm³",
          help="Updates with temperature")
m4.metric(label="🌡️ Th. Conductivity",    value=f"{props.get('Thermal Conductivity (W/m·K)', 0):.2f} W/m·K",
          help="Updates with temperature")
m5.metric(label="📐 Th. Expansion",        value=f"{props.get('Thermal Expansion (×10⁻⁶/°C)', 0):.2f} ×10⁻⁶/°C",
          help="Updates with temperature")
m6.metric(label="💎 Hardness",             value=f"{props.get('Hardness (HV)', 0):.1f} HV",
          help="Updates with temperature")

st.caption("🔄 Density, Thermal Conductivity, Thermal Expansion, and Hardness auto-update as you move the Temperature slider.")

# ─────────────────────────────────────────────
# 8. MATERIAL PHYSICAL PROPERTIES TABLE
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("🧪 Material Physical Properties (at current temperature)")

phys_display = {
    "Density (g/cm³)":               round(props.get('Density (g/cm³)', 0), 3),
    "Thermal Conductivity (W/m·K)":  round(props.get('Thermal Conductivity (W/m·K)', 0), 3),
    "Thermal Expansion (×10⁻⁶/°C)": round(props.get('Thermal Expansion (×10⁻⁶/°C)', 0), 3),
    "Hardness (HV)":                 round(props.get('Hardness (HV)', 0), 1),
}
st.table(pd.DataFrame(phys_display.items(), columns=["Property", "Value"]))



KNOWN_SPEEDS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
PALETTE      = ['#1f4e79','#c0392b','#1a7a4a','#7d3c98',
                '#b7770d','#117a8b','#5d4037','#2e86ab','#d35400']

# ── Build AI sweep across speeds ─────────────────────────────────
speed_rows = []
for s in KNOWN_SPEEDS:
    row = dict.fromkeys(expected_columns, 0.0)
    row['Lunar Contact']       = 1 if "Yes" in env else 0
    row['Temperature (°C)']    = float(temp)
    row['Normal Load (N)']     = float(load)
    row['Sliding Speed (m/s)'] = float(s)
    for pc, val in props.items():
        if pc in row:
            row[pc] = val
    mc = f'Material_{mat}'
    cc = f'Coating Material_{coat}'
    if mc in row: row[mc] = 1.0
    if cc in row: row[cc] = 1.0
    speed_rows.append(row)

speed_sweep_df  = pd.DataFrame(speed_rows, columns=expected_columns)
ai_cof_values   = [max(float(v), 0.0)
                   for v in model_cof.predict(speed_sweep_df)]

# ── Raw dataset mean ± std for selected material ──────────────────
raw_mat = raw_cof_speed[raw_cof_speed['Material'] == mat].copy()
raw_mat = raw_mat.set_index('Sliding Speed (m/s)')
raw_means = [float(raw_mat.loc[s, 'mean']) if s in raw_mat.index else np.nan
             for s in KNOWN_SPEEDS]
raw_stds  = [float(raw_mat.loc[s, 'std'])  if s in raw_mat.index else 0.0
             for s in KNOWN_SPEEDS]
raw_upper = [m + s for m, s in zip(raw_means, raw_stds)]
raw_lower = [max(m - s, 0) for m, s in zip(raw_means, raw_stds)]

# ── Y-axis range ──────────────────────────────────────────────────
all_y = [v for v in ai_cof_values + raw_upper if not np.isnan(v)]
y_min = max(min(all_y) * 0.85, 0)
y_max = max(all_y) * 1.15

tab_ai, tab_all = st.tabs([
    "🤖 AI Prediction — Your Configuration",
    "📊 All Materials — Dataset Comparison",
])




# ── TAB 2: All materials comparison from raw dataset ─────────────
with tab_all:
    fig_all = go.Figure()

    for i, material in enumerate(all_materials):
        color = PALETTE[i % len(PALETTE)]
        sub   = raw_cof_speed[raw_cof_speed['Material'] == material].copy()
        sub   = sub.sort_values('Sliding Speed (m/s)')

        means  = sub['mean'].tolist()
        stds   = sub['std'].tolist()
        speeds = sub['Sliding Speed (m/s)'].tolist()
        upper  = [m + s for m, s in zip(means, stds)]
        lower  = [max(m - s, 0) for m, s in zip(means, stds)]

        # ±1σ band
        fig_all.add_trace(go.Scatter(
            x=speeds + speeds[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor=color,
            opacity=0.10,
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip',
            legendgroup=material,
        ))

        is_selected = (material == mat)
        fig_all.add_trace(go.Scatter(
            x=speeds,
            y=means,
            mode='lines+markers',
            name=material + (' ◀ selected' if is_selected else ''),
            legendgroup=material,
            line=dict(color=color,
                      width=4 if is_selected else 1.8,
                      dash='solid' if is_selected else 'dot'),
            marker=dict(size=9 if is_selected else 5),
            hovertemplate=(
                f"<b>{material}</b><br>"
                "Speed: %{x} m/s<br>"
                "Mean COF: %{y:.4f}"
                "<extra></extra>"
            )
        ))

    # Current speed marker
    fig_all.add_vline(
        x=speed,
        line=dict(color='red', dash='dash', width=1.5),
        annotation_text=f"{speed} m/s",
        annotation_position="top right",
        annotation_font_color='red',
    )

    fig_all.update_layout(
        title=dict(
            text='COF vs Sliding Speed — All 9 Materials (Dataset Means ± 1σ)<br>'
                 '<sup>Averaged over all temps, loads, coatings & environments | '
                 'Bold = selected material | Dotted red = current speed</sup>',
            font=dict(size=14)
        ),
        xaxis=dict(
            title='Sliding Speed (m/s)',
            type='log',
            tickvals=KNOWN_SPEEDS,
            ticktext=[str(s) for s in KNOWN_SPEEDS],
        ),
        yaxis=dict(title='Mean COF'),
        hovermode='x unified',
        height=520,
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_all, use_container_width=True)


# ─────────────────────────────────────────────
# 10. HARDNESS vs TEMPERATURE GRAPHS
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📈 Hardness vs Temperature")

KNOWN_TEMPS = [-173, -153, -123, -93, -63, -23, 7, 37, 77, 127]

tab_b, tab_a = st.tabs([
    "🎯 Your Exact Configuration",
    "📊 All Materials Comparison",
])

with tab_b:
    st.caption(
        f"Hardness from dataset across all temperatures, "
        f"holding **{mat}** | **{coat}** | **Load={load}N** | "
        f"**Speed={speed}m/s** | **{env}** fixed."
    )

    rows = []
    for t in KNOWN_TEMPS:
        p = get_props_at_temp(mat, t, mat_props, temp_mat_props, global_mean)
        row = dict.fromkeys(expected_columns, 0.0)
        row['Lunar Contact']       = 1 if "Yes" in env else 0
        row['Temperature (°C)']    = float(t)
        row['Normal Load (N)']     = float(load)
        row['Sliding Speed (m/s)'] = float(speed)
        for pc, val in p.items():
            if pc in row:
                row[pc] = val
        mc = f'Material_{mat}'
        cc = f'Coating Material_{coat}'
        if mc in row: row[mc] = 1.0
        if cc in row: row[cc] = 1.0
        rows.append(row)

    sweep_df       = pd.DataFrame(rows, columns=expected_columns)
    hardness_sweep = sweep_df['Hardness (HV)'].tolist()

    h_min = min(hardness_sweep)
    h_max = max(hardness_sweep)
    h_pad = (h_max - h_min) * 0.3 if h_max != h_min else h_max * 0.01
    y_lo_b = h_min - h_pad
    y_hi_b = h_max + h_pad

    fig_b = go.Figure()

    if mat in temp_mat_std.index.get_level_values(0):
        std_vals = [
            float(temp_mat_std.loc[(mat, t), 'Hardness (HV)'])
            if (mat, t) in temp_mat_std.index else 0.0
            for t in KNOWN_TEMPS
        ]
        upper = [h + s for h, s in zip(hardness_sweep, std_vals)]
        lower = [max(h - s, 0) for h, s in zip(hardness_sweep, std_vals)]
        fig_b.add_trace(go.Scatter(
            x=KNOWN_TEMPS + KNOWN_TEMPS[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(31,78,121,0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='±1σ band',
            hoverinfo='skip',
        ))

    fig_b.add_trace(go.Scatter(
        x=KNOWN_TEMPS,
        y=hardness_sweep,
        mode='lines+markers',
        name=f'{mat} — {coat}',
        line=dict(color='#1f4e79', width=3),
        marker=dict(size=8, symbol='circle'),
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Temperature: %{x}°C<br>"
            "Hardness: %{y:.1f} HV"
            "<extra></extra>"
        )
    ))

    fig_b.add_vline(
        x=temp,
        line=dict(color='red', dash='dash', width=2),
        annotation_text=f"Current: {temp}°C",
        annotation_position="top right",
        annotation_font_color='red',
    )

    current_hardness = props.get('Hardness (HV)', 0)
    fig_b.add_trace(go.Scatter(
        x=[temp],
        y=[current_hardness],
        mode='markers',
        name=f'Current ({temp}°C)',
        marker=dict(color='red', size=14, symbol='star',
                    line=dict(color='darkred', width=2)),
        hovertemplate=(
            f"<b>Current Selection</b><br>"
            f"Temperature: {temp}°C<br>"
            f"Hardness: {current_hardness:.1f} HV"
            "<extra></extra>"
        )
    ))

    fig_b.add_vrect(x0=-173, x1=0, fillcolor='steelblue', opacity=0.04,
                    layer='below', line_width=0,
                    annotation_text='Cryogenic', annotation_position='top left',
                    annotation_font_size=10, annotation_font_color='steelblue')
    fig_b.add_vrect(x0=0, x1=127, fillcolor='tomato', opacity=0.04,
                    layer='below', line_width=0,
                    annotation_text='High-T', annotation_position='top right',
                    annotation_font_size=10, annotation_font_color='tomato')

    fig_b.update_layout(
        title=dict(
            text=f'Hardness vs Temperature — {mat} with {coat}<br>'
                 f'<sup>Load={load}N | Speed={speed}m/s | {env} | '
                 f'Red ★ = current slider position | Shaded = ±1σ</sup>',
            font=dict(size=14)
        ),
        xaxis_title='Temperature (°C)',
        yaxis=dict(title='Hardness (HV)', range=[y_lo_b, y_hi_b]),
        hovermode='x unified',
        height=500,
        legend=dict(font=dict(size=11)),
    )
    st.plotly_chart(fig_b, use_container_width=True)


with tab_a:
    st.caption(
        "Mean Hardness (HV) ± 1σ for all 9 substrate materials across "
        "the full lunar temperature range, averaged over all loads, "
        "speeds, coatings, and environments in the dataset."
    )

    all_means_list = []
    for material in all_materials:
        if material in temp_mat_props.index.get_level_values(0):
            for t in KNOWN_TEMPS:
                if (material, t) in temp_mat_props.index:
                    all_means_list.append(
                        float(temp_mat_props.loc[(material, t), 'Hardness (HV)']))

    if all_means_list:
        a_min = min(all_means_list)
        a_max = max(all_means_list)
        a_pad = (a_max - a_min) * 0.10
        y_lo_a = a_min - a_pad
        y_hi_a = a_max + a_pad
    else:
        y_lo_a, y_hi_a = 0, 5000

    fig_a = go.Figure()

    for i, material in enumerate(all_materials):
        color = PALETTE[i % len(PALETTE)]
        if material not in temp_mat_props.index.get_level_values(0):
            continue

        mean_vals = [
            float(temp_mat_props.loc[(material, t), 'Hardness (HV)'])
            if (material, t) in temp_mat_props.index else np.nan
            for t in KNOWN_TEMPS
        ]
        std_vals = [
            float(temp_mat_std.loc[(material, t), 'Hardness (HV)'])
            if (material, t) in temp_mat_std.index else 0.0
            for t in KNOWN_TEMPS
        ]
        upper = [m + s for m, s in zip(mean_vals, std_vals)]
        lower = [max(m - s, 0) for m, s in zip(mean_vals, std_vals)]

        fig_a.add_trace(go.Scatter(
            x=KNOWN_TEMPS + KNOWN_TEMPS[::-1],
            y=upper + lower[::-1],
            fill='toself', fillcolor=color, opacity=0.10,
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False, hoverinfo='skip', legendgroup=material,
        ))

        is_selected = (material == mat)
        fig_a.add_trace(go.Scatter(
            x=KNOWN_TEMPS, y=mean_vals,
            mode='lines+markers',
            name=material + (' ◀ selected' if is_selected else ''),
            legendgroup=material,
            line=dict(color=color,
                      width=4 if is_selected else 1.8,
                      dash='solid' if is_selected else 'dot'),
            marker=dict(size=9 if is_selected else 5),
            hovertemplate=(
                f"<b>{material}</b><br>"
                "Temperature: %{x}°C<br>"
                "Mean Hardness: %{y:.1f} HV"
                "<extra></extra>"
            )
        ))

    fig_a.add_vline(
        x=temp,
        line=dict(color='red', dash='dash', width=1.5),
        annotation_text=f"{temp}°C",
        annotation_position="top right",
        annotation_font_color='red',
    )
    fig_a.add_vrect(x0=-173, x1=0, fillcolor='steelblue', opacity=0.04,
                    layer='below', line_width=0,
                    annotation_text='Cryogenic', annotation_position='top left',
                    annotation_font_size=10, annotation_font_color='steelblue')
    fig_a.add_vrect(x0=0, x1=127, fillcolor='tomato', opacity=0.04,
                    layer='below', line_width=0,
                    annotation_text='High-T', annotation_position='top right',
                    annotation_font_size=10, annotation_font_color='tomato')

    fig_a.update_layout(
        title=dict(
            text='Hardness vs Temperature — All 9 Substrate Materials<br>'
                 '<sup>Mean ± 1σ from 64 783 data points | '
                 'Bold line = currently selected material | Dotted red = current temperature</sup>',
            font=dict(size=14)
        ),
        xaxis_title='Temperature (°C)',
        yaxis=dict(title='Hardness (HV)', range=[y_lo_a, y_hi_a]),
        hovermode='x unified',
        height=520,
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_a, use_container_width=True)
# ─────────────────────────────────────────────
# 12. ADVANCED TRIBOLOGY DASHBOARD (TABS)
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🔬 Advanced Tribological Analysis")

# Load the raw dataset for the advanced visualizations
@st.cache_data
def load_raw_data():
    raw_df = pd.read_csv('Lunar_Tribology_ML_Dataset__1_.csv')
    raw_df.columns = raw_df.columns.str.strip()
    return raw_df

df = load_raw_data()

# Create the tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Friction vs. Speed", 
    "🌡️ Thermal Profile", 
    "🛡️ Coating Effectiveness", 
    "⚖️ Friction/Wear Trade-off"
])

# -- TAB 1: SPEED PROFILE --
with tab1:
    st.subheader(f"📊 Kinematic Friction Profile: {mat}")
    
    KNOWN_SPEEDS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    mat_data = raw_cof_speed[raw_cof_speed['Material'] == mat].sort_values('Sliding Speed (m/s)')

    if not mat_data.empty:
        fig_sci = go.Figure()
        speeds = mat_data['Sliding Speed (m/s)'].tolist()
        means  = mat_data['mean'].tolist()
        stds   = mat_data['std'].tolist()
        
        upper = [m + s for m, s in zip(means, stds)]
        lower = [max(m - s, 0) for m, s in zip(means, stds)]

        # 1. Uncertainty Band
        fig_sci.add_trace(go.Scatter(
            x=speeds + speeds[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(0, 0, 0, 0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Experimental Variance (±1σ)',
            showlegend=True,
            hoverinfo='skip'
        ))

        # 2. Mean Profile (FIXED MARKER SYNTAX)
        fig_sci.add_trace(go.Scatter(
            x=speeds,
            y=means,
            mode='lines+markers',
            name=f'Mean Response ({mat})',
            line=dict(color='#000000', width=2.5),
            marker=dict(
                size=10, 
                symbol='circle', 
                color='white', 
                line=dict(width=2, color='#000000')
            ),
            hovertemplate="v: %{x} m/s<br>μ: %{y:.4f}<extra></extra>"
        ))

        # 3. Dynamic Operating Point
        fig_sci.add_trace(go.Scatter(
            x=[speed],
            y=[pred_cof],
            mode='markers',
            name='Current AI Prediction',
            marker=dict(color='#d62728', size=12, symbol='x', line=dict(width=2)),
            showlegend=True
        ))

        # 4. Layout
        fig_sci.update_layout(
            xaxis=dict(
                title="Sliding Velocity, <i>v</i> (m/s)", type="log",
                tickvals=KNOWN_SPEEDS, ticktext=[str(s) for s in KNOWN_SPEEDS],
                gridcolor='lightgrey', minor=dict(showgrid=True, gridcolor='#f0f0f0'),
                linecolor='black', mirror=True
            ),
            yaxis=dict(
                title="Friction Coefficient, <i>μ</i>",
                gridcolor='lightgrey', linecolor='black', mirror=True,
                zeroline=True, zerolinecolor='grey', range=[0, max(upper) * 1.2]
            ),
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="Black", borderwidth=1),
            font=dict(family="serif", size=14),
            height=500, plot_bgcolor="white", template="none"
        )
        st.plotly_chart(fig_sci, use_container_width=True)
    else:
        st.warning(f"No baseline data available for {mat} in the dataset.")

# -- TAB 2: THERMAL PROFILE --
with tab2:
    st.subheader(f"🌡️ Thermal Friction Response: {mat}")
    
    temp_data = df[df['Material'] == mat].groupby('Temperature (°C)')['COF'].agg(['mean', 'std']).reset_index()

    if not temp_data.empty:
        fig_temp = go.Figure()
        temps = temp_data['Temperature (°C)'].tolist()
        means = temp_data['mean'].tolist()
        stds  = temp_data['std'].tolist()
        
        upper = [m + s for m, s in zip(means, stds)]
        lower = [max(m - s, 0) for m, s in zip(means, stds)]

        fig_temp.add_trace(go.Scatter(
            x=temps + temps[::-1], y=upper + lower[::-1],
            fill='toself', fillcolor='rgba(214, 39, 40, 0.15)',
            line=dict(color='rgba(255,255,255,0)'), name='Variance (±1σ)', showlegend=True, hoverinfo='skip'
        ))

        fig_temp.add_trace(go.Scatter(
            x=temps, y=means, mode='lines+markers',
            name=f'Mean COF ({mat})', line=dict(color='#d62728', width=3),
            marker=dict(size=8, symbol='square'),
            hovertemplate="Temp: %{x} °C<br>Mean COF: %{y:.4f}<extra></extra>"
        ))

        fig_temp.add_vline(x=temp, line=dict(color='black', dash='dot', width=2), annotation_text=f"Selected: {temp} °C", annotation_position="top right")

        fig_temp.update_layout(
            xaxis=dict(title="Temperature (°C)", gridcolor='lightgrey'),
            yaxis=dict(title="Coefficient of Friction (COF)", gridcolor='lightgrey', zeroline=True),
            height=500, template="plotly_white", hovermode="x unified"
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    else:
        st.warning(f"No thermal data available for {mat}.")

# -- TAB 3: COATING COMPARISON --
with tab3:
    st.subheader(f"🛡️ Coating Effectiveness on {mat}")
    st.caption("Distribution of COF across all available coatings for the selected substrate.")
    
    mat_coat_data = df[df['Material'] == mat]

    if not mat_coat_data.empty:
        fig_coat = go.Figure()
        
        for c in sorted(mat_coat_data['Coating Material'].unique()):
            c_data = mat_coat_data[mat_coat_data['Coating Material'] == c]['COF']
            is_selected = (c == coat)
            fill_color = 'rgba(31, 119, 180, 0.7)' if is_selected else 'rgba(200, 200, 200, 0.5)'
            line_color = '#1f77b4' if is_selected else 'gray'
            
            fig_coat.add_trace(go.Box(
                y=c_data, name=c + (" (Selected)" if is_selected else ""),
                boxpoints='outliers', marker_color=line_color, fillcolor=fill_color,
                line=dict(width=2 if is_selected else 1)
            ))

        fig_coat.update_layout(
            xaxis=dict(title="Coating Material", tickangle=-45),
            yaxis=dict(title="Coefficient of Friction (COF)", gridcolor='lightgrey'),
            height=500, template="plotly_white", showlegend=False
        )
        st.plotly_chart(fig_coat, use_container_width=True)

# -- TAB 4: TRADE-OFF MAP --
with tab4:
    st.subheader(f"⚖️ Friction vs. Wear Performance: {mat}")
    
    tradeoff_data = df[df['Material'] == mat]
    wear_col = next(c for c in df.columns if 'Wear Rate' in c)

    if not tradeoff_data.empty:
        fig_trade = go.Figure()
        
        for env_status in ['Yes', 'No']:
            env_label = "Lunar Vacuum" if env_status == 'Yes' else "Earth Ambient"
            env_data = tradeoff_data[tradeoff_data['Lunar Contact'].astype(str).str.strip() == env_status]
            
            fig_trade.add_trace(go.Scatter(
                x=env_data['COF'], y=env_data[wear_col], mode='markers', name=env_label,
                marker=dict(size=6, opacity=0.6, line=dict(width=0.5, color='black')),
                hovertemplate="COF: %{x:.4f}<br>Wear: %{y:.2e}<extra></extra>"
            ))
            
        fig_trade.add_trace(go.Scatter(
            x=[pred_cof], y=[pred_wear_real], mode='markers', name='★ AI Prediction (Current Config)',
            marker=dict(color='gold', size=16, symbol='star', line=dict(width=2, color='black')),
            hovertemplate="Predicted COF: %{x:.4f}<br>Predicted Wear: %{y:.2e}<extra></extra>"
        ))

        fig_trade.update_layout(
            xaxis=dict(title="Coefficient of Friction (COF)", gridcolor='lightgrey'),
            yaxis=dict(title="Wear Rate (mm³/N·m)", type="log", gridcolor='lightgrey'),
            height=500, template="plotly_white", legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)")
        )
        st.plotly_chart(fig_trade, use_container_width=True)
# ─────────────────────────────────────────────
# 11. CONFIGURATION SUMMARY
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Full Configuration Summary")
summary = {
    "Environment":                   env,
    "Temperature (°C)":              temp,
    "Normal Load (N)":               load,
    "Sliding Speed (m/s)":           speed,
    "Substrate Material":            mat,
    "Coating":                       coat,
    "Density (g/cm³)":               f"{props.get('Density (g/cm³)', 0):.3f}",
    "Thermal Conductivity (W/m·K)":  f"{props.get('Thermal Conductivity (W/m·K)', 0):.2f}",
    "Thermal Expansion (×10⁻⁶/°C)": f"{props.get('Thermal Expansion (×10⁻⁶/°C)', 0):.2f}",
    "Hardness (HV)":                 f"{props.get('Hardness (HV)', 0):.1f}",
    "Predicted COF":                 f"{pred_cof:.4f}",
    "Predicted Wear Rate":           f"{pred_wear_real:.3e} mm³/N·m",
}
st.table(pd.DataFrame(summary.items(), columns=["Parameter", "Value"]))