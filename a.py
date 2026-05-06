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

    # ── Fallback dicts (material-only and coating-only averages) ──
    mat_props   = df.groupby('Material')[phys_cols].mean().to_dict('index')
    coat_props  = df.groupby('Coating Material')[phys_cols].mean().to_dict('index')
    global_mean = df[phys_cols].mean().to_dict()

    # ── FIX: group by Material + Coating + Temperature (3-level) ──
    temp_mat_props = (
        df.groupby(['Material', 'Coating Material', 'Temperature (°C)'])[phys_cols].mean()
    )
    temp_mat_std = (
        df.groupby(['Material', 'Coating Material', 'Temperature (°C)'])[phys_cols].std()
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
# HELPER: get interpolated props for material + coating + temp
# FIX: now accepts coating and uses 3-level index
# ─────────────────────────────────────────────
def get_props_at_temp(material, coating, temperature, mat_props, coat_props,
                      temp_mat_props, global_mean):
    phys_cols = ['Density (g/cm³)',
                 'Thermal Conductivity (W/m·K)',
                 'Thermal Expansion (×10⁻⁶/°C)',
                 'Hardness (HV)']

    # Build the 2-level key (Material, Coating Material)
    key = (material, coating)
    mat_coat_index = temp_mat_props.index.droplevel(-1)  # drop Temperature level

    if key in mat_coat_index:
        mat_temp_df = temp_mat_props.loc[key]          # now indexed by Temperature only
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

    # Fallback 1: material-only average
    if material in mat_props:
        return mat_props[material]

    # Fallback 2: coating-only average
    if coating in coat_props:
        return coat_props[coating]

    # Fallback 3: dataset global mean
    return global_mean


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
# 4. GET TEMPERATURE + COATING DEPENDENT PROPERTIES  (FIX: pass coat)
# ─────────────────────────────────────────────
props = get_props_at_temp(mat, coat, temp, mat_props, coat_props,
                          temp_mat_props, global_mean)

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
          help="Updates with temperature and coating")
m4.metric(label="🌡️ Th. Conductivity",    value=f"{props.get('Thermal Conductivity (W/m·K)', 0):.2f} W/m·K",
          help="Updates with temperature and coating")
m5.metric(label="📐 Th. Expansion",        value=f"{props.get('Thermal Expansion (×10⁻⁶/°C)', 0):.2f} ×10⁻⁶/°C",
          help="Updates with temperature and coating")
m6.metric(label="💎 Hardness",             value=f"{props.get('Hardness (HV)', 0):.1f} HV",
          help="Updates with temperature and coating")

st.caption("🔄 Density, Thermal Conductivity, Thermal Expansion, and Hardness auto-update as you move the Temperature slider **and** change the Coating selection.")

# ─────────────────────────────────────────────
# 8. MATERIAL PHYSICAL PROPERTIES TABLE
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("🧪 Material Physical Properties (at current temperature & coating)")

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

# ── Build AI sweep across speeds  (FIX: pass coat to get_props_at_temp) ──
speed_rows = []
for s in KNOWN_SPEEDS:
    row = dict.fromkeys(expected_columns, 0.0)
    row['Lunar Contact']       = 1 if "Yes" in env else 0
    row['Temperature (°C)']    = float(temp)
    row['Normal Load (N)']     = float(load)
    row['Sliding Speed (m/s)'] = float(s)
    sweep_props = get_props_at_temp(mat, coat, temp, mat_props, coat_props,
                                    temp_mat_props, global_mean)
    for pc, val in sweep_props.items():
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

# ── TAB 1: AI prediction for current config ───────────────────────
with tab_ai:
    fig_ai = go.Figure()

    # Raw dataset band for selected material
    fig_ai.add_trace(go.Scatter(
        x=KNOWN_SPEEDS + KNOWN_SPEEDS[::-1],
        y=raw_upper + raw_lower[::-1],
        fill='toself',
        fillcolor='rgba(100,100,100,0.10)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Dataset ±1σ (all coatings)',
        hoverinfo='skip',
    ))
    fig_ai.add_trace(go.Scatter(
        x=KNOWN_SPEEDS, y=raw_means,
        mode='lines+markers',
        name=f'Dataset mean ({mat})',
        line=dict(color='#888888', width=2, dash='dot'),
        marker=dict(size=6),
    ))

    # AI prediction line
    fig_ai.add_trace(go.Scatter(
        x=KNOWN_SPEEDS, y=ai_cof_values,
        mode='lines+markers',
        name=f'AI — {mat} + {coat}',
        line=dict(color='#1f4e79', width=3),
        marker=dict(size=8, symbol='circle'),
        hovertemplate="Speed: %{x} m/s<br>AI COF: %{y:.4f}<extra></extra>",
    ))

    # Current operating point
    fig_ai.add_trace(go.Scatter(
        x=[speed], y=[pred_cof],
        mode='markers',
        name='Current Operating Point',
        marker=dict(color='red', size=14, symbol='star',
                    line=dict(color='darkred', width=2)),
        hovertemplate=f"Speed: {speed} m/s<br>Predicted COF: {pred_cof:.4f}<extra></extra>",
    ))

    fig_ai.add_vline(x=speed, line=dict(color='red', dash='dash', width=1.5),
                     annotation_text=f"{speed} m/s",
                     annotation_position="top right",
                     annotation_font_color='red')

    fig_ai.update_layout(
        title=dict(
            text=f'COF vs Sliding Speed — AI Prediction for {mat} + {coat}<br>'
                 f'<sup>Temp={temp}°C | Load={load}N | {env} | '
                 f'Grey = dataset baseline (all coatings averaged)</sup>',
            font=dict(size=14)
        ),
        xaxis=dict(title='Sliding Speed (m/s)', type='log',
                   tickvals=KNOWN_SPEEDS, ticktext=[str(s) for s in KNOWN_SPEEDS]),
        yaxis=dict(title='COF', range=[y_min, y_max]),
        hovermode='x unified', height=500,
        legend=dict(font=dict(size=11)),
    )
    st.plotly_chart(fig_ai, use_container_width=True)

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
        # FIX: pass coat so hardness reflects this coating at each temperature
        p = get_props_at_temp(mat, coat, t, mat_props, coat_props,
                              temp_mat_props, global_mean)
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
    h_pad = (h_max - h_min) * 0.3 if h_max != h_min else max(h_max * 0.1, 10)
    y_lo_b = max(h_min - h_pad, 0)
    y_hi_b = h_max + h_pad

    fig_b = go.Figure()

    # ±1σ band — FIX: use 3-level std index
    std_vals = []
    for t in KNOWN_TEMPS:
        key3 = (mat, coat, t)
        if key3 in temp_mat_std.index:
            v = temp_mat_std.loc[key3, 'Hardness (HV)']
            std_vals.append(float(v) if not np.isnan(v) else 0.0)
        else:
            std_vals.append(0.0)

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
        f"speeds, and environments — shown for **{coat}** coating."
    )

    # FIX: filter all-materials comparison to the currently selected coating
    all_means_list = []
    for material in all_materials:
        key_m = (material, coat)
        mat_coat_idx = temp_mat_props.index.droplevel(-1)
        if key_m in mat_coat_idx:
            for t in KNOWN_TEMPS:
                key3 = (material, coat, t)
                if key3 in temp_mat_props.index:
                    all_means_list.append(
                        float(temp_mat_props.loc[key3, 'Hardness (HV)']))

    if all_means_list:
        a_min = min(all_means_list)
        a_max = max(all_means_list)
        a_pad = (a_max - a_min) * 0.10 if a_max != a_min else max(a_max * 0.05, 10)
        y_lo_a = max(a_min - a_pad, 0)
        y_hi_a = a_max + a_pad
    else:
        y_lo_a, y_hi_a = 0, 5000

    fig_a = go.Figure()

    for i, material in enumerate(all_materials):
        color = PALETTE[i % len(PALETTE)]
        key_m = (material, coat)
        mat_coat_idx = temp_mat_props.index.droplevel(-1)

        if key_m not in mat_coat_idx:
            continue

        mean_vals = []
        std_vals  = []
        for t in KNOWN_TEMPS:
            key3 = (material, coat, t)
            if key3 in temp_mat_props.index:
                mean_vals.append(float(temp_mat_props.loc[key3, 'Hardness (HV)']))
            else:
                mean_vals.append(np.nan)
            if key3 in temp_mat_std.index:
                v = temp_mat_std.loc[key3, 'Hardness (HV)']
                std_vals.append(float(v) if not np.isnan(v) else 0.0)
            else:
                std_vals.append(0.0)

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
            text=f'Hardness vs Temperature — All 9 Materials with {coat}<br>'
                 '<sup>Mean ± 1σ | Bold line = currently selected material | '
                 'Dotted red = current temperature</sup>',
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

@st.cache_data
def load_raw_data():
    raw_df = pd.read_csv('Lunar_Tribology_ML_Dataset__1_.csv')
    raw_df.columns = raw_df.columns.str.strip()
    return raw_df

df = load_raw_data()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Friction vs. Speed",
    "🌡️ Thermal Profile",
    "🛡️ Coating Effectiveness",
    "⚖️ Archard's Law",
    "🏆 Material Ranking",
    "📐 Thermal Expansion"
])

# -- TAB 1: SPEED PROFILE --
with tab1:
    st.subheader(f"📊 Kinematic Friction Profile: {mat}")

    mat_data = raw_cof_speed[raw_cof_speed['Material'] == mat].sort_values('Sliding Speed (m/s)')

    if not mat_data.empty:
        fig_sci = go.Figure()
        speeds = mat_data['Sliding Speed (m/s)'].tolist()
        means  = mat_data['mean'].tolist()
        stds   = mat_data['std'].tolist()

        upper = [m + s for m, s in zip(means, stds)]
        lower = [max(m - s, 0) for m, s in zip(means, stds)]

        # 1. Changed fillcolor to a semi-transparent white so it shows up on black
        fig_sci.add_trace(go.Scatter(
            x=speeds + speeds[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(255, 255, 255, 0.1)', 
            line=dict(color='rgba(255,255,255,0)'),
            name='Experimental Variance (±1σ)',
            showlegend=True,
            hoverinfo='skip'
        ))

        # 2. Changed the main line and marker borders from black (#000000) to white (#FFFFFF)
        fig_sci.add_trace(go.Scatter(
            x=speeds, y=means,
            mode='lines+markers',
            name=f'Mean Response ({mat})',
            line=dict(color='#FFFFFF', width=2.5),
            marker=dict(size=10, symbol='circle', color='black',
                        line=dict(width=2, color='#FFFFFF')),
            hovertemplate="v: %{x} m/s<br>μ: %{y:.4f}<extra></extra>"
        ))

        # 3. Tweaked the AI prediction marker to a brighter red for better contrast
        fig_sci.add_trace(go.Scatter(
            x=[speed], y=[pred_cof],
            mode='markers',
            name='Current AI Prediction',
            marker=dict(color='#ff4b4b', size=12, symbol='x', line=dict(width=2.5, color='#ff4b4b')),
            showlegend=True
        ))

        # 4. Updated layout, grids, legend, and text to dark mode equivalents
        fig_sci.update_layout(
            xaxis=dict(
                title="Sliding Velocity, <i>v</i> (m/s)", type="log",
                tickvals=KNOWN_SPEEDS, ticktext=[str(s) for s in KNOWN_SPEEDS],
                gridcolor='#333333', minor=dict(showgrid=True, gridcolor='#222222'),
                linecolor='white', mirror=True, color='white'
            ),
            yaxis=dict(
                title="Friction Coefficient, <i>μ</i>",
                gridcolor='#333333', linecolor='white', mirror=True, color='white',
                zeroline=True, zerolinecolor='#666666', range=[0, max(upper) * 1.2]
            ),
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98,
                        bgcolor="rgba(0,0,0,0.8)", bordercolor="white", borderwidth=1, font=dict(color="white")),
            font=dict(family="serif", size=14, color="white"),
            height=500, 
            plot_bgcolor="black", 
            paper_bgcolor="black", 
            template="plotly_dark"
        )
        st.plotly_chart(fig_sci, use_container_width=True)
    else:
        st.warning(f"No baseline data available for {mat} in the dataset.")
# -- TAB 2: THERMAL PROFILE --
with tab2:
    st.subheader(f"🌡️ Thermal Friction Response: {mat} with {coat}")

    # FIX: filter by both material AND coating for accurate thermal profile
    temp_data = (
        df[(df['Material'] == mat) & (df['Coating Material'] == coat)]
        .groupby('Temperature (°C)')['COF']
        .agg(['mean', 'std'])
        .reset_index()
    )

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
            line=dict(color='rgba(255,255,255,0)'),
            name='Variance (±1σ)', showlegend=True, hoverinfo='skip'
        ))

        fig_temp.add_trace(go.Scatter(
            x=temps, y=means, mode='lines+markers',
            name=f'Mean COF ({mat} + {coat})',
            line=dict(color='#d62728', width=3),
            marker=dict(size=8, symbol='square'),
            hovertemplate="Temp: %{x} °C<br>Mean COF: %{y:.4f}<extra></extra>"
        ))

        fig_temp.add_vline(x=temp, line=dict(color='black', dash='dot', width=2),
                           annotation_text=f"Selected: {temp} °C",
                           annotation_position="top right")

        fig_temp.update_layout(
            xaxis=dict(title="Temperature (°C)", gridcolor='lightgrey'),
            yaxis=dict(title="Coefficient of Friction (COF)", gridcolor='lightgrey', zeroline=True),
            height=500, template="plotly_white", hovermode="x unified"
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    else:
        st.warning(f"No thermal data available for {mat} + {coat}.")

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

# ── TAB 4: ARCHARD'S LAW VERIFICATION ───────────────────────────
with tab4:
    st.subheader(f"📉 Wear Rate vs Load — Archard's Law Verification")
    st.caption(
        f"AI-predicted wear rate at each of the 5 load values, holding **{mat}** + "
        f"**{coat}** | **Temp={temp}°C** | **Speed={speed} m/s** | **{env}** fixed. "
        f"Archard's Law predicts a slope of **1.0** on a log-log scale (wear ∝ load)."
    )

    KNOWN_LOADS = [1, 5, 10, 20, 50]

    # ── 1. AI predictions across all 5 loads ──────────────────────
    load_rows = []
    for ld in KNOWN_LOADS:
        row = dict.fromkeys(expected_columns, 0.0)
        row['Lunar Contact']       = 1 if "Yes" in env else 0
        row['Temperature (°C)']    = float(temp)
        row['Normal Load (N)']     = float(ld)
        row['Sliding Speed (m/s)'] = float(speed)
        lp = get_props_at_temp(mat, coat, temp, mat_props, coat_props,
                               temp_mat_props, global_mean)
        for pc, val in lp.items():
            if pc in row:
                row[pc] = val
        mc = f'Material_{mat}'
        cc = f'Coating Material_{coat}'
        if mc in row: row[mc] = 1.0
        if cc in row: row[cc] = 1.0
        load_rows.append(row)

    load_sweep_df = pd.DataFrame(load_rows, columns=expected_columns)
    ai_wear_log   = model_wear.predict(load_sweep_df)
    ai_wear_vals  = [10 ** float(v) for v in ai_wear_log]

    # ── 2. Raw dataset mean ± std for this material+coating ────────
    wear_col_name = next(c for c in df.columns if 'Wear Rate' in c)
    raw_load_data = (
        df[(df['Material'] == mat) & (df['Coating Material'] == coat)]
        .groupby('Normal Load (N)')[wear_col_name]
        .agg(['mean', 'std'])
        .reindex(KNOWN_LOADS)
    )
    raw_means = raw_load_data['mean'].tolist()
    raw_stds  = raw_load_data['std'].fillna(0).tolist()
    raw_upper = [max(m + s, 1e-15) for m, s in zip(raw_means, raw_stds)]
    raw_lower = [max(m - s, 1e-15) for m, s in zip(raw_means, raw_stds)]

    # ── 3. Archard theoretical line (slope = 1.0 on log-log) ──────
    ref_idx        = KNOWN_LOADS.index(10)
    ref_wear       = ai_wear_vals[ref_idx]
    archard_theory = [ref_wear * (ld / 10.0) ** 1.0 for ld in KNOWN_LOADS]

    # ── 4. Compute empirical AI slope on log-log ──────────────────
    log_loads    = np.log10(np.array(KNOWN_LOADS, dtype=float))
    log_ai_wears = np.log10(np.array(ai_wear_vals, dtype=float))
    ai_slope     = float(np.polyfit(log_loads, log_ai_wears, 1)[0])

    # ── 5. Build figure ───────────────────────────────────────────
    fig_arch = go.Figure()

    fig_arch.add_trace(go.Scatter(
        x=KNOWN_LOADS + KNOWN_LOADS[::-1],
        y=raw_upper + raw_lower[::-1],
        fill='toself',
        fillcolor='rgba(100,100,100,0.10)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Dataset ±1σ (raw data)',
        hoverinfo='skip',
        showlegend=True,
    ))

    fig_arch.add_trace(go.Scatter(
        x=KNOWN_LOADS,
        y=raw_means,
        mode='lines+markers',
        name='Dataset mean (raw)',
        line=dict(color='#888888', width=2, dash='dot'),
        marker=dict(size=7, symbol='square'),
        hovertemplate="Load: %{x} N<br>Raw mean wear: %{y:.3e} mm³/N·m<extra></extra>",
    ))

    fig_arch.add_trace(go.Scatter(
        x=KNOWN_LOADS,
        y=archard_theory,
        mode='lines',
        name='Archard Theory (slope = 1.0)',
        line=dict(color='#d62728', width=2.5, dash='dash'),
        hovertemplate="Load: %{x} N<br>Archard predicted: %{y:.3e} mm³/N·m<extra></extra>",
    ))

    fig_arch.add_trace(go.Scatter(
        x=KNOWN_LOADS,
        y=ai_wear_vals,
        mode='lines+markers',
        name=f'AI Prediction (slope = {ai_slope:.2f})',
        line=dict(color='#1f4e79', width=3),
        marker=dict(size=10, symbol='circle', color='white',
                    line=dict(width=2.5, color='#1f4e79')),
        hovertemplate="Load: %{x} N<br>AI predicted wear: %{y:.3e} mm³/N·m<extra></extra>",
    ))

    fig_arch.add_trace(go.Scatter(
        x=[load],
        y=[pred_wear_real],
        mode='markers',
        name=f'Current Config ({load} N)',
        marker=dict(color='red', size=14, symbol='star',
                    line=dict(color='darkred', width=2)),
        hovertemplate=f"Load: {load} N<br>AI wear: {pred_wear_real:.3e} mm³/N·m<extra></extra>",
    ))

    fig_arch.add_annotation(
        xref='paper', yref='paper',
        x=0.98, y=0.08,
        text=(
            f"<b>Slope Comparison</b><br>"
            f"Archard Theory: 1.00<br>"
            f"AI Learned:  {ai_slope:.2f}<br>"
            f"Deviation: {abs(ai_slope - 1.0):.2f}"
        ),
        showarrow=False,
        align='left',
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='#1f4e79',
        borderwidth=1.5,
        font=dict(size=12, family='monospace'),
        xanchor='right',
        yanchor='bottom',
    )

    fig_arch.update_layout(
        title=dict(
            text=(
                f"Wear Rate vs Normal Load — Archard's Law Verification<br>"
                f"<sup>{mat} + {coat} | Temp={temp}°C | Speed={speed} m/s | {env} | "
                f"Log-log scale — ideal Archard slope = 1.0</sup>"
            ),
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Normal Load, <i>F</i> (N)",
            type='log',
            tickvals=KNOWN_LOADS,
            ticktext=[str(l) for l in KNOWN_LOADS],
            gridcolor='lightgrey',
            linecolor='black',
            mirror=True,
        ),
        yaxis=dict(
            title="Wear Rate (mm³/N·m)",
            type='log',
            gridcolor='lightgrey',
            linecolor='black',
            mirror=True,
            exponentformat='e',
        ),
        legend=dict(
            yanchor='top', y=0.98,
            xanchor='left', x=0.02,
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=11),
        ),
        hovermode='x unified',
        height=520,
        plot_bgcolor='white',
        template='none',
        font=dict(family='serif', size=13),
    )

    st.plotly_chart(fig_arch, use_container_width=True)

    # ── 6. Interpretation text ────────────────────────────────────
    deviation = abs(ai_slope - 1.0)
    if deviation < 0.05:
        verdict = "🟢 **Excellent** — AI has learned near-perfect Archard scaling."
        detail  = "The model generalises the physical law, not just the data."
    elif deviation < 0.15:
        verdict = "🟡 **Good** — AI slope is close to Archard's theoretical value."
        detail  = "Minor deviation likely due to material/coating-specific effects."
    else:
        verdict = "🔴 **Notable deviation** — AI slope differs from ideal Archard Law."
        detail  = "This material+coating combination may have non-linear load behaviour."

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Archard Theoretical Slope", "1.00",
                 help="Archard's Law: Wear ∝ Load¹ → slope = 1.0 on log-log")
    col_b.metric("AI Learned Slope", f"{ai_slope:.3f}",
                 delta=f"{ai_slope - 1.0:+.3f} from theory",
                 delta_color="inverse")
    col_c.metric("R² Fit Quality",
                 f"{np.corrcoef(log_loads, log_ai_wears)[0,1]**2:.4f}",
                 help="How well AI predictions follow a power law on log-log scale")

    st.info(f"{verdict}\n\n{detail}")

# ── TAB 5: MATERIAL RANKING ───────────────────────────────────────
with tab5:
    st.subheader("🏆 Material Ranking")
    st.caption(
        f"All 9 materials ranked by AI predictions at **{coat}** | "
        f"**Temp={temp}°C** | **Load={load}N** | **Speed={speed}m/s** | **{env}**"
    )

    rank_tab_cof, rank_tab_wear, rank_tab_hard = st.tabs([
        "⚡ Rank by COF",
        "🔬 Rank by Wear Rate",
        "💎 Rank by Hardness",
    ])

    # ── Build one input row per material ──────────────────────────
    ranking_rows = []
    ranking_mats = []

    for material in all_materials:
        rp  = get_props_at_temp(material, coat, temp, mat_props, coat_props,
                                temp_mat_props, global_mean)
        row = dict.fromkeys(expected_columns, 0.0)
        row['Lunar Contact']       = 1 if "Yes" in env else 0
        row['Temperature (°C)']    = float(temp)
        row['Normal Load (N)']     = float(load)
        row['Sliding Speed (m/s)'] = float(speed)
        for pc, val in rp.items():
            if pc in row:
                row[pc] = val
        mc = f'Material_{material}'
        cc = f'Coating Material_{coat}'
        if mc in row: row[mc] = 1.0
        if cc in row: row[cc] = 1.0
        ranking_rows.append(row)
        ranking_mats.append(material)

    ranking_df     = pd.DataFrame(ranking_rows, columns=expected_columns)
    rank_cof_vals  = [max(float(v), 0.0) for v in model_cof.predict(ranking_df)]
    rank_wear_vals = [10 ** float(v)      for v in model_wear.predict(ranking_df)]

    results_df = pd.DataFrame({
        'Material':  ranking_mats,
        'COF':       rank_cof_vals,
        'Wear Rate': rank_wear_vals,
    })

    # ── Colour helper (FIXED) ─────────────────────────────────────
    # series and mat_list must be in the same order (already-sorted df)
    # ascending=True  → lowest value = green  (COF, Wear)
    # ascending=False → highest value = green (Hardness)
    def bar_colors_fixed(values, mat_list, selected, ascending=True):
        n      = len(values)
        order  = np.argsort(values)           # indices sorted best→worst
        if not ascending:
            order = order[::-1]
        rank_of = {mat_list[idx]: pos for pos, idx in enumerate(order)}
        colors  = []
        for material in mat_list:
            if material == selected:
                colors.append('#FFD700')      # gold
                continue
            pos  = rank_of[material]          # 0 = best, n-1 = worst
            norm = pos / max(n - 1, 1)        # 0.0 → 1.0
            r    = int(220 * norm)
            g    = int(180 * (1 - norm))
            colors.append(f'rgb({r},{g},50)')
        return colors

    # ── Shared layout updater ─────────────────────────────────────
    def rank_layout(fig, title_text, xtitle, xtype='linear'):
        fig.update_layout(
            title=dict(text=title_text, font=dict(size=13)),
            xaxis=dict(title=xtitle, gridcolor='lightgrey',
                       type=xtype,
                       exponentformat='e' if xtype == 'log' else None),
            yaxis=dict(title='', autorange='reversed'),
            height=400,
            plot_bgcolor='white',
            template='none',
            font=dict(family='serif', size=12),
            margin=dict(l=10, r=110, t=80, b=40),
        )

    def short(name):
        return (name.replace(' (WC-Co)', '').replace(' C17200', '')
                    .replace(' (Si3N4)', '').replace(' 7075', '')
                    .replace(' 301',    '').replace(' 718',   ''))

    # ════════════════════════════════════════════════════════════════
    # SUB-TAB A — Rank by COF
    # ════════════════════════════════════════════════════════════════
    with rank_tab_cof:
        cof_sorted         = results_df.sort_values('COF').reset_index(drop=True)
        cof_sorted['Rank'] = [f"#{i+1}" for i in range(len(cof_sorted))]
        clrs_cof           = bar_colors_fixed(
            cof_sorted['COF'].tolist(),
            cof_sorted['Material'].tolist(),
            mat, ascending=True
        )

        fig_cof = go.Figure(go.Bar(
            y=[f"{r}  {short(m)}" for r, m in zip(cof_sorted['Rank'], cof_sorted['Material'])],
            x=cof_sorted['COF'],
            orientation='h',
            marker=dict(color=clrs_cof, line=dict(color='black', width=0.6)),
            text=[f"{v:.4f}" for v in cof_sorted['COF']],
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>COF: %{x:.4f}<extra></extra>",
        ))

        sel_cof = results_df.loc[results_df['Material'] == mat, 'COF'].values[0]
        fig_cof.add_vline(x=sel_cof,
                          line=dict(color='#FFD700', width=2, dash='dash'),
                          annotation_text=f"{mat.split()[0]} ({sel_cof:.4f})",
                          annotation_position='top right',
                          annotation_font_color='#b8860b')

        rank_layout(fig_cof,
                    f"COF Ranking — {coat} | Temp={temp}°C | Load={load}N | Speed={speed}m/s | {env}",
                    "Predicted COF  (lower = better ✅)")
        st.plotly_chart(fig_cof, use_container_width=True)

        sel_rank_cof = cof_sorted[cof_sorted['Material'] == mat].index[0] + 1
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🥇 Best",  short(cof_sorted.iloc[0]['Material']),
                  f"COF = {cof_sorted.iloc[0]['COF']:.4f}")
        c2.metric("🔴 Worst", short(cof_sorted.iloc[-1]['Material']),
                  f"COF = {cof_sorted.iloc[-1]['COF']:.4f}")
        c3.metric("📍 Your Rank", f"#{sel_rank_cof} / 9",
                  f"COF = {sel_cof:.4f}")
        c4.metric("📉 Best↔Worst Gap",
                  f"{cof_sorted.iloc[-1]['COF'] - cof_sorted.iloc[0]['COF']:.4f}")

    # ════════════════════════════════════════════════════════════════
    # SUB-TAB B — Rank by Wear Rate
    # ════════════════════════════════════════════════════════════════
    with rank_tab_wear:
        wear_sorted         = results_df.sort_values('Wear Rate').reset_index(drop=True)
        wear_sorted['Rank'] = [f"#{i+1}" for i in range(len(wear_sorted))]
        clrs_wear           = bar_colors_fixed(
            wear_sorted['Wear Rate'].tolist(),
            wear_sorted['Material'].tolist(),
            mat, ascending=True
        )

        fig_wear = go.Figure(go.Bar(
            y=[f"{r}  {short(m)}" for r, m in zip(wear_sorted['Rank'], wear_sorted['Material'])],
            x=wear_sorted['Wear Rate'],
            orientation='h',
            marker=dict(color=clrs_wear, line=dict(color='black', width=0.6)),
            text=[f"{v:.2e}" for v in wear_sorted['Wear Rate']],
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Wear Rate: %{x:.3e} mm³/N·m<extra></extra>",
        ))

        sel_wear = results_df.loc[results_df['Material'] == mat, 'Wear Rate'].values[0]
        fig_wear.add_vline(x=sel_wear,
                           line=dict(color='#FFD700', width=2, dash='dash'),
                           annotation_text=f"{mat.split()[0]} ({sel_wear:.2e})",
                           annotation_position='top right',
                           annotation_font_color='#b8860b')

        rank_layout(fig_wear,
                    f"Wear Rate Ranking — {coat} | Temp={temp}°C | Load={load}N | Speed={speed}m/s | {env}",
                    "Predicted Wear Rate mm³/N·m  (lower = better ✅)",
                    xtype='log')
        st.plotly_chart(fig_wear, use_container_width=True)

        sel_rank_wear = wear_sorted[wear_sorted['Material'] == mat].index[0] + 1
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("🥇 Best",  short(wear_sorted.iloc[0]['Material']),
                  f"{wear_sorted.iloc[0]['Wear Rate']:.2e}")
        w2.metric("🔴 Worst", short(wear_sorted.iloc[-1]['Material']),
                  f"{wear_sorted.iloc[-1]['Wear Rate']:.2e}")
        w3.metric("📍 Your Rank", f"#{sel_rank_wear} / 9",
                  f"{sel_wear:.2e}")
        ratio = wear_sorted.iloc[-1]['Wear Rate'] / max(wear_sorted.iloc[0]['Wear Rate'], 1e-20)
        w4.metric("📉 Worst/Best Ratio", f"{ratio:.1f}×")

    # ════════════════════════════════════════════════════════════════
    # SUB-TAB C — Rank by Hardness
    # ════════════════════════════════════════════════════════════════
    with rank_tab_hard:
        hardness_vals = []
        for material in all_materials:
            rp = get_props_at_temp(material, coat, temp, mat_props, coat_props,
                                   temp_mat_props, global_mean)
            hardness_vals.append(rp.get('Hardness (HV)', 0.0))

        hard_df         = pd.DataFrame({'Material': all_materials,
                                        'Hardness': hardness_vals})
        hard_sorted     = hard_df.sort_values('Hardness', ascending=False).reset_index(drop=True)
        hard_sorted['Rank'] = [f"#{i+1}" for i in range(len(hard_sorted))]
        clrs_hard       = bar_colors_fixed(
            hard_sorted['Hardness'].tolist(),
            hard_sorted['Material'].tolist(),
            mat, ascending=False          # higher HV = green
        )

        fig_hard = go.Figure(go.Bar(
            y=[f"{r}  {short(m)}" for r, m in zip(hard_sorted['Rank'], hard_sorted['Material'])],
            x=hard_sorted['Hardness'],
            orientation='h',
            marker=dict(color=clrs_hard, line=dict(color='black', width=0.6)),
            text=[f"{v:.0f} HV" for v in hard_sorted['Hardness']],
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Hardness: %{x:.0f} HV<extra></extra>",
        ))

        sel_hard = hard_df.loc[hard_df['Material'] == mat, 'Hardness'].values[0]
        fig_hard.add_vline(x=sel_hard,
                           line=dict(color='#FFD700', width=2, dash='dash'),
                           annotation_text=f"{mat.split()[0]} ({sel_hard:.0f} HV)",
                           annotation_position='top right',
                           annotation_font_color='#b8860b')

        rank_layout(fig_hard,
                    f"Hardness Ranking — {coat} | Selected Temp={temp}°C  ",
                    "Hardness (HV)")
        st.plotly_chart(fig_hard, use_container_width=True)

        sel_rank_hard = hard_sorted[hard_sorted['Material'] == mat].index[0] + 1
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("🥇 Hardest",  short(hard_sorted.iloc[0]['Material']),
                  f"{hard_sorted.iloc[0]['Hardness']:.0f} HV")
        h2.metric("🔴 Softest",  short(hard_sorted.iloc[-1]['Material']),
                  f"{hard_sorted.iloc[-1]['Hardness']:.0f} HV")
        h3.metric("📍 Your Rank", f"#{sel_rank_hard} / 9",
                  f"{sel_hard:.0f} HV")
        hard_ratio = hard_sorted.iloc[0]['Hardness'] / max(hard_sorted.iloc[-1]['Hardness'], 1)
        h4.metric("📐 Hardest/Softest", f"{hard_ratio:.1f}×")

    # ── Overall verdict ───────────────────────────────────────────
    st.markdown("---")
    best_cof_mat  = cof_sorted.iloc[0]['Material']
    best_wear_mat = wear_sorted.iloc[0]['Material']

    if best_cof_mat == best_wear_mat:
        st.success(
            f"✅ **{best_cof_mat}** ranks #1 in both COF and Wear Rate "
            f"with **{coat}** at your current conditions."
        )
    else:
        st.warning(
            f"⚠️ Friction → **{best_cof_mat}** (#1 COF) | "
            f"Wear → **{best_wear_mat}** (#1 Wear Rate)"
        )
# -- TAB 6: THERMAL EXPANSION --
with tab6:
    st.subheader(f"📐 Thermal Expansion vs. Temperature ({coat} coating)")
    st.caption(
        "Tracking dimensional stability across the lunar temperature range. "
        "Lower values indicate better resistance to thermal shock and mechanical binding."
    )

    fig_te = go.Figure()

    # Calculate y-axis bounds for a clean graph
    te_all_vals = []
    for material in all_materials:
        for t in KNOWN_TEMPS:
            key3 = (material, coat, t)
            if key3 in temp_mat_props.index:
                val = float(temp_mat_props.loc[key3, 'Thermal Expansion (×10⁻⁶/°C)'])
                if not np.isnan(val):
                    te_all_vals.append(val)
    
    if te_all_vals:
        y_min_te = max(min(te_all_vals) * 0.85, 0)
        y_max_te = max(te_all_vals) * 1.15
    else:
        y_min_te, y_max_te = 0, 30

    # Plot each material
    for i, material in enumerate(all_materials):
        color = PALETTE[i % len(PALETTE)]
        key_m = (material, coat)
        mat_coat_idx = temp_mat_props.index.droplevel(-1)

        # Only plot if this material/coating combo exists
        if key_m not in mat_coat_idx:
            continue

        mean_vals = []
        for t in KNOWN_TEMPS:
            key3 = (material, coat, t)
            if key3 in temp_mat_props.index:
                mean_vals.append(float(temp_mat_props.loc[key3, 'Thermal Expansion (×10⁻⁶/°C)']))
            else:
                mean_vals.append(np.nan)

        is_selected = (material == mat)
        
        fig_te.add_trace(go.Scatter(
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
                "Th. Expansion: %{y:.2f} ×10⁻⁶/°C"
                "<extra></extra>"
            )
        ))

    # Add vertical line for the current temperature selected on the slider
    fig_te.add_vline(
        x=temp,
        line=dict(color='red', dash='dash', width=1.5),
        annotation_text=f"Current: {temp}°C",
        annotation_position="top right",
        annotation_font_color='red',
    )
    
    # Background temperature zones
    fig_te.add_vrect(x0=-173, x1=0, fillcolor='steelblue', opacity=0.04,
                    layer='below', line_width=0,
                    annotation_text='Cryogenic', annotation_position='top left',
                    annotation_font_size=10, annotation_font_color='steelblue')
    fig_te.add_vrect(x0=0, x1=127, fillcolor='tomato', opacity=0.04,
                    layer='below', line_width=0,
                    annotation_text='High-T', annotation_position='top right',
                    annotation_font_size=10, annotation_font_color='tomato')

    fig_te.update_layout(
        title=dict(
            text=f'Thermal Expansion across Lunar Temperatures<br>'
                 f'<sup>All materials with {coat} | Bold line = currently selected | '
                 f'Dotted red = {temp}°C</sup>',
            font=dict(size=14)
        ),
        xaxis_title='Temperature (°C)',
        yaxis=dict(title='Thermal Expansion (×10⁻⁶/°C)', range=[y_min_te, y_max_te]),
        hovermode='x unified',
        height=520,
        legend=dict(font=dict(size=10)),
        template='plotly_white'
    )
    st.plotly_chart(fig_te, use_container_width=True)
# ─────────────────────────────────────────────
# 13. MODEL ACCURACY REPORT (80/20 SPLIT)
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📊 Model Accuracy Report (80/20 Split)")
st.write("This section trains the XGBoost models on a random **80%** subset of the data and tests their accuracy on the unseen **20%** holdout set")

@st.cache_data
def evaluate_models():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    import numpy as np

    # Load data specifically for isolated evaluation
    df_eval = pd.read_csv('Lunar_Tribology_ML_Dataset__1_.csv')
    df_eval.columns = df_eval.columns.str.strip()

    wear_col = next(c for c in df_eval.columns if 'Wear Rate' in c)
    df_eval['Lunar Contact'] = df_eval['Lunar Contact'].str.strip().map({'Yes': 1, 'No': 0})
    df_eval['Log_Wear_Rate'] = np.log10(pd.to_numeric(df_eval[wear_col], errors='coerce').clip(lower=1e-12))

    df_enc = pd.get_dummies(df_eval, columns=['Material', 'Coating Material'])
    drop_cols = ['COF', wear_col, 'Log_Wear_Rate']
    X = df_enc.drop(columns=drop_cols)
    y_cof = df_enc['COF']
    y_wear = df_enc['Log_Wear_Rate']

    # 80/20 random split
    X_train, X_test, y_cof_train, y_cof_test, y_wear_train, y_wear_test = train_test_split(
        X, y_cof, y_wear, test_size=0.20, random_state=42
    )

    # Train params (matching the main model setup)
    params = dict(n_estimators=300, max_depth=6, learning_rate=0.08,
                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                  tree_method='hist')

    xgb_cof_eval = xgb.XGBRegressor(**params)
    xgb_wear_eval = xgb.XGBRegressor(**params)

    # Train on 80%
    xgb_cof_eval.fit(X_train, y_cof_train)
    xgb_wear_eval.fit(X_train, y_wear_train)

    # Predict on unseen 20%
    cof_preds = xgb_cof_eval.predict(X_test)
    wear_preds = xgb_wear_eval.predict(X_test)

    # Calculate metrics
    metrics = {
        "cof": {
            "r2": r2_score(y_cof_test, cof_preds),
            "mae": mean_absolute_error(y_cof_test, cof_preds),
            "rmse": np.sqrt(mean_squared_error(y_cof_test, cof_preds))
        },
        "wear": {
            "r2": r2_score(y_wear_test, wear_preds),
            "mae": mean_absolute_error(y_wear_test, wear_preds),
            "rmse": np.sqrt(mean_squared_error(y_wear_test, wear_preds))
        }
    }
    return metrics

with st.spinner("Evaluating model accuracy on a 20% holdout set..."):
    eval_metrics = evaluate_models()

col_eval1, col_eval2 = st.columns(2)

with col_eval1:
    st.subheader("⚡ Friction (COF) Predictor")
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{eval_metrics['cof']['r2']:.4f}", help="Closer to 1.0 is better")
    c2.metric("MAE", f"{eval_metrics['cof']['mae']:.4f}", help="Mean Absolute Error (lower is better)")
    c3.metric("RMSE", f"{eval_metrics['cof']['rmse']:.4f}", help="Root Mean Squared Error (lower is better)")

with col_eval2:
    st.subheader("🔬 Wear Rate Predictor (Log-Scale)")
    w1, w2, w3 = st.columns(3)
    w1.metric("R² Score", f"{eval_metrics['wear']['r2']:.4f}", help="Closer to 1.0 is better")
    w2.metric("MAE", f"{eval_metrics['wear']['mae']:.4f}", help="Mean Absolute Error (lower is better)")
    w3.metric("RMSE", f"{eval_metrics['wear']['rmse']:.4f}", help="Root Mean Squared Error (lower is better)")
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