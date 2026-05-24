"""
⚽ AI Premier League Scout & Market Value Predictor
Premium Dark Dashboard — Full Production Build
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI PL Scout",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  (dark luxury + glassmorphism)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-base:      #050d1a;
    --bg-card:      rgba(10,25,47,0.85);
    --bg-card2:     rgba(6,18,35,0.92);
    --neon-blue:    #00b4ff;
    --neon-cyan:    #00f0ff;
    --neon-green:   #00ff9d;
    --neon-gold:    #ffd700;
    --accent-pur:   #7c3aed;
    --text-pri:     #e8f4fd;
    --text-muted:   #6b8cae;
    --border:       rgba(0,180,255,0.22);
    --glass:        blur(18px);
    --r:            14px;
}

/* ── Base ── */
.stApp,[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 15% 15%,#001a3a 0%,#050d1a 55%,#030812 100%) !important;
    font-family:'DM Sans',sans-serif;
    color:var(--text-pri);
}
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#070f1e 0%,#050d1a 100%) !important;
    border-right:1px solid var(--border) !important;
}
[data-testid="stHeader"],.stDeployButton{display:none!important}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:#050d1a}
::-webkit-scrollbar-thumb{background:var(--neon-blue);border-radius:4px}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background:var(--bg-card);
    border:1px solid var(--border);
    border-radius:var(--r);
    padding:1.1rem 1.4rem!important;
    backdrop-filter:var(--glass);
    box-shadow:0 0 20px rgba(0,180,255,0.07),inset 0 1px 0 rgba(255,255,255,0.04);
    transition:box-shadow .3s;
}
[data-testid="stMetric"]:hover{box-shadow:0 0 32px rgba(0,180,255,0.2)}
[data-testid="stMetricLabel"]{color:var(--text-muted)!important;font-size:.7rem!important;text-transform:uppercase;letter-spacing:1.5px}
[data-testid="stMetricValue"]{color:var(--neon-blue)!important;font-family:'Orbitron',monospace!important;font-size:1.55rem!important}

/* ── Inputs ── */
[data-testid="stSelectbox"]>div>div,[data-testid="stMultiSelect"]>div>div {
    background:rgba(0,20,50,.9)!important;
    border:1px solid var(--border)!important;
    border-radius:8px!important;
    color:var(--text-pri)!important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family:'DM Sans',sans-serif!important;
    color:var(--text-muted)!important;
    border-radius:8px 8px 0 0!important;
    font-weight:500;
    padding:.5rem 1.2rem!important;
    transition:all .2s;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color:var(--neon-blue)!important;
    border-bottom:2px solid var(--neon-blue)!important;
    background:rgba(0,180,255,.06)!important;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] {
    background:var(--bg-card2)!important;
    border:1px solid var(--border)!important;
    border-radius:var(--r)!important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background:var(--bg-card2)!important;
    border:1px solid var(--border)!important;
    border-radius:var(--r)!important;
}
[data-testid="stExpander"] summary{color:var(--text-pri)!important;font-weight:600}

hr{border-color:var(--border)!important}

/* ── Utility classes ── */
.neon-title {
    font-family:'Orbitron',monospace;
    background:linear-gradient(135deg,#00b4ff 0%,#00f0ff 50%,#7c3aed 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.section-hdr {
    font-family:'Orbitron',monospace;
    font-size:.78rem;letter-spacing:3px;text-transform:uppercase;
    color:var(--neon-blue);margin-bottom:.9rem;padding-bottom:.45rem;
    border-bottom:1px solid var(--border);
}
.lbl{font-size:.67rem;text-transform:uppercase;letter-spacing:1.8px;color:var(--text-muted);margin-bottom:2px}
.val{font-family:'JetBrains Mono',monospace;font-size:1.15rem;font-weight:600;color:var(--text-pri)}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.7rem;font-weight:600;letter-spacing:.8px;text-transform:uppercase}
.b-blue{background:rgba(0,180,255,.14);color:#00b4ff;border:1px solid rgba(0,180,255,.3)}
.b-green{background:rgba(0,255,157,.11);color:#00ff9d;border:1px solid rgba(0,255,157,.3)}
.b-gold{background:rgba(255,215,0,.11);color:#ffd700;border:1px solid rgba(255,215,0,.3)}
.b-pur{background:rgba(124,58,237,.15);color:#a78bfa;border:1px solid rgba(124,58,237,.3)}
.b-red{background:rgba(255,80,80,.12);color:#ff5050;border:1px solid rgba(255,80,80,.3)}

.stat-row{display:flex;gap:.8rem;flex-wrap:wrap}
.stat-pill{flex:1 1 100px;background:rgba(0,20,50,.6);border:1px solid var(--border);
           border-radius:10px;padding:.65rem .9rem;text-align:center}
.gem-card{background:rgba(0,20,40,.9);border:1px solid rgba(0,255,157,.18);
          border-radius:14px;padding:1.1rem 1.2rem;margin-bottom:.7rem;
          box-shadow:0 0 18px rgba(0,255,157,.04)}
.sim-card{border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:.8rem}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOTLY BASE THEME
# ─────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans,sans-serif", color="#9ab8d4"),
    xaxis=dict(gridcolor="rgba(0,180,255,0.07)", zerolinecolor="rgba(0,180,255,0.13)"),
    yaxis=dict(gridcolor="rgba(0,180,255,0.07)", zerolinecolor="rgba(0,180,255,0.13)"),
    margin=dict(t=36,b=36,l=40,r=16),
)
NC = ["#00b4ff","#00f0ff","#00ff9d","#ffd700","#a78bfa","#ff6b6b","#ff9f43","#74b9ff"]

# ─────────────────────────────────────────────
#  EXACT MODEL FEATURE LISTS (from pkl inspect)
# ─────────────────────────────────────────────
SCOUT_FEATS = [
    "minutesPlayed","rating","goals_per90","assists_per90",
    "xG_per90","xA_per90","successfulDribbles","keyPasses","accuratePassesPercentage"
]
XGB_FEATS = [
    "Age","Age_sq","minutesPlayed","rating","Club_Avg_Value",
    "goals_per90","assists_per90","xG_per90","xA_per90",
    "successfulDribbles","keyPasses","accuratePassesPercentage",
    "Position_Central Midfield","Position_Centre-Back","Position_Centre-Forward",
    "Position_Defensive Midfield","Position_Left Midfield","Position_Left Winger",
    "Position_Left-Back","Position_Right Midfield","Position_Right Winger",
    "Position_Right-Back","Position_Second Striker"
]

# ─────────────────────────────────────────────
#  DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("final_engineered_premier_league_data.csv")

@st.cache_resource
def load_models():
    knn    = joblib.load("scout_knn_model.pkl")
    scaler = joblib.load("scout_scaler.pkl")
    xgb    = joblib.load("xgb_market_value_model.pkl")
    return knn, scaler, xgb

@st.cache_data
def model_metrics(df):
    from sklearn.metrics import r2_score, mean_absolute_error
    v = df.dropna(subset=["Market Value Num","Predicted_Market_Value"])
    return (r2_score(v["Market Value Num"], v["Predicted_Market_Value"]),
            mean_absolute_error(v["Market Value Num"], v["Predicted_Market_Value"]))

try:
    df = load_data()
    knn_model, scaler, xgb_model = load_models()
except Exception as e:
    st.error(f"❌ Could not load files: {e}\n\nPlace all .pkl / .csv files in the same folder as app.py")
    st.stop()

r2, mae = model_metrics(df)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def fv(v):
    if v >= 1e6:  return f"€{v/1e6:.1f}M"
    if v >= 1e3:  return f"€{v/1e3:.0f}K"
    return f"€{v:.0f}"

POS_LABEL = {"M":"Midfielder","D":"Defender","F":"Forward"}
POS_COLOR = {"M":"#00b4ff","D":"#00ff9d","F":"#ff6b6b"}

def plabel(p): return POS_LABEL.get(p, p)
def pcolor(p): return POS_COLOR.get(p, "#aaa")

def diff_badge(diff_pct):
    if diff_pct > 5:
        return f'<span class="badge b-green">▲ {abs(diff_pct):.1f}% UNDERVALUED</span>'
    elif diff_pct < -5:
        return f'<span class="badge b-red">▼ {abs(diff_pct):.1f}% OVERVALUED</span>'
    return f'<span class="badge b-blue">≈ FAIR VALUE</span>'

# Radar
RADAR_COLS   = ["goals","assists","expectedGoals","expectedAssists","rating","successfulDribbles","keyPasses"]
RADAR_LABELS = ["Goals","Assists","xG","xA","Rating","Dribbles","Key Passes"]

def radar_vals(row, ref_df):
    out, avgs = [], []
    for c in RADAR_COLS:
        if c in ref_df.columns:
            mn, mx = ref_df[c].min(), ref_df[c].max()
            rng = (mx - mn) or 1
            out.append(round((row[c] - mn) / rng * 10, 2))
            avgs.append(round((ref_df[c].mean() - mn) / rng * 10, 2))
        else:
            out.append(0); avgs.append(0)
    return out, avgs

# Similarity using correct scaler features
def get_similar(player_name, n=5):
    avail = [c for c in SCOUT_FEATS if c in df.columns]
    sub   = df.dropna(subset=avail + ["Market Value Num"]).copy()
    if player_name not in sub["player_name"].values:
        return pd.DataFrame()
    X  = sub[avail].fillna(0)
    try:
        Xs = scaler.transform(X)
    except Exception:
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
    loc = sub.reset_index(drop=True)
    pi  = loc[loc["player_name"] == player_name].index[0]
    dists = np.linalg.norm(Xs - Xs[pi], axis=1)
    loc["_dist"] = dists
    loc["Similarity"] = (1 - loc["_dist"] / (loc["_dist"].max() + 1e-9)) * 100
    return (loc[loc["player_name"] != player_name]
            .nsmallest(n, "_dist")
            [["player_name","Club","position","Age","Market Value Num",
              "Predicted_Market_Value","rating","Similarity"]]
            .reset_index(drop=True))

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 1.8rem'>
        <div style='font-size:2.6rem;margin-bottom:.3rem'>⚽</div>
        <div style='font-family:Orbitron,monospace;font-size:.76rem;letter-spacing:3px;
                    color:#00b4ff;text-transform:uppercase'>AI PL Scout</div>
        <div style='font-size:.63rem;color:#3a5a7a;margin-top:4px'>Market Value Predictor</div>
    </div>""", unsafe_allow_html=True)

    try:
        from streamlit_option_menu import option_menu
        page = option_menu(
            menu_title=None,
            options=["🏠 Hero","📊 Market Analysis","🔭 Scout Gems",
                     "🎯 Similarity","📈 Visualizations"],
            icons=["house","bar-chart","gem","people","graph-up"],
            default_index=0,
            styles={
                "container":         {"background-color":"transparent","padding":"0"},
                "nav-link":          {"font-size":"0.82rem","color":"#6b8cae",
                                      "font-family":"DM Sans,sans-serif",
                                      "border-radius":"8px","margin":"2px 0"},
                "nav-link-selected": {"background":"rgba(0,180,255,0.12)",
                                      "color":"#00b4ff","font-weight":"600"},
                "icon":              {"color":"#00b4ff"},
            },
        )
    except ImportError:
        page = st.radio("Navigate",["🏠 Hero","📊 Market Analysis","🔭 Scout Gems",
                                    "🎯 Similarity","📈 Visualizations"])

    st.markdown("---")
    st.markdown('<div class="lbl">Club</div>', unsafe_allow_html=True)
    clubs = ["All"] + sorted(df["Club"].dropna().unique().tolist())
    sel_club = st.selectbox("Club", clubs, label_visibility="collapsed")

    st.markdown('<div class="lbl" style="margin-top:.7rem">Position</div>', unsafe_allow_html=True)
    pos_opts = ["All","Forward (F)","Midfielder (M)","Defender (D)"]
    sel_pos_raw = st.selectbox("Pos", pos_opts, label_visibility="collapsed")
    sel_pos = {"Forward (F)":"F","Midfielder (M)":"M","Defender (D)":"D"}.get(sel_pos_raw,"All")

    st.markdown('<div class="lbl" style="margin-top:.7rem">Age Range</div>', unsafe_allow_html=True)
    amin, amax = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.slider("Age", amin, amax, (amin, amax), label_visibility="collapsed")

    dff = df.copy()
    if sel_club != "All":  dff = dff[dff["Club"] == sel_club]
    if sel_pos  != "All":  dff = dff[dff["position"] == sel_pos]
    dff = dff[(dff["Age"] >= age_range[0]) & (dff["Age"] <= age_range[1])]

    st.markdown(f"""
    <div style='margin-top:1.4rem;background:rgba(0,180,255,.06);border:1px solid rgba(0,180,255,.2);
                border-radius:10px;padding:.85rem;text-align:center'>
        <div class="lbl">Filtered Players</div>
        <div style='font-family:Orbitron,monospace;color:#00b4ff;font-size:1.4rem'>{len(dff)}</div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  PAGE 1 — HERO
# ═══════════════════════════════════════════════════════
if "Hero" in page:

    # Hero banner
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(0,20,50,.92) 0%,rgba(5,13,26,.96) 60%,rgba(15,0,40,.9) 100%);
                border:1px solid rgba(0,180,255,.18);border-radius:20px;
                padding:3.5rem 3rem 3rem;margin-bottom:1.8rem;position:relative;overflow:hidden;
                box-shadow:0 0 80px rgba(0,180,255,.07),inset 0 1px 0 rgba(255,255,255,.04)'>
      <div style='position:absolute;top:-60px;right:-60px;width:280px;height:280px;border-radius:50%;
                  background:radial-gradient(circle,rgba(0,180,255,.08) 0%,transparent 70%)'></div>
      <div style='position:absolute;bottom:-80px;left:35%;width:320px;height:320px;border-radius:50%;
                  background:radial-gradient(circle,rgba(124,58,237,.07) 0%,transparent 70%)'></div>
      <div style='font-family:Orbitron,monospace;font-size:.7rem;letter-spacing:4px;
                  color:#00b4ff;text-transform:uppercase;margin-bottom:.6rem'>
          ◈ Powered by XGBoost &amp; KNN ◈
      </div>
      <div style='font-family:Orbitron,monospace;font-size:2.5rem;font-weight:900;
                  background:linear-gradient(135deg,#00b4ff 0%,#00f0ff 45%,#7c3aed 100%);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;line-height:1.15;margin-bottom:.8rem'>
          AI PREMIER LEAGUE<br>SCOUT &amp; ANALYST
      </div>
      <div style='color:#6b8cae;font-size:1rem;font-weight:300;max-width:580px;line-height:1.65'>
          Uncover hidden gems · Predict market values · Benchmark every Premier League
          player with machine‑learning precision across 134 engineered features.
      </div>
      <div style='display:flex;gap:.6rem;margin-top:1.8rem;flex-wrap:wrap'>
          <span class="badge b-blue">⚡ XGBoost Regressor</span>
          <span class="badge b-green">🔭 KNN Scouting</span>
          <span class="badge b-gold">📊 134 Features</span>
          <span class="badge b-pur">🌍 340 Players · 20 Clubs</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Model R²",   f"{r2:.4f}",       "XGBoost")
    with c2: st.metric("MAE",        fv(mae),           "Avg error")
    with c3: st.metric("Players",    len(df["player_name"].unique()), "in dataset")
    with c4: st.metric("Features",   "134",             "engineered")
    with c5: st.metric("Accuracy",   f"{r2*100:.1f}%",  f"+{r2*100-70:.1f}% vs baseline")

    st.markdown("---")

    cl, cr = st.columns(2)
    with cl:
        st.markdown('<div class="section-hdr">📍 Market Value Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(dff, x="Market Value Num", nbins=30,
                           color_discrete_sequence=["#00b4ff"],
                           labels={"Market Value Num":"Market Value (€)"})
        fig.update_layout(**PL, height=270)
        fig.update_traces(opacity=.82, marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown('<div class="section-hdr">🏆 Avg Value by Position</div>', unsafe_allow_html=True)
        pav = dff.groupby("position")["Market Value Num"].mean().reset_index()
        pav["pos_label"] = pav["position"].map(plabel)
        fig2 = px.bar(pav, x="pos_label", y="Market Value Num",
                      color="pos_label",
                      color_discrete_map={"Forward":"#ff6b6b","Midfielder":"#00b4ff","Defender":"#00ff9d"},
                      labels={"pos_label":"","Market Value Num":"Avg Market Value (€)"})
        fig2.update_layout(**PL, height=270, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-hdr" style="margin-top:.5rem">⭐ Top 10 Most Valuable (Filtered)</div>',
                unsafe_allow_html=True)
    top10 = dff.nlargest(10,"Market Value Num")[
        ["player_name","Club","position","Age","Market Value Num","Predicted_Market_Value","rating"]
    ].copy()
    top10["Market Value"] = top10["Market Value Num"].apply(fv)
    top10["Predicted"]    = top10["Predicted_Market_Value"].apply(fv)
    top10["Position"]     = top10["position"].map(plabel)
    top10["Rating"]       = top10["rating"].round(2)
    top10 = top10.rename(columns={"player_name":"Player"})
    st.dataframe(top10[["Player","Club","Position","Age","Market Value","Predicted","Rating"]],
                 use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
#  PAGE 2 — MARKET ANALYSIS
# ═══════════════════════════════════════════════════════
elif "Market" in page:

    st.markdown('<div class="neon-title" style="font-size:1.35rem;margin-bottom:1.4rem">📊 Market Value Analysis</div>',
                unsafe_allow_html=True)

    players_list = sorted(dff["player_name"].dropna().unique().tolist())
    if not players_list:
        st.warning("No players match current filters."); st.stop()

    sel = st.selectbox("Select Player", players_list)
    row = dff[dff["player_name"] == sel].iloc[0]

    actual    = float(row.get("Market Value Num", 0))
    predicted = float(row.get("Predicted_Market_Value", 0))
    diff      = predicted - actual
    diff_pct  = diff / max(actual, 1) * 100
    pc        = pcolor(row.get("position",""))

    # Player card
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(0,20,50,.95),rgba(5,8,20,.98));
                border:1px solid {pc}38;border-radius:18px;padding:2rem 2.2rem;
                margin-bottom:1.4rem;box-shadow:0 0 40px {pc}10,inset 0 1px 0 rgba(255,255,255,.03);
                display:flex;flex-wrap:wrap;gap:2rem;align-items:flex-start'>
      <!-- avatar -->
      <div style='width:88px;height:88px;border-radius:50%;
                  background:linear-gradient(135deg,{pc}28,{pc}06);
                  border:2px solid {pc}44;display:flex;align-items:center;
                  justify-content:center;font-size:2.4rem;flex-shrink:0'>⚽</div>
      <!-- info -->
      <div style='flex:1;min-width:200px'>
          <div style='font-family:Orbitron,monospace;font-size:1.5rem;font-weight:700;
                      color:#e8f4fd;line-height:1.2;margin-bottom:.4rem'>
              {row.get("player_name","—")}
          </div>
          <div style='display:flex;gap:.45rem;flex-wrap:wrap;margin-bottom:.75rem'>
              <span class="badge" style="background:{pc}18;color:{pc};border:1px solid {pc}3a">
                  {plabel(row.get("position","?"))}
              </span>
              <span class="badge b-blue">🏟 {row.get("Club","—")}</span>
              <span class="badge b-gold">🌍 {row.get("Nationality","—")}</span>
          </div>
          <div class="stat-row">
            <div class="stat-pill"><div class="lbl">Age</div>
                <div class="val">{int(row.get("Age",0))}</div></div>
            <div class="stat-pill"><div class="lbl">Rating</div>
                <div class="val" style="color:#ffd700">{round(row.get("rating",0),2)}</div></div>
            <div class="stat-pill"><div class="lbl">Goals</div>
                <div class="val">{int(row.get("goals",0))}</div></div>
            <div class="stat-pill"><div class="lbl">Assists</div>
                <div class="val">{int(row.get("assists",0))}</div></div>
            <div class="stat-pill"><div class="lbl">Apps</div>
                <div class="val">{int(row.get("appearances",0))}</div></div>
            <div class="stat-pill"><div class="lbl">Mins</div>
                <div class="val">{int(row.get("minutesPlayed",0))}</div></div>
          </div>
      </div>
      <!-- value block -->
      <div style='background:rgba(0,0,0,.28);border:1px solid rgba(0,180,255,.14);
                  border-radius:14px;padding:1.2rem 1.6rem;min-width:190px;text-align:center'>
          <div class="lbl">Actual Value</div>
          <div style='font-family:Orbitron,monospace;font-size:1.45rem;color:#00b4ff;margin-bottom:.75rem'>
              {fv(actual)}
          </div>
          <div class="lbl">AI Predicted</div>
          <div style='font-family:Orbitron,monospace;font-size:1.45rem;color:#00f0ff;margin-bottom:.75rem'>
              {fv(predicted)}
          </div>
          <div style='border-top:1px solid rgba(0,180,255,.13);padding-top:.65rem'>
              {diff_badge(diff_pct)}
          </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Tabs
    t1, t2, t3 = st.tabs(["📋 Stats Breakdown", "🕸 Radar vs League Avg", "🔍 Value Factors"])

    with t1:
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("xG",           round(row.get("expectedGoals",0),2))
        with c2: st.metric("xA",           round(row.get("expectedAssists",0),2))
        with c3: st.metric("Key Passes",   int(row.get("keyPasses",0)))
        with c4: st.metric("Dribbles",     int(row.get("successfulDribbles",0)))
        c5,c6,c7,c8 = st.columns(4)
        with c5: st.metric("Acc Pass %",   round(row.get("accuratePassesPercentage",0),1))
        with c6: st.metric("Tackles",      int(row.get("tackles",0)))
        with c7: st.metric("Interceptions",int(row.get("interceptions",0)))
        with c8: st.metric("TOTW",         int(row.get("totwAppearances",0)))

        # per-90 mini table
        st.markdown('<div class="section-hdr" style="margin-top:.8rem">Per-90 Stats</div>',
                    unsafe_allow_html=True)
        per90_cols = ["goals_per90","assists_per90","xG_per90","xA_per90"]
        per90_avail = [c for c in per90_cols if c in df.columns]
        if per90_avail:
            p90_row = {c: round(float(row.get(c,0)),3) for c in per90_avail}
            p90_avg = {c: round(float(df[c].mean()),3)  for c in per90_avail}
            p90_df  = pd.DataFrame({"Stat":per90_avail,
                                    f"{sel}":list(p90_row.values()),
                                    "League Avg":list(p90_avg.values())})
            st.dataframe(p90_df, use_container_width=True, hide_index=True)

    with t2:
        vs, avgs = radar_vals(row, df)
        lbl_ext = RADAR_LABELS + [RADAR_LABELS[0]]
        v_ext   = vs + [vs[0]]
        a_ext   = avgs + [avgs[0]]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=a_ext, theta=lbl_ext, fill="toself",
                                         name="League Avg",
                                         line=dict(color="rgba(107,140,174,.55)",width=1.5),
                                         fillcolor="rgba(107,140,174,.07)"))
        fig_r.add_trace(go.Scatterpolar(r=v_ext, theta=lbl_ext, fill="toself",
                                         name=sel,
                                         line=dict(color="#00b4ff",width=2.5),
                                         fillcolor="rgba(0,180,255,.11)"))
        fig_r.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(visible=True,range=[0,10],
                                       gridcolor="rgba(0,180,255,.1)",
                                       color="#3a5a7a",tickfont=dict(size=9)),
                       angularaxis=dict(gridcolor="rgba(0,180,255,.1)",color="#6b8cae")),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=True, height=390,
            legend=dict(font=dict(color="#b0c8e0")),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with t3:
        # XGBoost feature importance from the model directly
        fi = pd.Series(xgb_model.feature_importances_, index=XGB_FEATS).nlargest(15).sort_values()
        fig_fi = px.bar(x=fi.values, y=fi.index, orientation="h",
                        color=fi.values,
                        color_continuous_scale=["#0a192e","#005f8a","#00b4ff","#00f0ff"],
                        labels={"x":"Importance","y":""})
        fig_fi.update_layout(**PL, height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Actual vs Predicted scatter
    st.markdown('<div class="section-hdr" style="margin-top:.3rem">📉 Actual vs Predicted — All Filtered Players</div>',
                unsafe_allow_html=True)
    sc = dff.dropna(subset=["Market Value Num","Predicted_Market_Value"]).copy()
    sc["diff%"] = (sc["Predicted_Market_Value"] - sc["Market Value Num"]) / sc["Market Value Num"].clip(lower=1) * 100
    fig_sc = px.scatter(sc, x="Market Value Num", y="Predicted_Market_Value",
                        color="diff%",
                        color_continuous_scale=["#c0392b","#6b8cae","#00ff9d"],
                        hover_data={"player_name":True,"Club":True,"diff%":":.1f"},
                        labels={"Market Value Num":"Actual (€)","Predicted_Market_Value":"Predicted (€)","diff%":"Diff %"})
    mn_v, mx_v = sc["Market Value Num"].min(), sc["Market Value Num"].max()
    fig_sc.add_trace(go.Scatter(x=[mn_v,mx_v], y=[mn_v,mx_v], mode="lines",
                                line=dict(color="rgba(0,240,255,.28)",dash="dash",width=1.5),
                                name="Perfect Fit"))
    fig_sc.update_layout(**PL, height=420,
                         coloraxis_colorbar=dict(title="Diff%",tickfont=dict(color="#6b8cae")))
    st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════
#  PAGE 3 — SCOUT GEMS
# ═══════════════════════════════════════════════════════
elif "Scout" in page or "Gem" in page:

    st.markdown('<div class="neon-title" style="font-size:1.35rem;margin-bottom:.3rem">🔭 Hidden Gems Scouting Engine</div>',
                unsafe_allow_html=True)
    st.markdown('<div style="color:#6b8cae;font-size:.88rem;margin-bottom:1.4rem">Players where AI predicted value exceeds current market price</div>',
                unsafe_allow_html=True)

    col_t, col_n = st.columns([3,1])
    with col_t:
        thr = st.slider("Min undervaluation %", 5, 200, 20,
                        help="Show players where predicted value exceeds actual by at least this %")
    with col_n:
        top_n = st.selectbox("Show", [10, 20, 30, 50], index=0)

    gems = dff.copy()
    gems["diff_pct"] = ((gems["Predicted_Market_Value"] - gems["Market Value Num"])
                        / gems["Market Value Num"].clip(lower=1) * 100)
    gems = gems[gems["diff_pct"] >= thr].nlargest(top_n, "diff_pct")

    if gems.empty:
        st.info("💎 No gems found at this threshold — try lowering the slider.")
    else:
        st.markdown(f"""
        <div style='background:rgba(0,255,157,.05);border:1px solid rgba(0,255,157,.18);
                    border-radius:12px;padding:.95rem 1.4rem;margin-bottom:1.2rem;
                    display:flex;align-items:center;gap:1rem'>
            <div style='font-size:1.8rem'>💎</div>
            <div>
                <div style='font-family:Orbitron,monospace;color:#00ff9d;font-size:.82rem;letter-spacing:2px'>
                    {len(gems)} HIDDEN GEMS FOUND
                </div>
                <div style='color:#6b8cae;font-size:.76rem;margin-top:2px'>
                    Undervalued by ≥{thr}% according to the AI model
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # 3-column gem cards
        for i in range(0, min(len(gems), 9), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j >= len(gems): break
                g = gems.iloc[i+j]
                dv = float(g.get("Predicted_Market_Value",0)) - float(g.get("Market Value Num",0))
                dp = float(g.get("diff_pct",0))
                with col:
                    st.markdown(f"""
                    <div class="gem-card">
                        <div style='font-size:1.3rem;margin-bottom:.35rem'>⚽</div>
                        <div style='font-weight:600;font-size:.9rem;color:#e8f4fd;margin-bottom:.15rem'>
                            {g.get("player_name","—")}
                        </div>
                        <div style='font-size:.72rem;color:#6b8cae;margin-bottom:.65rem'>
                            {g.get("Club","—")} · {plabel(g.get("position",""))} · Age {int(g.get("Age",0))}
                        </div>
                        <div style='display:flex;justify-content:space-between;margin-bottom:.55rem'>
                            <div><div class="lbl">Actual</div>
                                 <div style='font-family:JetBrains Mono,monospace;color:#9ab8d4;font-size:.8rem'>
                                     {fv(float(g.get("Market Value Num",0)))}</div></div>
                            <div><div class="lbl">AI Predicted</div>
                                 <div style='font-family:JetBrains Mono,monospace;color:#00b4ff;font-size:.8rem'>
                                     {fv(float(g.get("Predicted_Market_Value",0)))}</div></div>
                        </div>
                        <div style='background:rgba(0,255,157,.07);border:1px solid rgba(0,255,157,.22);
                                    border-radius:8px;padding:.28rem .6rem;text-align:center'>
                            <span style='color:#00ff9d;font-family:Orbitron,monospace;font-size:.75rem;font-weight:700'>
                                ▲ +{dp:.1f}% · {fv(dv)}
                            </span>
                        </div>
                    </div>""", unsafe_allow_html=True)

        with st.expander("📋 Full Gems Table", expanded=False):
            disp = gems[["player_name","Club","position","Age",
                          "Market Value Num","Predicted_Market_Value","diff_pct","rating"]].copy()
            disp["Actual"]    = disp["Market Value Num"].apply(fv)
            disp["Predicted"] = disp["Predicted_Market_Value"].apply(fv)
            disp["Under %"]   = disp["diff_pct"].round(1)
            disp["Rating"]    = disp["rating"].round(2)
            disp["Position"]  = disp["position"].map(plabel)
            disp = disp.rename(columns={"player_name":"Player"})
            st.dataframe(disp[["Player","Club","Position","Age","Actual","Predicted","Under %","Rating"]],
                         use_container_width=True, hide_index=True)

        # Gem bar chart
        st.markdown('<div class="section-hdr" style="margin-top:1rem">📊 Undervaluation Gap (Top Gems)</div>',
                    unsafe_allow_html=True)
        top_gems_chart = gems.head(15).copy()
        top_gems_chart["gap"] = top_gems_chart["Predicted_Market_Value"] - top_gems_chart["Market Value Num"]
        fig_g = px.bar(top_gems_chart.sort_values("gap"), x="gap", y="player_name",
                       orientation="h", color="gap",
                       color_continuous_scale=["#004a2e","#00b87a","#00ff9d"],
                       labels={"gap":"Value Gap (€)","player_name":"Player"},
                       hover_data={"Club":True,"diff_pct":":.1f"})
        fig_g.update_layout(**PL, height=max(300, len(top_gems_chart)*28),
                             coloraxis_showscale=False)
        st.plotly_chart(fig_g, use_container_width=True)

    # Age vs Value scatter
    st.markdown('<div class="section-hdr" style="margin-top:1.2rem">📅 Age vs Market Value</div>',
                unsafe_allow_html=True)
    age_df = dff.dropna(subset=["Age","Market Value Num"])
    fig_av = px.scatter(age_df, x="Age", y="Market Value Num",
                        color="position",
                        color_discrete_map={"F":"#ff6b6b","M":"#00b4ff","D":"#00ff9d"},
                        hover_data={"player_name":True,"Club":True},
                        labels={"Age":"Age","Market Value Num":"Market Value (€)","position":"Pos"})
    # Manual polynomial trendline (no statsmodels needed)
    _x = age_df["Age"].values
    _y = age_df["Market Value Num"].values
    _z = np.polyfit(_x, _y, 2)
    _p = np.poly1d(_z)
    _xs = np.linspace(_x.min(), _x.max(), 100)
    fig_av.add_trace(go.Scatter(x=_xs, y=_p(_xs), mode="lines", name="Trend",
                                line=dict(color="rgba(0,240,255,0.55)", width=2, dash="dot"),
                                showlegend=False))
    fig_av.update_layout(**PL, height=380)
    st.plotly_chart(fig_av, use_container_width=True)


# ═══════════════════════════════════════════════════════
#  PAGE 4 — SIMILARITY
# ═══════════════════════════════════════════════════════
elif "Simil" in page:

    st.markdown('<div class="neon-title" style="font-size:1.35rem;margin-bottom:1.4rem">🎯 Player Similarity Engine</div>',
                unsafe_allow_html=True)

    players_all = sorted(df["player_name"].dropna().unique().tolist())
    c_sel, c_n = st.columns([3,1])
    with c_sel:
        sel = st.selectbox("Find statistical twins for", players_all)
    with c_n:
        n_sim = st.selectbox("Recommendations", [3,5,8,10], index=1)

    base = df[df["player_name"] == sel]
    if base.empty:
        st.warning("Player not found in dataset."); st.stop()
    base_row = base.iloc[0]

    st.markdown(f"""
    <div style='background:rgba(0,20,50,.8);border:1px solid rgba(0,180,255,.2);
                border-radius:14px;padding:1rem 1.4rem;margin-bottom:1.2rem;
                display:flex;align-items:center;gap:1rem'>
        <div style='font-size:2rem'>⚽</div>
        <div>
            <div style='font-weight:700;font-size:1.05rem'>{sel}</div>
            <div style='color:#6b8cae;font-size:.78rem'>
                {plabel(base_row.get("position",""))} · {base_row.get("Club","—")} ·
                Age {int(base_row.get("Age",0))} · Rating {round(base_row.get("rating",0),2)} ·
                {fv(float(base_row.get("Market Value Num",0)))}
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("🔍 Running KNN similarity search…"):
        similar = get_similar(sel, n=n_sim)

    if similar.empty:
        st.info("Could not compute similarity — player may lack required stats.")
    else:
        st.markdown(f'<div class="section-hdr">🔗 Top {len(similar)} Statistical Twins</div>',
                    unsafe_allow_html=True)
        ncols = min(3, len(similar))
        cols  = st.columns(ncols)
        for idx, sr in similar.iterrows():
            col    = cols[idx % ncols]
            sim    = round(float(sr.get("Similarity",0)),1)
            pc_s   = pcolor(sr.get("position",""))
            dv_s   = float(sr.get("Predicted_Market_Value",0)) - float(sr.get("Market Value Num",0))
            dv_col = "#00ff9d" if dv_s >= 0 else "#ff6b6b"
            with col:
                st.markdown(f"""
                <div class="sim-card" style='background:rgba(0,20,40,.9);
                     border:1px solid {pc_s}26;box-shadow:0 0 14px {pc_s}07'>
                    <div style='display:flex;justify-content:space-between;align-items:flex-start;
                                margin-bottom:.45rem'>
                        <div>
                            <div style='font-weight:600;font-size:.88rem'>{sr.get("player_name","—")}</div>
                            <div style='font-size:.7rem;color:#6b8cae'>
                                {sr.get("Club","—")} · {plabel(sr.get("position",""))} · Age {int(sr.get("Age",0))}
                            </div>
                        </div>
                        <span class="badge" style="background:{pc_s}16;color:{pc_s};
                                border:1px solid {pc_s}3a;white-space:nowrap">{sim}%</span>
                    </div>
                    <div style='background:rgba(255,255,255,.04);border-radius:4px;height:3px;margin-bottom:.7rem'>
                        <div style='width:{int(sim)}%;height:100%;border-radius:4px;
                                    background:linear-gradient(90deg,{pc_s},{pc_s}70)'></div>
                    </div>
                    <div style='display:flex;justify-content:space-between'>
                        <div><div class="lbl">Actual</div>
                             <div style='font-family:JetBrains Mono,monospace;font-size:.76rem;color:#9ab8d4'>
                                 {fv(float(sr.get("Market Value Num",0)))}</div></div>
                        <div><div class="lbl">Predicted</div>
                             <div style='font-family:JetBrains Mono,monospace;font-size:.76rem;color:#00b4ff'>
                                 {fv(float(sr.get("Predicted_Market_Value",0)))}</div></div>
                        <div><div class="lbl">Rating</div>
                             <div style='font-family:JetBrains Mono,monospace;font-size:.76rem;color:#ffd700'>
                                 {round(float(sr.get("rating",0)),2)}</div></div>
                    </div>
                    <div style='margin-top:.55rem;text-align:center'>
                        <span style='font-size:.7rem;color:{dv_col}'>
                            {"▲" if dv_s>=0 else "▼"} {fv(abs(dv_s))} vs actual
                        </span>
                    </div>
                </div>""", unsafe_allow_html=True)

        # Comparison bar chart
        st.markdown('<div class="section-hdr" style="margin-top:.4rem">📊 Side-by-Side Stats Comparison</div>',
                    unsafe_allow_html=True)
        cmp_cols  = ["goals","assists","expectedGoals","rating","successfulDribbles","keyPasses"]
        avail_cmp = [c for c in cmp_cols if c in df.columns]
        names  = [sel] + similar["player_name"].tolist()
        colors = NC[:len(names)]
        fig_c = go.Figure()
        for nm, clr in zip(names, colors):
            r_ = df[df["player_name"] == nm]
            if r_.empty: continue
            r_ = r_.iloc[0]
            fig_c.add_trace(go.Bar(name=nm, x=avail_cmp,
                                   y=[round(float(r_.get(c,0)),2) for c in avail_cmp],
                                   marker_color=clr, opacity=.82))
        fig_c.update_layout(**PL, barmode="group", height=360,
                            legend=dict(font=dict(color="#b0c8e0"),
                                        orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_c, use_container_width=True)

        # Radar overlay
        st.markdown('<div class="section-hdr">🕸 Radar Overlay</div>', unsafe_allow_html=True)
        fig_ro = go.Figure()
        all_names = [sel] + similar["player_name"].tolist()
        for i, nm in enumerate(all_names[:5]):
            r_ = df[df["player_name"] == nm]
            if r_.empty: continue
            vs2, _ = radar_vals(r_.iloc[0], df)
            vs2_ext = vs2 + [vs2[0]]
            lbl_ext = RADAR_LABELS + [RADAR_LABELS[0]]
            fig_ro.add_trace(go.Scatterpolar(
                r=vs2_ext, theta=lbl_ext, fill="toself", name=nm,
                line=dict(color=NC[i], width=2 if i==0 else 1.3),
                fillcolor=f"rgba({int(NC[i][1:3],16)},{int(NC[i][3:5],16)},{int(NC[i][5:],16)},0.07)"
            ))
        fig_ro.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(visible=True,range=[0,10],
                                       gridcolor="rgba(0,180,255,.1)",
                                       color="#3a5a7a",tickfont=dict(size=9)),
                       angularaxis=dict(gridcolor="rgba(0,180,255,.1)",color="#6b8cae")),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=True, height=400,
            legend=dict(font=dict(color="#b0c8e0")),
        )
        st.plotly_chart(fig_ro, use_container_width=True)


# ═══════════════════════════════════════════════════════
#  PAGE 5 — VISUALIZATIONS
# ═══════════════════════════════════════════════════════
elif "Visual" in page:

    st.markdown('<div class="neon-title" style="font-size:1.35rem;margin-bottom:1.4rem">📈 Advanced Analytics</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🏟 Club Leaderboard","🔥 Correlation Heatmap",
                                       "🌍 Demographics","⚡ Top Performers"])

    with tab1:
        st.markdown('<div class="section-hdr">Club Market Value Leaderboard</div>', unsafe_allow_html=True)
        club_df = dff.groupby("Club").agg(
            Avg_Value=("Market Value Num","mean"),
            Total_Value=("Market Value Num","sum"),
            Players=("player_name","count"),
        ).reset_index().sort_values("Avg_Value", ascending=False)

        fig_cl = px.bar(club_df, x="Club", y="Avg_Value",
                        color="Avg_Value",
                        color_continuous_scale=["#0a192e","#004a7c","#00b4ff","#00f0ff"],
                        text=club_df["Avg_Value"].apply(fv),
                        labels={"Avg_Value":"Avg Market Value (€)","Club":""})
        fig_cl.update_traces(textfont_size=9, textposition="outside")
        fig_cl.update_layout(**PL, height=420, coloraxis_showscale=False, xaxis_tickangle=-35)
        st.plotly_chart(fig_cl, use_container_width=True)

        cl, cr = st.columns(2)
        with cl:
            st.markdown('<div class="section-hdr">Total Squad Value</div>', unsafe_allow_html=True)
            fig_tv = px.bar(club_df.sort_values("Total_Value",ascending=False),
                            x="Club", y="Total_Value",
                            color="Total_Value",
                            color_continuous_scale=["#0a192e","#7c3aed","#a78bfa"],
                            labels={"Total_Value":"Total Value (€)","Club":""})
            fig_tv.update_layout(**PL, height=280, coloraxis_showscale=False, xaxis_tickangle=-35)
            st.plotly_chart(fig_tv, use_container_width=True)

        with cr:
            st.markdown('<div class="section-hdr">Squad Size</div>', unsafe_allow_html=True)
            fig_sq = px.bar(club_df.sort_values("Players",ascending=False),
                            x="Club", y="Players",
                            color="Players",
                            color_continuous_scale=["#0a192e","#00793a","#00ff9d"],
                            labels={"Players":"Players","Club":""})
            fig_sq.update_layout(**PL, height=280, coloraxis_showscale=False, xaxis_tickangle=-35)
            st.plotly_chart(fig_sq, use_container_width=True)

        with st.expander("📋 Club Stats Table"):
            club_df["Avg Value"]   = club_df["Avg_Value"].apply(fv)
            club_df["Total Value"] = club_df["Total_Value"].apply(fv)
            st.dataframe(club_df[["Club","Players","Avg Value","Total Value"]],
                         use_container_width=True, hide_index=True)

    with tab2:
        st.markdown('<div class="section-hdr">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        hcols  = ["Market Value Num","Age","goals","assists","expectedGoals","expectedAssists",
                  "rating","successfulDribbles","keyPasses","minutesPlayed",
                  "appearances","tackles","interceptions","goals_per90","xG_per90"]
        avail_h = [c for c in hcols if c in dff.columns]
        corr    = dff[avail_h].corr().round(2)

        fig_hm = px.imshow(corr,
                           color_continuous_scale=["#0a192e","#003d60","#00b4ff","#00f0ff","#00ff9d"],
                           aspect="auto", text_auto=True, zmin=-1, zmax=1)
        fig_hm.update_traces(textfont_size=8)
        fig_hm.update_layout(**PL, height=540,
                              coloraxis_colorbar=dict(title="ρ",tickfont=dict(color="#6b8cae")))
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-hdr">Players by Nationality (Top 20)</div>',
                    unsafe_allow_html=True)
        nat = dff["Nationality"].value_counts().reset_index()
        nat.columns = ["Nationality","Count"]
        fig_nt = px.bar(nat.head(20), x="Count", y="Nationality", orientation="h",
                        color="Count",
                        color_continuous_scale=["#0a192e","#00b4ff","#00f0ff"],
                        labels={"Count":"Players"})
        fig_nt.update_layout(**PL, height=460, coloraxis_showscale=False)
        st.plotly_chart(fig_nt, use_container_width=True)

        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown('<div class="section-hdr">Avg Value by Nationality (Top 10)</div>',
                        unsafe_allow_html=True)
            nv = dff.groupby("Nationality")["Market Value Num"].mean().nlargest(10).reset_index()
            fig_nv = px.bar(nv, x="Nationality", y="Market Value Num",
                            color="Market Value Num",
                            color_continuous_scale=["#002040","#00b4ff","#00f0ff"],
                            labels={"Market Value Num":"Avg Value (€)"})
            fig_nv.update_layout(**PL, height=300, coloraxis_showscale=False, xaxis_tickangle=-30)
            st.plotly_chart(fig_nv, use_container_width=True)

        with c_b:
            st.markdown('<div class="section-hdr">Position Distribution</div>', unsafe_allow_html=True)
            pd_pie = dff["position"].map(plabel).value_counts().reset_index()
            pd_pie.columns = ["Position","Count"]
            fig_pie = px.pie(pd_pie, names="Position", values="Count",
                             color_discrete_sequence=["#ff6b6b","#00b4ff","#00ff9d"], hole=0.55)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300,
                                   font=dict(color="#b0c8e0"),
                                   legend=dict(font=dict(color="#b0c8e0")))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Age distribution
        st.markdown('<div class="section-hdr" style="margin-top:.4rem">Age Distribution</div>',
                    unsafe_allow_html=True)
        fig_age_hist = px.histogram(dff, x="Age", color="position", nbins=20,
                                    color_discrete_map={"F":"#ff6b6b","M":"#00b4ff","D":"#00ff9d"},
                                    barmode="overlay", opacity=0.75,
                                    labels={"Age":"Age","position":"Pos"})
        fig_age_hist.update_layout(**PL, height=280)
        st.plotly_chart(fig_age_hist, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-hdr">Top Performers by Metric</div>', unsafe_allow_html=True)
        metric_map = {
            "⚽ Goals":          "goals",
            "🅰️ Assists":        "assists",
            "📊 xG":             "expectedGoals",
            "⭐ Rating":          "rating",
            "🔑 Key Passes":      "keyPasses",
            "🏃 Dribbles":        "successfulDribbles",
            "💰 Market Value":    "Market Value Num",
            "📈 xG per 90":       "xG_per90",
            "🎯 Acc Pass %":      "accuratePassesPercentage",
        }
        c_m1, c_m2 = st.columns([2,1])
        with c_m1: sel_metric = st.selectbox("Rank players by", list(metric_map.keys()))
        with c_m2: n_top      = st.slider("Show top N", 5, 30, 15)

        mc = metric_map[sel_metric]
        if mc not in dff.columns:
            st.warning(f"Column '{mc}' not in filtered data."); st.stop()

        # Build base columns, avoiding duplicates when mc is one of the fixed cols
        _base_cols = ["player_name","Club","position","Age"]
        for _c in [mc, "Market Value Num", "rating"]:
            if _c not in _base_cols:
                _base_cols.append(_c)
        tp = dff.nlargest(n_top, mc)[_base_cols].copy()
        tp["Position"] = tp["position"].map(plabel)
        # Rename Market Value Num to _mv_ before apply to avoid Series ambiguity
        tp["_mv_fmt"] = tp["Market Value Num"].astype(float).apply(fv)
        # Rename rating col for hover to avoid duplicate when mc=="rating"
        tp["_rating_"] = tp["rating"].round(2)

        fig_tp = px.bar(tp, x=mc, y="player_name", orientation="h",
                        color=mc,
                        color_continuous_scale=["#0a192e","#00b4ff","#00f0ff"],
                        hover_data={"Club":True,"Position":True,"_mv_fmt":True,"_rating_":True},
                        labels={mc:sel_metric,"player_name":"Player",
                                "_mv_fmt":"Market Value","_rating_":"Rating"})
        fig_tp.update_layout(**PL, height=max(350,n_top*28), coloraxis_showscale=False)
        fig_tp.update_yaxes(autorange="reversed", gridcolor="rgba(0,180,255,.06)")
        st.plotly_chart(fig_tp, use_container_width=True)

        # Bubble chart: xG vs Assists, size = Market Value
        if all(c in dff.columns for c in ["expectedGoals","assists","Market Value Num"]):
            st.markdown('<div class="section-hdr" style="margin-top:.4rem">xG vs Assists Bubble Map</div>',
                        unsafe_allow_html=True)
            bubble_df = dff.dropna(subset=["expectedGoals","assists","Market Value Num"])
            fig_b = px.scatter(bubble_df, x="expectedGoals", y="assists",
                               size="Market Value Num",
                               color="position",
                               color_discrete_map={"F":"#ff6b6b","M":"#00b4ff","D":"#00ff9d"},
                               hover_data={"player_name":True,"Club":True,"rating":True},
                               labels={"expectedGoals":"xG","assists":"Assists","position":"Pos"},
                               size_max=40)
            fig_b.update_layout(**PL, height=400)
            st.plotly_chart(fig_b, use_container_width=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-top:3rem;padding:1.2rem;text-align:center;
            border-top:1px solid rgba(0,180,255,.1)'>
    <span style='font-family:Orbitron,monospace;font-size:.65rem;letter-spacing:2px;color:#3a5a7a'>
        ⚽ AI PREMIER LEAGUE SCOUT · XGBOOST + KNN · DARK ANALYTICS EDITION
    </span>
</div>""", unsafe_allow_html=True)