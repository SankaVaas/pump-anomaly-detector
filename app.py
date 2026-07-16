"""
app.py — Streamlit frontend for pump vibration anomaly detection.

Calls the FastAPI backend (api/main.py) for all inference.

Start backend first:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Then start frontend:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import requests
import time
import scipy.io

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE    = "http://localhost:8000"
WINDOW_SIZE = 512

st.set_page_config(
    page_title="Pump Anomaly Detector",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.score-card {
    background: #1a1d2e; border: 1px solid #2a2d3e;
    border-radius: 12px; padding: 1.5rem 2rem; text-align: center; margin-bottom: 1rem;
}
.score-label { font-size: 11px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6b7280; margin-bottom: 0.25rem; }
.score-value { font-family: 'JetBrains Mono', monospace; font-size: 2.8rem;
    font-weight: 500; line-height: 1; margin-bottom: 0.5rem; }
.score-normal  { color: #34d399; }
.score-anomaly { color: #f87171; }
.score-unknown { color: #9ca3af; }

.status-badge { display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 600; letter-spacing: 0.06em; }
.badge-normal  { background: #064e3b; color: #34d399; }
.badge-anomaly { background: #7f1d1d; color: #f87171; }
.badge-unknown { background: #1f2937; color: #9ca3af; }

.metric-box { background: #1a1d2e; border: 1px solid #2a2d3e;
    border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; }
.metric-box-label { font-size: 11px; color: #6b7280; margin-bottom: 2px; }
.metric-box-value { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; color: #e5e7eb; }

.alert-info    { background:#1e3a5f; border-left:3px solid #3b82f6; color:#93c5fd;
    border-radius:8px; padding:0.75rem 1rem; font-size:13px; margin-bottom:0.75rem; }
.alert-success { background:#052e16; border-left:3px solid #22c55e; color:#86efac;
    border-radius:8px; padding:0.75rem 1rem; font-size:13px; margin-bottom:0.75rem; }
.alert-danger  { background:#450a0a; border-left:3px solid #ef4444; color:#fca5a5;
    border-radius:8px; padding:0.75rem 1rem; font-size:13px; margin-bottom:0.75rem; }
.alert-warning { background:#451a03; border-left:3px solid #f97316; color:#fdba74;
    border-radius:8px; padding:0.75rem 1rem; font-size:13px; margin-bottom:0.75rem; }

.stTabs [data-baseweb="tab-list"] { gap:4px; background:#1a1d2e; border-radius:8px; padding:4px; }
.stTabs [data-baseweb="tab"] { border-radius:6px; color:#6b7280; font-size:13px; font-weight:500; }
.stTabs [aria-selected="true"] { background:#2a2d3e !important; color:#e5e7eb !important; }
.stTextArea textarea { font-family:'JetBrains Mono',monospace !important; font-size:13px !important;
    background:#1a1d2e !important; border:1px solid #2a2d3e !important; color:#e5e7eb !important; }
.stTextInput input { font-family:'JetBrains Mono',monospace !important;
    background:#1a1d2e !important; border:1px solid #2a2d3e !important; color:#e5e7eb !important; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── API helpers ─────────────────────────────────────────────────────────────────
def api_get(path: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        return {"error": str(e)}

def api_post(path: str, body: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        return {"error": str(e)}

def api_delete(path: str) -> dict | None:
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        return {"error": str(e)}

def parse_floats(text: str) -> np.ndarray:
    text  = text.replace("\n", ",").replace(";", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return np.array([float(p) for p in parts], dtype=np.float32)

def score_card_html(score, threshold, is_anomaly=None, label="Reconstruction error"):
    if score is None:
        cls, badge_cls, badge_text = "score-unknown", "badge-unknown", "NO DATA"
    elif is_anomaly:
        cls, badge_cls, badge_text = "score-anomaly", "badge-anomaly", "ANOMALY DETECTED"
    else:
        cls, badge_cls, badge_text = "score-normal",  "badge-normal",  "NORMAL"
    val = f"{score:.5f}" if score is not None else "—"
    return f"""<div class="score-card">
        <div class="score-label">{label}</div>
        <div class="score-value {cls}">{val}</div>
        <span class="status-badge {badge_cls}">{badge_text}</span>
    </div>"""

@st.cache_data
def demo_normal_signal() -> np.ndarray:
    """
    Download one 512-sample window from the CWRU normal baseline (97.mat).
    This is real bearing vibration data the model was trained on.
    Falls back to synthetic if download fails.
    """
    try:
        import urllib.request, io, tempfile, os
        import scipy.io
        url  = "https://engineering.case.edu/sites/default/files/97.mat"
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            mat = scipy.io.loadmat(tmp.name)
        os.unlink(tmp.name)
        key = next(k for k in mat if "DE_time" in k)
        sig = mat[key].ravel().astype(np.float32)
        return sig[:WINDOW_SIZE]
    except Exception:
        # Fallback: synthetic normal
        t   = np.arange(WINDOW_SIZE) / 12000
        rng = np.random.default_rng(0)
        return (0.8*np.sin(2*np.pi*50*t) + 0.4*np.sin(2*np.pi*150*t)
                + rng.normal(0, 0.03, WINDOW_SIZE)).astype(np.float32)


@st.cache_data
def demo_fault_signal() -> np.ndarray:
    """
    Download one 512-sample window from CWRU ball fault (105.mat).
    This is real bearing fault data the model was evaluated on.
    Falls back to synthetic if download fails.
    """
    try:
        import urllib.request, tempfile, os
        import scipy.io
        url  = "https://engineering.case.edu/sites/default/files/105.mat"
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            mat = scipy.io.loadmat(tmp.name)
        os.unlink(tmp.name)
        key = next(k for k in mat if "DE_time" in k)
        sig = mat[key].ravel().astype(np.float32)
        return sig[:WINDOW_SIZE]
    except Exception:
        # Fallback: synthetic fault
        t   = np.arange(WINDOW_SIZE) / 12000
        rng = np.random.default_rng(1)
        return (0.8*np.sin(2*np.pi*50*t)
                + 3.0*np.sin(2*np.pi*105*t)*0.5*(1+np.sin(2*np.pi*2*t))
                + rng.normal(0, 0.03, WINDOW_SIZE)).astype(np.float32)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛢️ Pump Anomaly Detector")
    st.markdown("---")

    health = api_get("/health")

    if health is None:
        st.markdown('<div class="alert-danger">Backend offline.<br>Run:<br><code>uvicorn api.main:app --reload</code></div>', unsafe_allow_html=True)
        threshold = 0.330
        backend_ok = False
    elif health.get("model") != "loaded":
        st.markdown(f'<div class="alert-warning">Backend up but model not loaded.<br>{health.get("error","")}</div>', unsafe_allow_html=True)
        threshold  = health.get("threshold", 0.330)
        backend_ok = False
    else:
        st.markdown('<div class="alert-success">Backend ✓ &nbsp; Model loaded</div>', unsafe_allow_html=True)
        threshold  = health.get("threshold", 0.330)
        backend_ok = True

    st.markdown("**Anomaly threshold**")
    new_threshold = st.number_input(
        "threshold", min_value=0.001, max_value=10.0,
        value=float(threshold), step=0.001, format="%.4f",
        label_visibility="collapsed",
        help="95th percentile of normal data. Lower = more sensitive.",
    )
    if backend_ok and new_threshold != threshold:
        res = api_post("/threshold", new_threshold)
        if res:
            st.success(f"Threshold updated to {new_threshold:.4f}")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#4b5563;line-height:1.7'>"
        "CWRU bearing dataset<br>"
        "3 fault types × 4 speeds<br>"
        "Mean recall: 0.983 (11 conditions)<br>"
        "Inference: ONNX · CPU only<br>"
        "Window: 512 samples @ 12 kHz"
        "</div>", unsafe_allow_html=True
    )
    if backend_ok:
        st.markdown(f"[API docs →](http://localhost:8000/docs)", unsafe_allow_html=False)

# ── Main ─────────────────────────────────────────────────────────────────────────
st.markdown("### Vibration Anomaly Detection")
if not backend_ok:
    st.markdown(
        '<div class="alert-danger">Backend is offline. Start it with:<br>'
        '<code>uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload</code></div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="alert-info">Enter vibration samples below. Each window is '
        f'512 samples. Reconstruction error above <code>{threshold:.4f}</code> is flagged as an anomaly.</div>',
        unsafe_allow_html=True
    )

tab1, tab2, tab3 = st.tabs(["⚡ Single window", "📈 Live session", "📋 Batch entry"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single window
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Single window scoring")
    st.markdown('<div class="alert-info">Use a demo button to load and score instantly, or paste 512 samples manually.</div>', unsafe_allow_html=True)

    col_input, col_result = st.columns([3, 2])

    with col_input:
        c1, c2 = st.columns(2)
        demo_clicked = None
        if c1.button("Demo — normal signal"):
            demo_clicked = "normal"
            st.session_state["sw_prefill"] = ",".join(
                f"{v:.6f}" for v in demo_normal_signal()
            )
        if c2.button("Demo — fault signal"):
            demo_clicked = "fault"
            st.session_state["sw_prefill"] = ",".join(
                f"{v:.6f}" for v in demo_fault_signal()
            )

        # Always read from sw_prefill so the text area reflects
        # exactly which demo was last loaded
        raw = st.text_area(
            "512 samples (comma-separated)",
            value=st.session_state.get("sw_prefill", ""),
            height=180,
            placeholder="0.012, -0.034, 0.056, ...",
        )

        score_btn = st.button("Score this window", type="primary", key="sw_btn")
        auto_score = demo_clicked is not None

    with col_result:
        if (score_btn or auto_score) and raw.strip():
            try:
                values = parse_floats(raw)
                if len(values) != WINDOW_SIZE:
                    st.error(f"Need exactly {WINDOW_SIZE} values. Got {len(values)}.")
                else:
                    res = api_post("/score/window", {"samples": values.tolist()})
                    if res is None:
                        st.error("Backend offline.")
                    elif "error" in res:
                        st.error(res["error"])
                    else:
                        st.markdown(
                            score_card_html(res["score"], res["threshold"], res["is_anomaly"]),
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="metric-box"><div class="metric-box-label">Threshold</div>'
                            f'<div class="metric-box-value">{res["threshold"]:.4f}</div></div>'
                            f'<div class="metric-box"><div class="metric-box-label">Margin</div>'
                            f'<div class="metric-box-value">{res["margin"]:+.4f}</div></div>'
                            f'<div class="metric-box"><div class="metric-box-label">Inference</div>'
                            f'<div class="metric-box-value">{res["inference_ms"]:.1f} ms</div></div>',
                            unsafe_allow_html=True
                        )
                        # Mini plot
                        import plotly.graph_objects as go
                        color = "#f87171" if res["is_anomaly"] else "#34d399"
                        fig   = go.Figure(go.Scatter(y=values, mode='lines',
                                          line=dict(color=color, width=1)))
                        fig.update_layout(height=150, margin=dict(l=0,r=0,t=8,b=0),
                            paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                            font=dict(color='#6b7280', size=10), showlegend=False,
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=True, gridcolor='#2a2d3e', zeroline=False))
                        st.plotly_chart(fig, use_container_width=True,
                                        config=dict(displayModeBar=False))
            except ValueError as e:
                st.error(f"Parse error: {e}")
        else:
            st.markdown(score_card_html(None, threshold), unsafe_allow_html=True)
            st.caption("Score appears here after submission.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live session
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Live session — timeline")
    st.markdown(
        '<div class="alert-info">Each submission adds one reading to the timeline. '
        'Session is stored in the backend — persists until you clear it.</div>',
        unsafe_allow_html=True
    )

    col_l, col_r = st.columns([3, 2])

    with col_l:
        live_raw = st.text_area(
            "512 samples for this reading",
            height=130,
            placeholder="Paste 512 comma-separated values",
            key="live_raw",
        )
        label = st.text_input("Label (optional)", placeholder="e.g. T+10min", key="live_label")

        c1, c2, c3 = st.columns(3)
        add_btn   = c1.button("Add reading", type="primary", key="live_add")
        demo_btn  = c2.button("Add demo point", key="live_demo")
        clear_btn = c3.button("Clear session",  key="live_clear")

        if clear_btn:
            api_delete("/session")
            st.rerun()

        if demo_btn:
            session_data = api_get("/session")
            n = len(session_data["readings"]) if session_data else 0
            sig = demo_normal_signal() if n % 8 < 5 else demo_fault_signal()
            body = {"samples": sig.tolist(), "label": f"T+{n*5}min"}
            api_post("/session/add", body)
            st.rerun()

        if add_btn and live_raw.strip():
            try:
                values = parse_floats(live_raw)
                if len(values) != WINDOW_SIZE:
                    st.error(f"Need {WINDOW_SIZE} values. Got {len(values)}.")
                else:
                    body = {"samples": values.tolist(),
                            "label": label.strip() or None}
                    res = api_post("/session/add", body)
                    if res is None:
                        st.error("Backend offline.")
                    elif "error" in res:
                        st.error(res["error"])
                    else:
                        st.rerun()
            except ValueError as e:
                st.error(f"Parse error: {e}")

    with col_r:
        session_data = api_get("/session")
        if session_data and session_data.get("readings"):
            latest = session_data["readings"][-1]
            summ   = session_data["summary"]
            st.markdown(
                score_card_html(latest["score"], latest["threshold"],
                                latest["is_anomaly"], "Latest reading"),
                unsafe_allow_html=True
            )
            anomaly_color = "#f87171" if summ["anomalies"] else "#34d399"
            st.markdown(
                f'<div class="metric-box"><div class="metric-box-label">Total readings</div>'
                f'<div class="metric-box-value">{summ["total"]}</div></div>'
                f'<div class="metric-box"><div class="metric-box-label">Anomalies</div>'
                f'<div class="metric-box-value" style="color:{anomaly_color}">'
                f'{summ["anomalies"]}</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(score_card_html(None, threshold, label="Latest reading"),
                        unsafe_allow_html=True)
            st.caption("Add readings to start the timeline.")

    # Timeline chart
    session_data = api_get("/session")
    if session_data and session_data.get("readings"):
        import plotly.graph_objects as go
        readings = session_data["readings"]
        scores   = [r["score"]  for r in readings]
        labels   = [r["label"]  for r in readings]
        colors   = ["#f87171" if r["is_anomaly"] else "#34d399" for r in readings]

        fig = go.Figure()
        fig.add_hline(y=threshold, line_dash="dash", line_color="#6b7280",
                      annotation_text=f"Threshold {threshold:.3f}",
                      annotation_font_color="#6b7280")
        fig.add_trace(go.Scatter(x=labels, y=scores, mode='lines+markers',
                                  line=dict(color='#374151', width=1.5),
                                  marker=dict(color=colors, size=10)))
        fig.update_layout(height=260, margin=dict(l=0,r=0,t=20,b=0),
            paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
            font=dict(color='#6b7280', size=11), showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, tickangle=-30),
            yaxis=dict(title='Reconstruction MSE', showgrid=True,
                       gridcolor='#2a2d3e', zeroline=False))
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Batch entry
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Batch scoring")
    st.markdown(
        f'<div class="alert-info">Paste any number of values. '
        f'The backend splits them into {WINDOW_SIZE}-sample windows automatically. '
        f'Trailing samples that don\'t fill a complete window are discarded.</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([3, 1])
    batch_raw = c1.text_area(
        f"Vibration samples (min {WINDOW_SIZE})",
        height=200,
        placeholder="0.012, -0.034, 0.056, ... (any length)",
        key="batch_raw",
    )
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Load demo\n(5 normal + 3 fault)"):
            t   = np.arange(WINDOW_SIZE * 8) / 12000
            rng = np.random.default_rng(42)
            n_part = (0.8*np.sin(2*np.pi*50*t[:WINDOW_SIZE*5])
                      + rng.normal(0, 0.03, WINDOW_SIZE*5))
            f_part = (0.8*np.sin(2*np.pi*50*t[:WINDOW_SIZE*3])
                      + 3.0*np.sin(2*np.pi*105*t[:WINDOW_SIZE*3])
                      * 0.5*(1+np.sin(2*np.pi*2*t[:WINDOW_SIZE*3]))
                      + rng.normal(0, 0.03, WINDOW_SIZE*3))
            demo = np.concatenate([n_part, f_part])
            st.session_state["batch_demo"] = ",".join(f"{v:.6f}" for v in demo)
            st.rerun()

    if "batch_demo" in st.session_state:
        batch_raw = st.session_state.pop("batch_demo")

    if st.button("Score all windows", type="primary", key="batch_btn") and batch_raw.strip():
        try:
            values = parse_floats(batch_raw)
            if len(values) < WINDOW_SIZE:
                st.error(f"Need at least {WINDOW_SIZE} samples. Got {len(values)}.")
            else:
                with st.spinner("Scoring..."):
                    res = api_post("/score/raw", {"samples": values.tolist()})

                if res is None:
                    st.error("Backend offline.")
                elif "error" in res:
                    st.error(res["error"])
                else:
                    summ = res["summary"]
                    if res["samples_discarded"]:
                        st.warning(f"{res['samples_discarded']} trailing samples discarded (incomplete window).")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total windows", summ["total_windows"])
                    col2.metric("Normal",   summ["normal_windows"])
                    col3.metric("Anomalies", summ["anomaly_windows"],
                                delta=f"{summ['anomaly_rate']*100:.1f}%",
                                delta_color="inverse")
                    col4.metric("Max MSE", f"{summ['max_score']:.4f}")

                    import plotly.graph_objects as go
                    windows = res["windows"]
                    scores  = [w["score"]      for w in windows]
                    colors  = ["#f87171" if w["is_anomaly"] else "#34d399" for w in windows]

                    fig = go.Figure()
                    fig.add_hline(y=threshold, line_dash="dash", line_color="#6b7280",
                                  annotation_text=f"Threshold {threshold:.3f}",
                                  annotation_font_color="#6b7280")
                    fig.add_trace(go.Bar(x=list(range(1, len(windows)+1)),
                                         y=scores, marker_color=colors))
                    fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0),
                        paper_bgcolor='#1a1d2e', plot_bgcolor='#1a1d2e',
                        font=dict(color='#6b7280', size=11), showlegend=False,
                        xaxis=dict(title='Window #', showgrid=False, zeroline=False),
                        yaxis=dict(title='Reconstruction MSE', showgrid=True,
                                   gridcolor='#2a2d3e', zeroline=False))
                    st.plotly_chart(fig, use_container_width=True,
                                    config=dict(displayModeBar=False))

                    flagged = [w for w in windows if w["is_anomaly"]]
                    if flagged:
                        st.markdown("**Flagged windows:**")
                        cols = st.columns(min(4, len(flagged)))
                        for idx, w in enumerate(flagged):
                            cols[idx % 4].markdown(
                                f'<div class="metric-box">'
                                f'<div class="metric-box-label">Window {w["window_index"]+1}</div>'
                                f'<div class="metric-box-value" style="color:#f87171">{w["score"]:.4f}</div>'
                                f'</div>', unsafe_allow_html=True
                            )
                    else:
                        st.markdown('<div class="alert-success">All windows normal — no anomalies detected.</div>',
                                    unsafe_allow_html=True)
        except ValueError as e:
            st.error(f"Parse error: {e}")