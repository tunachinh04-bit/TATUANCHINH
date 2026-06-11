# -*- coding: utf-8 -*-
"""
==============================================================================
AIDEOM-VN  |  AI-Driven Economic Decision Optimization Model for Vietnam
Hệ thống hỗ trợ ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI
------------------------------------------------------------------------------
Bài tập lớn: Các mô hình ra quyết định
Sinh viên : Tạ Tuấn Chinh      |     Mã sinh viên: 23051191
Dữ liệu thực tế Việt Nam 2020-2025 (GSO, World Bank, MoST, MIC, MPI, GII)
------------------------------------------------------------------------------
Dashboard tích hợp 12 bài tối ưu hoá & học tăng cường (Bài 1 -> Bài 12).
Chạy:  streamlit run app.py
Phụ thuộc tối thiểu: streamlit, numpy, pandas, plotly, scipy  (pulp tuỳ chọn)
==============================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linprog, minimize

# ------------------------------------------------------------------ MIP solver
# Ưu tiên PuLP/CBC; nếu không có thì dùng scipy.optimize.milp (đi kèm scipy).
try:
    import pulp
    HAS_PULP = True
except Exception:
    HAS_PULP = False
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
except Exception:
    HAS_MILP = False

# ============================================================================
# CẤU HÌNH TRANG & GIAO DIỆN
# ============================================================================
st.set_page_config(
    page_title="AIDEOM-VN | Tạ Tuấn Chinh",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bảng màu chủ đạo (cảm hứng cờ Việt Nam: đỏ - vàng, nền sáng học thuật)
ACCENT = "#d4001f"     # đỏ
ACCENT2 = "#f4b400"    # vàng
INK = "#11203a"        # xanh mực
PALETTE = ["#d4001f", "#f4b400", "#1565c0", "#2e7d32", "#6a1b9a",
           "#00838f", "#ef6c00", "#5d4037", "#455a64", "#c2185b"]

st.markdown(f"""
<style>
    .stApp {{ background-color: #f6f8fb; }}
    h1, h2, h3, h4 {{ color: {INK}; font-weight: 700; }}
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #11203a 0%, #1c2f50 100%);
    }}
    section[data-testid="stSidebar"] * {{ color: #e8edf5 !important; }}
    section[data-testid="stSidebar"] .stRadio label {{ font-size: 0.92rem; }}
    /* Thẻ số liệu */
    .metric-card {{
        background:#ffffff; border-radius:14px; padding:16px 18px;
        border:1px solid #e6ebf2; box-shadow:0 2px 8px rgba(17,32,58,.05);
    }}
    .metric-card .label {{ color:#64748b; font-size:.80rem; letter-spacing:.04em;
        text-transform:uppercase; margin-bottom:4px; }}
    .metric-card .value {{ color:{INK}; font-size:1.55rem; font-weight:800; }}
    .metric-card .delta {{ font-size:.80rem; }}
    /* Hộp thông tin sinh viên */
    .student-box {{
        background: rgba(255,255,255,.08); border:1px solid rgba(244,180,0,.45);
        border-radius:12px; padding:12px 14px; margin-top:8px;
    }}
    .student-box .nm {{ font-weight:800; color:#f4b400 !important; font-size:1.02rem; }}
    .student-box .row {{ font-size:.85rem; opacity:.95; margin-top:2px; }}
    .badge {{ display:inline-block; background:{ACCENT}; color:#fff !important;
        border-radius:999px; padding:2px 10px; font-size:.72rem; font-weight:700; }}
    .policy {{ background:#fff7ed; border-left:4px solid {ACCENT2};
        border-radius:8px; padding:12px 16px; margin:8px 0; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:4px; }}
    .stTabs [data-baseweb="tab"] {{ background:#eef2f8; border-radius:8px 8px 0 0; }}
    .stTabs [aria-selected="true"] {{ background:{ACCENT}; color:#fff; }}
</style>
""", unsafe_allow_html=True)


def card(label, value, delta=None, color=ACCENT):
    d = f"<div class='delta' style='color:{color}'>{delta}</div>" if delta else ""
    st.markdown(
        f"<div class='metric-card'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div>{d}</div>", unsafe_allow_html=True)


def section(title, sub=""):
    s = f"<span style='color:#64748b;font-weight:400;font-size:.95rem'> — {sub}</span>" if sub else ""
    st.markdown(f"### {title}{s}", unsafe_allow_html=True)


def policy_box(md):
    st.markdown(f"<div class='policy'>{md}</div>", unsafe_allow_html=True)


# ============================================================================
# DỮ LIỆU GỐC (nhúng trực tiếp — đảm bảo chạy độc lập trên Streamlit Cloud)
# Tự động đọc đè từ thư mục data/ nếu tồn tại.
# ============================================================================
@st.cache_data
def load_macro():
    df = pd.DataFrame({
        "year": [2020, 2021, 2022, 2023, 2024, 2025],
        "GDP_trillion_VND": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
        "K_trillion_VND": [16500, 17800, 19600, 21300, 23500, 25900],
        "L_million": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
        "D_digital_pct": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
        "AI_tech_firms_thousand": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
        "H_trained_pct": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
    })
    return df


@st.cache_data
def load_sectors():
    df = pd.DataFrame({
        "sector_id": list(range(1, 11)),
        "sector_name_vi": ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng",
                            "Khai khoáng", "Bán buôn-bán lẻ", "Tài chính-Ngân hàng",
                            "Logistics-Vận tải", "CNTT-Truyền thông", "Giáo dục-Đào tạo", "Y tế"],
        "growth_rate_2024_pct": [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
        "productivity_million_VND_per_worker": [103.4, 241.2, 168.8, 1290.5, 145.3,
                                                1072.4, 321.4, 713.8, 205.7, 437.1],
        "spillover_coef_0_1": [0.35, 0.78, 0.42, 0.30, 0.55, 0.85, 0.72, 0.92, 0.65, 0.60],
        "export_billion_USD": [40.5, 290.9, 2.5, 8.2, 5.5, 1.2, 3.1, 178.0, 0.0, 0.0],
        "labor_million": [13.20, 11.50, 4.80, 0.30, 7.80, 0.55, 1.95, 0.62, 2.15, 0.75],
        "ai_readiness_0_100": [15, 55, 20, 30, 48, 72, 42, 88, 38, 45],
        "automation_risk_pct": [18, 42, 25, 55, 38, 52, 35, 28, 22, 18],
    })
    return df


@st.cache_data
def load_regions():
    df = pd.DataFrame({
        "region_id": list(range(1, 7)),
        "region_name_vi": ["Trung du miền núi phía Bắc", "Đồng bằng sông Hồng",
                            "Bắc Trung Bộ và Duyên hải miền Trung", "Tây Nguyên",
                            "Đông Nam Bộ", "Đồng bằng sông Cửu Long"],
        "grdp_per_capita_million_VND": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
        "fdi_registered_billion_USD": [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
        "digital_index_0_100": [38, 78, 55, 32, 82, 48],
        "ai_readiness_0_100": [22, 68, 40, 18, 75, 30],
        "trained_labor_pct": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
        "rd_intensity_pct": [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
        "internet_penetration_pct": [72, 92, 84, 68, 94, 78],
        "gini_coef": [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
    })
    return df


MACRO = load_macro()
SECTORS = load_sectors()
REGIONS = load_regions()
REGION_SHORT = ["TDMNPB", "ĐBSH", "BTB&DHMT", "TN", "ĐNB", "ĐBSCL"]

# ============================================================================
# SIDEBAR — MỤC LỤC + THÔNG TIN SINH VIÊN
# ============================================================================
st.sidebar.markdown(
    "<div style='font-size:1.35rem;font-weight:900;color:#f4b400 !important'>🇻🇳 AIDEOM-VN</div>"
    "<div style='font-size:.78rem;opacity:.85;margin-bottom:6px'>Vietnam Economic Decision Support System</div>",
    unsafe_allow_html=True)

PAGES = [
    "🏠 Trang chủ",
    "🌱 Bài 1 · Cobb-Douglas mở rộng",
    "💰 Bài 2 · LP ngân sách số",
    "📊 Bài 3 · Chỉ số ưu tiên 10 ngành",
    "🗺️ Bài 4 · LP phân bổ ngành-vùng",
    "🎯 Bài 5 · MIP chọn 15 dự án",
    "🏆 Bài 6 · TOPSIS xếp hạng 6 vùng",
    "🌐 Bài 7 · NSGA-II đa mục tiêu",
    "⏳ Bài 8 · Tối ưu động 2026-2035",
    "👷 Bài 9 · Lao động & AI",
    "🎲 Bài 10 · Quy hoạch ngẫu nhiên 2 GĐ",
    "🤖 Bài 11 · Q-learning RL",
    "🧠 Bài 12 · AIDEOM-VN tích hợp",
]
choice = st.sidebar.radio("MỤC LỤC", PAGES, label_visibility="visible")

st.sidebar.markdown(
    "<div class='student-box'>"
    "<span class='badge'>BÀI TẬP LỚN</span>"
    "<div class='row'>Các mô hình ra quyết định</div>"
    "<div class='nm'>Tạ Tuấn Chinh</div>"
    "<div class='row'>Mã sinh viên: <b>23051191</b></div>"
    "<div class='row' style='margin-top:6px'>📦 GitHub:</div>"
    "<div class='row'><a href='https://github.com/anoreo07/AIDEOMVN' "
    "style='color:#f4b400 !important'>anoreo07/AIDEOMVN</a></div>"
    "</div>", unsafe_allow_html=True)
st.sidebar.caption("Dữ liệu Việt Nam 2020-2025 · GSO · WB · MoST · MIC · MPI · GII")


# ============================================================================
# TRANG CHỦ — tổng hợp dữ liệu gốc & nội dung bài tập
# ============================================================================
def render_home():
    st.markdown(
        f"<h1 style='margin-bottom:0'>AIDEOM-VN <span style='color:{ACCENT}'>·</span> "
        "Hệ thống hỗ trợ ra quyết định kinh tế số</h1>"
        "<p style='color:#475569;font-size:1.05rem;margin-top:4px'>"
        "Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI — "
        "tích hợp 12 bài tối ưu hoá, MCDM, đa mục tiêu, ngẫu nhiên và học tăng cường.</p>",
        unsafe_allow_html=True)

    g25 = MACRO.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("GDP 2025 (ngh.tỷ VND)", "12.847,6", "▲ 8,02% so với 2024", "#2e7d32")
    with c2: card("GDP/người 2025 (USD)", "5.026", "▲ từ 4.700 (2024)", "#2e7d32")
    with c3: card("Kinh tế số / GDP", "≈19,5%", "Mục tiêu 2030: 30%", ACCENT)
    with c4: card("Đóng góp KH-CN/GDP", "2,49%", "1,68% trực tiếp + 0,81% lan toả", INK)

    st.divider()
    section("📚 Bản đồ 12 bài tập theo 4 cấp độ")
    tiers = pd.DataFrame({
        "Cấp độ": ["DỄ", "DỄ", "DỄ", "TRUNG BÌNH", "TRUNG BÌNH", "TRUNG BÌNH",
                   "KHÁ KHÓ", "KHÁ KHÓ", "KHÁ KHÓ", "KHÓ", "KHÓ", "KHÓ"],
        "Bài": [f"Bài {i}" for i in range(1, 13)],
        "Nội dung": [
            "Hàm sản xuất Cobb-Douglas mở rộng + AI/số hoá, phân rã tăng trưởng",
            "LP phân bổ ngân sách số 4 hạng mục + giá đối ngẫu",
            "Chỉ số ưu tiên Priorityᵢ cho 10 ngành (chuẩn hoá min-max)",
            "LP phân bổ ngân sách ngành-vùng + ràng buộc công bằng vùng",
            "MIP 0-1 lựa chọn 15 dự án chuyển đổi số (knapsack tổng quát)",
            "TOPSIS + Entropy xếp hạng 6 vùng theo sẵn sàng AI",
            "NSGA-II tối ưu Pareto 4 mục tiêu (tăng trưởng/bao trùm/môi trường/an ninh)",
            "Tối ưu động liên thời gian 2026-2035 (Cobb-Douglas + tích luỹ vốn)",
            "Mô phỏng tác động AI tới thị trường lao động (NetJob ròng)",
            "Quy hoạch ngẫu nhiên 2 giai đoạn (VSS & EVPI)",
            "Q-learning / DQN cho chính sách kinh tế thích nghi (MDP)",
            "Tích hợp 6 module + dashboard 5 kịch bản chính sách",
        ],
        "Công cụ": ["numpy", "scipy/pulp", "numpy", "scipy/pulp", "pulp/milp",
                    "numpy", "evolutionary", "scipy", "scipy/pulp",
                    "scipy (SP)", "Q-learning", "tích hợp"],
    })
    st.dataframe(tiers, use_container_width=True, hide_index=True, height=460)

    st.divider()
    section("🗃️ Dữ liệu gốc Việt Nam 2020-2025", "3 bộ dữ liệu thực tế")
    t1, t2, t3 = st.tabs(["📈 Vĩ mô 2020-2025", "🏭 10 ngành 2024", "🗺️ 6 vùng 2024"])
    with t1:
        st.dataframe(MACRO, use_container_width=True, hide_index=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=MACRO.year, y=MACRO.GDP_trillion_VND, name="GDP (ngh.tỷ VND)",
                    marker_color=ACCENT, opacity=.85)
        fig.add_trace(go.Scatter(x=MACRO.year, y=MACRO.D_digital_pct, name="Kinh tế số/GDP (%)",
                      mode="lines+markers", line=dict(color=ACCENT2, width=3)), secondary_y=True)
        fig.update_layout(height=360, template="plotly_white",
                          title="GDP và tỷ trọng kinh tế số")
        fig.update_yaxes(title_text="GDP (ngh.tỷ VND)", secondary_y=False)
        fig.update_yaxes(title_text="Kinh tế số/GDP (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        st.dataframe(SECTORS, use_container_width=True, hide_index=True)
        fig = px.scatter(SECTORS, x="ai_readiness_0_100", y="growth_rate_2024_pct",
                         size="labor_million", color="spillover_coef_0_1",
                         hover_name="sector_name_vi", color_continuous_scale="RdYlGn",
                         labels={"ai_readiness_0_100": "AI Readiness",
                                 "growth_rate_2024_pct": "Tăng trưởng 2024 (%)"})
        fig.update_layout(height=380, template="plotly_white",
                          title="10 ngành: AI readiness × Tăng trưởng (kích thước = lao động)")
        st.plotly_chart(fig, use_container_width=True)
    with t3:
        st.dataframe(REGIONS, use_container_width=True, hide_index=True)
        fig = px.bar(REGIONS, x="region_name_vi", y="ai_readiness_0_100",
                     color="digital_index_0_100", color_continuous_scale="Reds",
                     labels={"region_name_vi": "", "ai_readiness_0_100": "AI Readiness"})
        fig.update_layout(height=360, template="plotly_white",
                          title="Mức sẵn sàng AI của 6 vùng kinh tế-xã hội")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    section("🏛️ Khung chính sách tham chiếu")
    policy_box(
        "<b>Nghị quyết 57-NQ/TW</b> (2024) — đột phá phát triển KHCN, ĐMST & chuyển đổi số · "
        "<b>QĐ 749/QĐ-TTg</b> — Chương trình Chuyển đổi số quốc gia · "
        "<b>QĐ 127/QĐ-TTg</b> — Chiến lược quốc gia về AI đến 2030 · "
        "<b>QĐ 411/QĐ-TTg</b> — Chiến lược kinh tế số & xã hội số · "
        "<b>Cam kết COP26</b> — phát thải ròng bằng 0 vào 2050.")
    st.caption("👉 Chọn từng bài ở **MỤC LỤC** bên trái. Mỗi bài gồm: Bối cảnh · Mô hình "
               "toán học · Kết quả (tương tác) · Thảo luận chính sách.")


# ============================================================================
# BÀI 1 — HÀM SẢN XUẤT COBB-DOUGLAS MỞ RỘNG
# ============================================================================
@st.cache_data
def cobb_compute(a, b, g, d, th):
    K = MACRO.K_trillion_VND.values.astype(float)
    L = MACRO.L_million.values.astype(float)
    D = MACRO.D_digital_pct.values.astype(float)
    AI = MACRO.AI_tech_firms_thousand.values.astype(float)
    H = MACRO.H_trained_pct.values.astype(float)
    Y = MACRO.GDP_trillion_VND.values.astype(float)
    core = K**a * L**b * D**g * AI**d * H**th
    A = Y / core
    Ybar = A.mean()
    Yhat = Ybar * core
    mape = float(np.mean(np.abs((Y - Yhat) / Y)) * 100)
    return K, L, D, AI, H, Y, A, Yhat, mape


def render_bai1():
    st.title("🌱 Bài 1 · Hàm sản xuất Cobb-Douglas mở rộng với AI và số hoá")
    policy_box("<b>Bối cảnh:</b> GDP 2024 đạt 11.511,9 ngh.tỷ VND (+7,09%); năng suất lao động "
               "245 tr.VND/người (2025); KH-CN đóng góp 2,49% GDP. Câu hỏi: mô hình hoá nền kinh tế "
               "bằng Cobb-Douglas mở rộng có thêm số hoá D, năng lực AI và vốn nhân lực H — yếu tố "
               "nào đóng góp lớn nhất cho tăng trưởng?")
    st.latex(r"Y_t = A_t\,K_t^{\alpha} L_t^{\beta} D_t^{\gamma} AI_t^{\delta} H_t^{\theta},"
             r"\quad \alpha+\beta+\gamma+\delta+\theta = 1")

    st.sidebar.markdown("**⚙️ Hệ số co giãn (Bài 1)**")
    a = st.sidebar.slider("α — vốn vật chất K", 0.0, 0.6, 0.33, 0.01)
    b = st.sidebar.slider("β — lao động L", 0.0, 0.6, 0.42, 0.01)
    g = st.sidebar.slider("γ — số hoá D", 0.0, 0.3, 0.10, 0.01)
    d = st.sidebar.slider("δ — năng lực AI", 0.0, 0.3, 0.08, 0.01)
    th = st.sidebar.slider("θ — nhân lực số H", 0.0, 0.3, 0.07, 0.01)
    tot = a + b + g + d + th
    if abs(tot - 1) > 1e-9:
        st.sidebar.warning(f"Σ hệ số = {tot:.2f} ≠ 1 (CRS bị vi phạm)")

    K, L, D, AI, H, Y, A, Yhat, mape = cobb_compute(a, b, g, d, th)
    yrs = MACRO.year.values

    c1, c2, c3 = st.columns(3)
    with c1: card("TFP 2025 (A₂₀₂₅)", f"{A[-1]:.2f}", f"▲ từ {A[0]:.2f} (2020)", "#2e7d32")
    with c2: card("MAPE dự báo (Ā)", f"{mape:.2f}%", "Sai số tuyệt đối TB", ACCENT)
    with c3: card("Tăng GDP TB/năm", f"{(np.log(Y[-1]/Y[0])/5)*100:.2f}%", "CAGR 2020-2025", INK)

    section("1.4.1 & 1.4.2 · TFP và dự báo Ŷ vs thực tế")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(x=yrs, y=A, markers=True, labels={"x": "Năm", "y": "TFP Aₜ"})
        fig.update_traces(line_color=ACCENT, line_width=3)
        fig.update_layout(height=320, template="plotly_white", title="Năng suất nhân tố tổng hợp Aₜ")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_bar(x=yrs, y=Y, name="Y thực tế", marker_color=INK, opacity=.8)
        fig.add_trace(go.Scatter(x=yrs, y=Yhat, name="Ŷ dự báo (Ā)",
                      mode="lines+markers", line=dict(color=ACCENT2, width=3)))
        fig.update_layout(height=320, template="plotly_white", title="Ŷ dự báo vs Y thực tế")
        st.plotly_chart(fig, use_container_width=True)

    section("1.4.3 · Phân rã tăng trưởng (Δln) 2020-2025")
    dln = lambda x: np.log(x[-1] / x[0]) / (len(x) - 1)
    contrib = {"TFP (Aₜ)": dln(A), "Vốn K": a * dln(K), "Lao động L": b * dln(L),
               "Số hoá D": g * dln(D), "Năng lực AI": d * dln(AI), "Nhân lực H": th * dln(H)}
    gtot = sum(contrib.values())
    dfc = pd.DataFrame({"Yếu tố": list(contrib),
                        "Đóng góp (đ.vị/năm)": [round(v, 4) for v in contrib.values()],
                        "Tỷ trọng (%)": [round(v / gtot * 100, 1) for v in contrib.values()]})
    cc1, cc2 = st.columns([1.1, 1])
    with cc1:
        st.dataframe(dfc, use_container_width=True, hide_index=True)
    with cc2:
        fig = px.bar(dfc, x="Yếu tố", y="Tỷ trọng (%)", color="Yếu tố",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(height=320, template="plotly_white", showlegend=False,
                          title="Tỷ trọng đóng góp tăng trưởng")
        st.plotly_chart(fig, use_container_width=True)

    section("1.4.4 · Kịch bản dự báo GDP 2030")
    s1, s2, s3 = st.columns(3)
    D30 = s1.slider("D 2030 (% KTS/GDP)", 19.5, 40.0, 30.0, 0.5)
    AI30 = s2.slider("AI 2030 (ngh. DN số)", 80.0, 150.0, 100.0, 1.0)
    H30 = s3.slider("H 2030 (% LĐ đào tạo)", 29.2, 45.0, 35.0, 0.5)
    gKL = st.slider("Tăng trưởng K, L (%/năm)", 0.0, 10.0, 6.0, 0.5)
    gtfp = st.slider("Tăng TFP (%/năm)", 0.0, 3.0, 1.2, 0.1)
    K30 = K[-1] * (1 + gKL / 100) ** 5
    L30 = L[-1] * (1 + gKL / 100) ** 5
    A30 = A[-1] * (1 + gtfp / 100) ** 5
    Y30 = A30 * K30**a * L30**b * D30**g * AI30**d * H30**th
    cagr30 = (np.log(Y30 / Y[-1]) / 5) * 100
    r1, r2 = st.columns(2)
    with r1: card("GDP dự báo 2030 (ngh.tỷ VND)", f"{Y30:,.0f}", f"CAGR {cagr30:.2f}%/năm", "#2e7d32")
    with r2: card("Quy đổi (tỷ USD, ~25.500 VND/USD)", f"{Y30*1000/25.5:,.0f}", "tham khảo", INK)

    policy_box("<b>Thảo luận:</b> (a) TFP tăng đều 2020-2025 → tăng trưởng dần dựa vào năng suất, "
               "không chỉ tích luỹ vốn — dấu hiệu tích cực về chất lượng tăng trưởng. "
               "(b) Trong D/AI/H, <b>số hoá D</b> thường đóng góp lớn nhất do tăng nhanh nhất giai đoạn này. "
               "(c) Mục tiêu 30% KTS/GDP 2030 khả thi nhưng cần ràng buộc đầu tư H đi kèm để hấp thụ AI.")


# ============================================================================
# BÀI 2 — LP PHÂN BỔ NGÂN SÁCH 4 HẠNG MỤC ĐẦU TƯ SỐ
# ============================================================================
def solve_bai2(budget=100.0, x3_min=20.0, tech_share=0.35,
               coef=(0.85, 1.20, 0.95, 1.35), mins=(25, 15, 20, 10)):
    c = [-v for v in coef]
    A_ub = [[1, 1, 1, 1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1],
            [tech_share, -(1 - tech_share), tech_share, -(1 - tech_share)]]
    b_ub = [budget, -mins[0], -mins[1], -x3_min, -mins[3], 0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")
    return res


def render_bai2():
    st.title("💰 Bài 2 · LP phân bổ ngân sách số (4 hạng mục)")
    policy_box("<b>Bối cảnh (QĐ 749/QĐ-TTg):</b> phân bổ 100.000 tỷ VND năm 2026 cho hạ tầng số (x₁), "
               "AI & dữ liệu (x₂), nhân lực số (x₃), R&D công nghệ (x₄) nhằm tối đa hoá tăng GDP kỳ vọng.")
    st.latex(r"\max Z = 0{,}85x_1 + 1{,}20x_2 + 0{,}95x_3 + 1{,}35x_4")
    st.caption("s.t. Σx ≤ B · x₁≥25 · x₂≥15 · x₃≥20 · x₄≥10 · x₂+x₄ ≥ 35%·Σx")

    st.sidebar.markdown("**⚙️ Tham số (Bài 2)**")
    budget = st.sidebar.slider("Ngân sách tổng B (ngh.tỷ)", 100, 160, 100, 5)
    x3min = st.sidebar.slider("Sàn nhân lực số x₃ ≥", 20, 40, 20, 1)
    tech = st.sidebar.slider("Tỷ trọng công nghệ chiến lược", 0.25, 0.55, 0.35, 0.05)

    res = solve_bai2(budget, x3min, tech)
    labels = ["x₁ Hạ tầng số", "x₂ AI & dữ liệu", "x₃ Nhân lực số", "x₄ R&D"]
    if not res.success:
        st.error("❌ Bài toán KHÔNG khả thi với cấu hình hiện tại (vd: x₃ quá cao so với ngân sách).")
        return
    x = res.x
    c1, c2, c3 = st.columns(3)
    with c1: card("Z* (tăng GDP kỳ vọng)", f"{-res.fun:.2f}", "ngh.tỷ VND", "#2e7d32")
    with c2: card("Hiệu suất Z*/B", f"{-res.fun/budget:.3f}", "đồng GDP/đồng vốn", INK)
    with c3: card("Phần dành R&D x₄", f"{x[3]:.1f}", f"{x[3]/budget*100:.0f}% ngân sách", ACCENT)

    col1, col2 = st.columns([1, 1])
    with col1:
        dfx = pd.DataFrame({"Hạng mục": labels, "Phân bổ (ngh.tỷ)": np.round(x, 2)})
        st.dataframe(dfx, use_container_width=True, hide_index=True)
        fig = px.pie(dfx, names="Hạng mục", values="Phân bổ (ngh.tỷ)", hole=.45,
                     color_discrete_sequence=PALETTE)
        fig.update_layout(height=300, template="plotly_white", title="Cơ cấu phân bổ tối ưu")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        section("2.4.2 · Giá đối ngẫu (shadow price)")
        try:
            shadow = res.ineqlin.marginals
            sp_budget = -shadow[0]
        except Exception:
            sp_budget = None
        if sp_budget is not None:
            st.metric("Shadow price ràng buộc ngân sách", f"{sp_budget:.3f}",
                      help="Tăng 1 ngh.tỷ ngân sách → Z* tăng thêm bấy nhiêu")
            st.info(f"💡 Mỗi **1 tỷ VND** ngân sách tăng thêm → GDP kỳ vọng tăng ~**{sp_budget:.2f} tỷ VND**. "
                    "Đây là cận trên hợp lý của chi phí cơ hội của vốn công.")
        section("2.4.3 · Độ nhạy Z*(B)")
        Bs = np.arange(100, 161, 5)
        Zs = [(-solve_bai2(B, x3min, tech).fun) if solve_bai2(B, x3min, tech).success else np.nan
              for B in Bs]
        fig = px.line(x=Bs, y=Zs, markers=True, labels={"x": "Ngân sách B", "y": "Z*"})
        fig.update_traces(line_color=ACCENT, line_width=3)
        fig.update_layout(height=240, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    section("2.4.4 · Ưu tiên nhân lực số: x₃ ≥ 30")
    r30 = solve_bai2(budget, 30, tech)
    if r30.success:
        st.success(f"✅ Vẫn khả thi. Z* = **{-r30.fun:.2f}** "
                   f"(thay đổi {(-r30.fun)-(-solve_bai2(budget,20,tech).fun):+.2f} so với x₃≥20). "
                   "Ưu tiên nhân lực số làm giảm nhẹ Z* do đẩy vốn khỏi R&D hệ số cao.")
    else:
        st.error("❌ Không khả thi khi x₃ ≥ 30 với ngân sách hiện tại.")

    policy_box("<b>Thảo luận:</b> R&D (x₄) hệ số cao nhất (1,35) nhờ lan toả dài hạn nhưng sàn tối thiểu "
               "thấp nhất (10) — phản ánh rủi ro và độ trễ; trong thực tế cần cân nhắc năng lực hấp thụ "
               "và ưu tiên hạ tầng/an sinh của ngân sách nhà nước.")


# ============================================================================
# BÀI 3 — CHỈ SỐ ƯU TIÊN NGÀNH Priorityᵢ
# ============================================================================
def priority_scores(weights, w_risk):
    good = SECTORS[["growth_rate_2024_pct", "productivity_million_VND_per_worker",
                    "spillover_coef_0_1", "export_billion_USD", "labor_million",
                    "ai_readiness_0_100"]].values.astype(float)
    bad = SECTORS["automation_risk_pct"].values.astype(float)
    Xg = (good - good.min(0)) / (good.max(0) - good.min(0))
    Xb = (bad.max() - bad) / (bad.max() - bad.min())
    return Xg @ np.array(weights) + w_risk * Xb, Xg, Xb


def render_bai3():
    st.title("📊 Bài 3 · Chỉ số ưu tiên ngành (10 ngành)")
    policy_box("<b>Bối cảnh:</b> ngành nào nên đẩy mạnh chuyển đổi số & AI trước để tạo hiệu ứng lan toả "
               "tối đa? Xây dựng chỉ số ưu tiên định lượng từ 7 tiêu chí (chuẩn hoá min-max, đảo dấu Rủi ro).")
    st.latex(r"Priority_i = a_1 G + a_2 P + a_3 Spill + a_4 Exp + a_5 Emp + a_6 AI - a_7 Risk")

    st.sidebar.markdown("**⚙️ Trọng số (Bài 3)**")
    a1 = st.sidebar.slider("a₁ Tăng trưởng", 0.0, 0.4, 0.15, 0.01)
    a2 = st.sidebar.slider("a₂ Năng suất", 0.0, 0.4, 0.15, 0.01)
    a3 = st.sidebar.slider("a₃ Lan toả", 0.0, 0.4, 0.20, 0.01)
    a4 = st.sidebar.slider("a₄ Xuất khẩu", 0.0, 0.4, 0.15, 0.01)
    a5 = st.sidebar.slider("a₅ Việc làm", 0.0, 0.4, 0.10, 0.01)
    a6 = st.sidebar.slider("a₆ AI Readiness", 0.0, 0.4, 0.20, 0.01)
    a7 = st.sidebar.slider("a₇ Rủi ro (đảo dấu)", 0.0, 0.4, 0.15, 0.01)

    scores, Xg, Xb = priority_scores([a1, a2, a3, a4, a5, a6], a7)
    df = SECTORS[["sector_name_vi"]].copy()
    df["Priority"] = scores
    df = df.sort_values("Priority", ascending=False).reset_index(drop=True)
    df.index += 1

    section("3.4.2 · Xếp hạng 10 ngành theo Priority")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.dataframe(df.rename_axis("Hạng").reset_index(), use_container_width=True, hide_index=True)
    with c2:
        fig = px.bar(df.sort_values("Priority"), x="Priority", y="sector_name_vi",
                     orientation="h", color="Priority", color_continuous_scale="RdYlGn")
        fig.update_layout(height=400, template="plotly_white", yaxis_title="",
                          title="Chỉ số ưu tiên ngành")
        st.plotly_chart(fig, use_container_width=True)

    top3 = df.head(3)["sector_name_vi"].tolist()
    st.success(f"🏆 **Top-3 ưu tiên:** {', '.join(top3)}")

    section("3.4.3 · Độ nhạy theo trọng số AI Readiness (a₆)")
    a6_range = np.arange(0.05, 0.41, 0.05)
    heat = []
    for a6v in a6_range:
        w = np.array([a1, a2, a3, a4, a5, a6v]); w = w / w.sum() * (1 - a7)
        sc, _, _ = priority_scores(w, a7)
        heat.append(sc)
    heat = np.array(heat)
    fig = px.imshow(heat.T, x=[f"{v:.2f}" for v in a6_range],
                    y=SECTORS.sector_name_vi, color_continuous_scale="Reds",
                    labels={"x": "Trọng số a₆ (AI)", "y": "", "color": "Priority"}, aspect="auto")
    fig.update_layout(height=380, template="plotly_white", title="Heatmap độ nhạy Priority theo a₆")
    st.plotly_chart(fig, use_container_width=True)

    section("3.4.4 · So sánh 'Định hướng tăng trưởng' vs 'Bao trùm'")
    sc_g, _, _ = priority_scores([0.25, 0.20, 0.15, 0.20, 0.05, 0.10], 0.05)
    sc_i, _, _ = priority_scores([0.10, 0.10, 0.25, 0.05, 0.25, 0.10], 0.15)
    cmp = pd.DataFrame({"Ngành": SECTORS.sector_name_vi,
                        "Tăng trưởng": sc_g, "Bao trùm": sc_i})
    cmp["Top tăng trưởng"] = cmp["Tăng trưởng"].rank(ascending=False).astype(int)
    cmp["Top bao trùm"] = cmp["Bao trùm"].rank(ascending=False).astype(int)
    g3 = cmp.nsmallest(3, "Top tăng trưởng")["Ngành"].tolist()
    i3 = cmp.nsmallest(3, "Top bao trùm")["Ngành"].tolist()
    cA, cB = st.columns(2)
    cA.info("**Top-3 Tăng trưởng:** " + ", ".join(g3))
    cB.info("**Top-3 Bao trùm:** " + ", ".join(i3))

    policy_box("<b>Thảo luận:</b> Khai khoáng có năng suất rất cao (1.290 tr.VND/LĐ) nhưng tăng trưởng âm, "
               "lan toả thấp và rủi ro tự động hoá cao → không vào nhóm ưu tiên. Kết quả Top-3 (CNTT, "
               "Tài chính-NH, CN chế biến) phù hợp tinh thần <b>Nghị quyết 57-NQ/TW</b> ưu tiên ngành "
               "có sức lan toả số cao.")


# ============================================================================
# BÀI 4 — LP PHÂN BỔ NGÂN SÁCH NGÀNH-VÙNG (công bằng vùng)
# ============================================================================
BETA4 = np.array([  # 6 vùng x 4 hạng mục [I, D, AI, H]
    [1.15, 0.85, 0.55, 1.30], [0.95, 1.25, 1.40, 1.05], [1.05, 0.95, 0.85, 1.15],
    [1.20, 0.75, 0.45, 1.35], [0.90, 1.30, 1.55, 1.00], [1.10, 0.85, 0.65, 1.25]])
D0 = np.array([38, 78, 55, 32, 82, 48], float)


def solve_bai4(equity=True, lam=0.70, total=50000, floor=5000, cap=12000,
               h_floor=12000, gamma=0.002):
    # 24 biến x[r,j] flatten + 1 biến phụ M (Dmax). thứ tự: 24 x + M
    nv = 25
    c = np.zeros(nv); c[:24] = -BETA4.flatten()
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    # C1 tổng ngân sách
    row = np.zeros(nv); row[:24] = 1; A_ub.append(row); b_ub.append(total)
    for r in range(6):
        # C2 sàn vùng (>= floor -> -sum <= -floor)
        row = np.zeros(nv); row[r*4:r*4+4] = -1; A_ub.append(row); b_ub.append(-floor)
        # C3 trần vùng
        row = np.zeros(nv); row[r*4:r*4+4] = 1; A_ub.append(row); b_ub.append(cap)
    # C4 sàn nhân lực H (cột index 3 mỗi vùng)
    row = np.zeros(nv)
    for r in range(6): row[r*4+3] = -1
    A_ub.append(row); b_ub.append(-h_floor)
    if equity:
        for r in range(6):
            # D0r + gamma*x_D,r <= M
            row = np.zeros(nv); row[r*4+1] = gamma; row[24] = -1
            A_ub.append(row); b_ub.append(-D0[r])
            # D0r + gamma*x_D,r >= lam*M  -> -gamma*x_D,r + lam*M <= D0r
            row = np.zeros(nv); row[r*4+1] = -gamma; row[24] = lam
            A_ub.append(row); b_ub.append(D0[r])
    bounds = [(0, None)] * 24 + [(0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res


def render_bai4():
    st.title("🗺️ Bài 4 · LP phân bổ ngân sách số ngành-vùng")
    policy_box("<b>Bối cảnh (QĐ 411/QĐ-TTg):</b> phân bổ 50.000 tỷ VND cho 6 vùng × 4 hạng mục "
               "(I-hạ tầng, D-CĐS DN, AI, H-nhân lực) tối đa hoá GDP gain nhưng bảo đảm công bằng vùng.")
    st.latex(r"\max Z=\sum_r\sum_j \beta_{j,r}x_{j,r}\;;\;\; "
             r"D_r+\gamma x_{D,r}\ge \lambda\max_r(D_r+\gamma x_{D,r})")

    st.sidebar.markdown("**⚙️ Tham số (Bài 4)**")
    lam = st.sidebar.slider("λ — ngưỡng công bằng C5", 0.0, 0.9, 0.70, 0.05)
    cap = st.sidebar.slider("Trần ngân sách/vùng (ngh.tỷ)", 9000, 15000, 12000, 500)

    res = solve_bai4(True, lam, cap=cap)
    res_no = solve_bai4(False, cap=cap)
    if not res.success:
        st.error("❌ Không khả thi — thử giảm λ hoặc tăng trần vùng. "
                 f"(Lưu ý: với λ≈0,70 và D thấp ở Tây Nguyên/Bắc, ràng buộc công bằng có thể vô nghiệm; "
                 "đây là 'bẫy khả thi' có chủ đích trong đề bài.)")
        return
    X = res.x[:24].reshape(6, 4)
    items = ["I", "D", "AI", "H"]
    c1, c2, c3 = st.columns(3)
    with c1: card("Z* (có công bằng)", f"{-res.fun:,.0f}", "tỷ VND GDP gain", "#2e7d32")
    with c2: card("Z* (bỏ công bằng)", f"{-res_no.fun:,.0f}", "tỷ VND", INK)
    with c3:
        cost = (-res_no.fun) - (-res.fun)
        card("Chi phí công bằng vùng", f"{cost:,.0f}", f"{cost/(-res_no.fun)*100:.2f}% Z*", ACCENT)

    section("4.4.1 & 4.4.3 · Ma trận phân bổ tối ưu (heatmap)")
    dfX = pd.DataFrame(X, columns=items, index=REGION_SHORT)
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.dataframe(dfX.style.format("{:,.0f}"), use_container_width=True)
    with c2:
        fig = px.imshow(X, x=items, y=REGION_SHORT, color_continuous_scale="Reds",
                        text_auto=".0f", aspect="auto",
                        labels={"color": "ngh.tỷ VND"})
        fig.update_layout(height=360, template="plotly_white", title="Phân bổ x[vùng, hạng mục]")
        st.plotly_chart(fig, use_container_width=True)

    region_total = X.sum(1)
    top_r = REGIONS.region_name_vi.values[np.argmax(region_total)]
    st.success(f"🏆 Vùng nhận nhiều ngân sách nhất: **{top_r}** ({region_total.max():,.0f} ngh.tỷ). "
               f"Hạng mục nổi bật toàn quốc: **{items[int(np.argmax(X.sum(0)))]}**.")

    section("4.4.4 · So sánh có / không ràng buộc công bằng")
    Xno = res_no.x[:24].reshape(6, 4)
    cmp = pd.DataFrame({"Vùng": REGION_SHORT,
                        "Có công bằng": X.sum(1), "Bỏ công bằng": Xno.sum(1)})
    fig = px.bar(cmp, x="Vùng", y=["Có công bằng", "Bỏ công bằng"], barmode="group",
                 color_discrete_sequence=[ACCENT, "#94a3b8"])
    fig.update_layout(height=320, template="plotly_white", title="Tổng ngân sách mỗi vùng")
    st.plotly_chart(fig, use_container_width=True)

    policy_box("<b>Thảo luận:</b> Bỏ ràng buộc công bằng, vốn dồn về ĐBSH/ĐNB (hệ số AI cao) → "
               "khoảng cách vùng giãn rộng. Trần ngân sách/vùng (C3) đóng vai 'phân quyền', làm giảm Z* "
               "một phần đổi lấy cân bằng. Tây Nguyên hệ số AI thấp (0,45) → mô hình ưu tiên H và I trước.")


# ============================================================================
# BÀI 5 — MIP LỰA CHỌN 15 DỰ ÁN CHUYỂN ĐỔI SỐ
# ============================================================================
PROJ = pd.DataFrame({
    "Mã": [f"P{i}" for i in range(1, 16)],
    "Tên": ["TT dữ liệu QG Hoà Lạc", "TT dữ liệu QG phía Nam", "5G toàn quốc",
            "VNeID 2.0", "Cổng DVC QG v3", "Y tế số QG", "Giáo dục số K-12",
            "TT AI QG + supercomputing", "Sandbox fintech", "Logistics thông minh",
            "Nông nghiệp số ĐBSCL", "Đào tạo 50.000 kỹ sư AI", "KCN bán dẫn BN-BG",
            "An ninh mạng QG (SOC)", "Open Data QG"],
    "C": [12000, 11500, 18000, 4500, 3200, 5800, 6500, 15000, 2500, 7200, 4800, 8500, 20000, 3800, 1500],
    "C1": [8500, 7500, 12000, 3500, 2500, 4000, 4500, 9000, 1800, 5000, 3500, 5500, 13000, 2800, 1200],
    "B": [21500, 20800, 32500, 9200, 6800, 11400, 12200, 28500, 5800, 13800, 8500, 16200, 35000, 7500, 3800],
})


def solve_bai5(total=80000, y1y2=80000, force_p1p2=False):
    C = PROJ.C.values.astype(float); C1 = PROJ.C1.values.astype(float); B = PROJ.B.values.astype(float)
    n = 15
    if HAS_PULP:
        m = pulp.LpProblem("sel", pulp.LpMaximize)
        y = pulp.LpVariable.dicts("y", range(n), cat="Binary")
        m += pulp.lpSum(B[i] * y[i] for i in range(n))
        m += pulp.lpSum(C[i] * y[i] for i in range(n)) <= total
        m += pulp.lpSum(C1[i] * y[i] for i in range(n)) <= 40000
        m += y[0] + y[1] <= 1
        m += y[7] <= y[11]; m += y[12] <= y[11]
        m += y[3] + y[4] >= 1; m += y[13] >= 1
        m += pulp.lpSum(y[i] for i in range(n)) >= 7
        m += pulp.lpSum(y[i] for i in range(n)) <= 11
        if force_p1p2: m += y[0] == 1; m += y[1] == 1
        st_ = m.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[m.status] != "Optimal":
            return None, None
        xv = np.array([y[i].value() for i in range(n)])
        return xv, pulp.value(m.objective)
    # fallback scipy.milp
    cons = [LinearConstraint(C, -np.inf, total), LinearConstraint(C1, -np.inf, 40000)]
    a = np.zeros(n); a[0] = a[1] = 1; cons.append(LinearConstraint(a, -np.inf, 1))
    a = np.zeros(n); a[7] = 1; a[11] = -1; cons.append(LinearConstraint(a, -np.inf, 0))
    a = np.zeros(n); a[12] = 1; a[11] = -1; cons.append(LinearConstraint(a, -np.inf, 0))
    a = np.zeros(n); a[3] = a[4] = 1; cons.append(LinearConstraint(a, 1, np.inf))
    a = np.zeros(n); a[13] = 1; cons.append(LinearConstraint(a, 1, np.inf))
    cons.append(LinearConstraint(np.ones(n), 7, 11))
    lb = np.zeros(n); ub = np.ones(n)
    if force_p1p2: lb[0] = lb[1] = 1
    res = milp(c=-B, constraints=cons, integrality=np.ones(n), bounds=Bounds(lb, ub))
    if not res.success: return None, None
    return np.round(res.x), -res.fun


def render_bai5():
    st.title("🎯 Bài 5 · MIP lựa chọn dự án chuyển đổi số (15 dự án)")
    policy_box("<b>Bối cảnh:</b> 15 dự án ứng cử, ngân sách 80.000 tỷ VND (2026-2030). Chọn tập dự án tối "
               "đa hoá lợi ích NPV với ràng buộc loại trừ, tiên quyết, cân đối lĩnh vực và số lượng 7≤Σy≤11.")
    solver_name = "PuLP/CBC" if HAS_PULP else "scipy.optimize.milp"
    st.caption(f"Giải bằng **{solver_name}** · biến nhị phân yᵢ ∈ {{0,1}}")

    st.sidebar.markdown("**⚙️ Tham số (Bài 5)**")
    total = st.sidebar.slider("Ngân sách tổng 5 năm (ngh.tỷ)", 60000, 120000, 80000, 5000)
    force = st.sidebar.checkbox("Bắt buộc cả P1 & P2 (redundancy)", False)

    xv, Z = solve_bai5(total, force_p1p2=force)
    if xv is None:
        st.error("❌ Bài toán không khả thi với cấu hình hiện tại.")
        return
    sel = PROJ[xv > 0.5].copy()
    cost = sel.C.sum()
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Tổng lợi ích Z*", f"{Z:,.0f}", "tỷ VND NPV", "#2e7d32")
    with c2: card("Tổng chi phí", f"{cost:,.0f}", f"{cost/total*100:.0f}% ngân sách", INK)
    with c3: card("Số dự án chọn", f"{int(xv.sum())}", "trong khoảng 7-11", ACCENT)
    with c4: card("NPV biên Z*/chi phí", f"{Z/cost:.2f}", "lần", "#2e7d32")

    section("5.4.1 · Danh mục dự án được chọn")
    show = PROJ.copy()
    show["Được chọn"] = np.where(xv > 0.5, "✅", "—")
    show["B/C"] = (show.B / show.C).round(2)
    st.dataframe(show[["Mã", "Tên", "C", "B", "B/C", "Được chọn"]],
                 use_container_width=True, hide_index=True, height=420)

    fig = px.scatter(PROJ, x="C", y="B", text="Mã", color=(xv > 0.5),
                     color_discrete_map={True: ACCENT, False: "#94a3b8"},
                     labels={"C": "Chi phí (tỷ)", "B": "Lợi ích NPV (tỷ)", "color": "Chọn"})
    fig.update_traces(textposition="top center", marker_size=12)
    fig.update_layout(height=380, template="plotly_white", title="Chi phí × Lợi ích NPV (15 dự án)")
    st.plotly_chart(fig, use_container_width=True)

    section("5.4.3 · Bắt buộc cả P1 & P2 (redundancy)")
    xv2, Z2 = solve_bai5(total, force_p1p2=True)
    base = solve_bai5(total, force_p1p2=False)[1]
    if xv2 is not None:
        st.warning(f"Khi ép chọn cả P1 & P2: Z* = **{Z2:,.0f}** "
                   f"(giảm **{base-Z2:,.0f}** tỷ so với {base:,.0f}). Redundancy 2 trung tâm dữ liệu "
                   "có chi phí cơ hội rõ rệt vì cả hai cùng chiếm ngân sách lớn.")
    else:
        st.error("❌ Không khả thi khi ép cả P1 & P2.")

    policy_box("<b>Thảo luận:</b> P15 (Open Data) tỷ lệ B/C cao nhưng quy mô nhỏ, dễ bị loại do ràng buộc "
               "số lượng và cạnh tranh ngân sách. Ép P14 (an ninh mạng) làm giảm nhẹ Z* nhưng hợp lý về "
               "rủi ro hệ thống. Hiệu ứng cộng hưởng P8↔P13 cần thêm biến tích zₐ=y₈·y₁₃ (tuyến tính hoá).")


# ============================================================================
# BÀI 6 — TOPSIS XẾP HẠNG 6 VÙNG (sẵn sàng AI)
# ============================================================================
CRIT6 = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD", "digital_index_0_100",
         "ai_readiness_0_100", "trained_labor_pct", "rd_intensity_pct",
         "internet_penetration_pct", "gini_coef"]
CRIT6_LBL = ["GRDP/người", "FDI", "Digital Index", "AI Readiness",
             "LĐ đào tạo", "R&D/GRDP", "Internet", "Gini"]
IS_BENEFIT = np.array([1, 1, 1, 1, 1, 1, 1, 0])


def topsis(w):
    X = REGIONS[CRIT6].values.astype(float)
    R = X / np.sqrt((X**2).sum(0))
    V = R * w
    Astar = np.where(IS_BENEFIT == 1, V.max(0), V.min(0))
    Aneg = np.where(IS_BENEFIT == 1, V.min(0), V.max(0))
    Sstar = np.sqrt(((V - Astar)**2).sum(1))
    Sneg = np.sqrt(((V - Aneg)**2).sum(1))
    return Sneg / (Sstar + Sneg)


def entropy_weights():
    X = REGIONS[CRIT6].values.astype(float).copy()
    X[:, IS_BENEFIT == 0] = X[:, IS_BENEFIT == 0].max(0) - X[:, IS_BENEFIT == 0]
    P = X / X.sum(0)
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    return d / d.sum()


def render_bai6():
    st.title("🏆 Bài 6 · TOPSIS xếp hạng 6 vùng theo mức sẵn sàng AI")
    policy_box("<b>Bối cảnh (QĐ 127/QĐ-TTg):</b> chọn vùng triển khai trung tâm AI & sandbox dữ liệu trước. "
               "Áp dụng TOPSIS (chuẩn hoá vector, khoảng cách tới phương án lý tưởng tốt/xấu).")

    st.sidebar.markdown("**⚙️ Trọng số TOPSIS (Bài 6)**")
    use_entropy = st.sidebar.checkbox("Dùng trọng số Entropy (khách quan)", False)
    w_default = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
    w_ai = st.sidebar.slider("Trọng số AI Readiness", 0.05, 0.40, 0.20, 0.05)
    w = w_default.copy(); w[3] = w_ai; w = w / w.sum()
    if use_entropy:
        w = entropy_weights()

    C = topsis(w)
    df = REGIONS[["region_name_vi"]].copy()
    df["C*"] = C
    df = df.sort_values("C*", ascending=False).reset_index(drop=True)
    df.index += 1

    c1, c2 = st.columns([1, 1.2])
    with c1:
        section("6.4.1 · Xếp hạng C*")
        st.dataframe(df.rename_axis("Hạng").reset_index(), use_container_width=True, hide_index=True)
        st.markdown("**Bộ trọng số đang dùng:**")
        wdf = pd.DataFrame({"Tiêu chí": CRIT6_LBL, "w": np.round(w, 3)})
        st.dataframe(wdf, use_container_width=True, hide_index=True, height=320)
    with c2:
        fig = px.bar(df.sort_values("C*"), x="C*", y="region_name_vi", orientation="h",
                     color="C*", color_continuous_scale="Reds")
        fig.update_layout(height=320, template="plotly_white", yaxis_title="",
                          title="Hệ số gần gũi C* (càng cao càng ưu tiên)")
        st.plotly_chart(fig, use_container_width=True)
        # so sánh chuyên gia vs entropy
        Ce, Cd = topsis(entropy_weights()), topsis(w_default / w_default.sum())
        comp = pd.DataFrame({"Vùng": REGION_SHORT, "Chuyên gia": Cd, "Entropy": Ce})
        fig2 = px.line(comp, x="Vùng", y=["Chuyên gia", "Entropy"], markers=True,
                       color_discrete_sequence=[ACCENT, INK])
        fig2.update_layout(height=260, template="plotly_white", title="6.4.2 · So sánh trọng số")
        st.plotly_chart(fig2, use_container_width=True)

    top3 = df.head(3)["region_name_vi"].tolist()
    st.success(f"🏆 **3 vùng đặt trung tâm AI (QĐ 127):** {', '.join(top3)}")

    section("6.4.3 · Độ nhạy theo trọng số AI Readiness")
    rng = np.arange(0.10, 0.41, 0.05)
    ranks = []
    for v in rng:
        wv = w_default.copy(); wv[3] = v; wv = wv / wv.sum()
        ranks.append(topsis(wv))
    ranks = np.array(ranks)
    fig = go.Figure()
    for i in range(6):
        fig.add_trace(go.Scatter(x=rng, y=ranks[:, i], mode="lines+markers",
                                 name=REGION_SHORT[i], line=dict(color=PALETTE[i])))
    fig.update_layout(height=320, template="plotly_white", xaxis_title="Trọng số AI Readiness",
                      yaxis_title="C*", title="Ổn định Top-3 khi tăng w_AI")
    st.plotly_chart(fig, use_container_width=True)

    policy_box("<b>Thảo luận:</b> ĐNB và ĐBSH dẫn đầu ổn định → nên đặt trung tâm AI quốc gia đầu tiên. "
               "Khi dùng Entropy, vùng có phương sai dữ liệu lớn được tăng trọng số → xếp hạng giữa bảng "
               "biến động mạnh nhất. AI Readiness và Internet tương quan cao → có thể gộp/PCA để tránh "
               "đếm trùng. Cần thêm tiêu chí địa-chính trị khi chọn đủ 3 trung tâm.")


# ============================================================================
# BÀI 7 — TỐI ƯU ĐA MỤC TIÊU PARETO (NSGA-II nhẹ bằng numpy)
# ============================================================================
E_R = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])      # phát thải
RHO_R = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])    # rủi ro/AI
SIG_R = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])    # giảm rủi ro/H


def evaluate7(X):  # X: (N,6,4)
    f1 = (BETA4 * X).sum((1, 2))                      # max GDP gain
    sums = X.sum(2)
    f2 = np.abs(sums - sums.mean(1, keepdims=True)).mean(1)   # min Gini~MAD
    f3 = (E_R * (X[:, :, 0] + X[:, :, 2])).sum(1)     # min phát thải
    f4 = (RHO_R * X[:, :, 2]).sum(1) - (SIG_R * X[:, :, 3]).sum(1)  # min rủi ro
    return np.column_stack([-f1, f2, f3, f4])         # tất cả về dạng min


def feasible_sample(n=4000, total=50000, floor=5000, cap=12000, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        reg = rng.uniform(floor, cap, size=(n, 6))
        reg = reg / reg.sum(1, keepdims=True) * rng.uniform(0.85, 1.0, (n, 1)) * total
        ok = (reg.sum(1) <= total + 1) & (reg.min(1) >= floor - 1) & (reg.max(1) <= cap + 1)
        for r in reg[ok]:
            w = rng.dirichlet(np.ones(4), size=6)      # tỷ lệ I,D,AI,H mỗi vùng
            X = (r[:, None] * w)
            out.append(X)
            if len(out) >= n:
                break
    return np.array(out[:n])


def pareto_front(F):
    n = len(F); dom = np.ones(n, bool)
    for i in range(n):
        if not dom[i]:
            continue
        d = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
        d[i] = False
        dom[d] = False
    return dom


@st.cache_data
def run_nsga2(n=4000):
    X = feasible_sample(n)
    F = evaluate7(X)
    mask = pareto_front(F)
    return X[mask], F[mask]


def render_bai7():
    st.title("🌐 Bài 7 · Tối ưu đa mục tiêu Pareto (NSGA-II)")
    policy_box("<b>Bối cảnh:</b> phát triển kinh tế số đồng thời 4 mục tiêu xung đột — (1) tăng trưởng GDP, "
               "(2) bao trùm (giảm Gini), (3) môi trường (COP26), (4) an ninh dữ liệu. Nghiệm là <b>tập "
               "Pareto</b>, không phải một điểm tối ưu duy nhất.")
    st.caption("Triển khai bằng lấy mẫu tiến hoá + lọc không-bị-trội (numpy) để chạy nhẹ trên web; "
               "tương đương ý tưởng NSGA-II của pymoo.")

    Xp, Fp = run_nsga2()
    f1 = -Fp[:, 0]; f2 = Fp[:, 1]; f3 = Fp[:, 2]; f4 = Fp[:, 3]
    c1, c2, c3 = st.columns(3)
    with c1: card("Số nghiệm Pareto", f"{len(Fp)}", "không bị trội", INK)
    with c2: card("GDP gain cao nhất", f"{f1.max():,.0f}", "tỷ VND (f₁)", "#2e7d32")
    with c3: card("Bao trùm tốt nhất", f"{f2.min():,.0f}", "MAD nhỏ nhất (f₂)", ACCENT)

    section("7.4.2 · Biên Pareto 3D & toạ độ song song")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter_3d(x=f1, y=f2, z=f3, color=f4,
                            labels={"x": "f₁ GDP", "y": "f₂ Gini", "z": "f₃ Phát thải", "color": "f₄"},
                            color_continuous_scale="RdYlGn_r")
        fig.update_traces(marker_size=4)
        fig.update_layout(height=420, template="plotly_white", title="Biên Pareto (f₁,f₂,f₃)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        dpc = pd.DataFrame({"f₁ Tăng trưởng": f1, "f₂ Bao trùm": f2,
                            "f₃ Môi trường": f3, "f₄ An ninh": f4})
        fig = px.parallel_coordinates(dpc, color="f₁ Tăng trưởng",
                                      color_continuous_scale="Reds")
        fig.update_layout(height=420, template="plotly_white", title="Parallel coordinates 4 mục tiêu")
        st.plotly_chart(fig, use_container_width=True)

    section("7.4.3 · Nghiệm thoả hiệp (TOPSIS trên tập Pareto)")
    w74 = st.columns(4)
    wg = w74[0].slider("Tăng trưởng", 0.0, 1.0, 0.40, 0.05)
    wi = w74[1].slider("Bao trùm", 0.0, 1.0, 0.25, 0.05)
    we = w74[2].slider("Môi trường", 0.0, 1.0, 0.20, 0.05)
    ws = w74[3].slider("An ninh", 0.0, 1.0, 0.15, 0.05)
    W = np.array([wg, wi, we, ws]); W = W / W.sum()
    # TOPSIS trên F (tất cả là min) -> chuẩn hoá
    Fn = Fp / np.sqrt((Fp**2).sum(0))
    Vt = Fn * W
    best = Vt.min(0); worst = Vt.max(0)
    Sp = np.sqrt(((Vt - best)**2).sum(1)); Sn = np.sqrt(((Vt - worst)**2).sum(1))
    Cc = Sn / (Sp + Sn)
    idx = int(np.argmax(Cc))
    fmax = int(np.argmax(f1))   # nghiệm tăng trưởng cao nhất
    rcomp = pd.DataFrame({
        "Nghiệm": ["Thoả hiệp (TOPSIS)", "Tăng trưởng cao nhất"],
        "f₁ GDP": [f1[idx], f1[fmax]], "f₂ Bao trùm": [f2[idx], f2[fmax]],
        "f₃ Môi trường": [f3[idx], f3[fmax]], "f₄ An ninh": [f4[idx], f4[fmax]]})
    st.dataframe(rcomp.style.format({"f₁ GDP": "{:,.0f}", "f₂ Bao trùm": "{:,.0f}",
                 "f₃ Môi trường": "{:,.0f}", "f₄ An ninh": "{:,.0f}"}),
                 use_container_width=True, hide_index=True)
    dg = (f1[fmax] - f1[idx]) / f1[idx] * 100
    di = (f2[fmax] - f2[idx]) / abs(f2[idx]) * 100
    st.info(f"💡 **Chi phí cơ hội:** nghiệm tăng trưởng cao nhất hơn nghiệm thoả hiệp {dg:+.1f}% GDP "
            f"nhưng xấu đi {di:+.1f}% về bao trùm (Gini). Đánh đổi tăng trưởng ↔ công bằng là rõ rệt.")

    policy_box("<b>Thảo luận:</b> Đường biên Pareto cho thấy đánh đổi tăng trưởng–bao trùm gắn với cơ cấu "
               "vùng chênh lệch của Việt Nam. Trọng số (0,40;0,25;0,20;0,15) thiên tăng trưởng — để bám sát "
               "COP26 & QĐ 127 nên nâng trọng số môi trường/an ninh. NSGA-II cung cấp lựa chọn, không thay "
               "quyết định chính trị.")


# ============================================================================
# BÀI 8 — TỐI ƯU ĐỘNG LIÊN THỜI GIAN 2026-2035
# ============================================================================
@st.cache_data
def solve_bai8(rho=0.97, shock=False, strategy="optimize"):
    T = 10
    dK, dD, dAI = 0.05, 0.12, 0.15
    thH, mu = 0.8, 0.02
    p1, p2, p3 = 0.003, 0.002, 0.004
    K0, L, D0_, AI0, H0, A0 = 27500., 54., 20.3, 86., 30., 34.9
    aK, bL, gD, dAI_, thH_ = 0.33, 0.42, 0.10, 0.08, 0.07

    def simulate(shares):
        K, D, AI, H, A = K0, D0_, AI0, H0, A0
        Cs, Ks, Ds, AIs, Hs, Ys = [], [], [], [], [], []
        for t in range(T):
            Y = A * K**aK * L**bL * D**gD * AI**dAI_ * H**thH_
            if shock and t == 2:
                Y *= 0.92
            sK, sD, sAI, sH = shares
            invest = (sK + sD + sAI + sH)
            inv_total = min(invest, 0.6) * Y       # tối đa 60% Y dành đầu tư
            IK, ID, IAI, IH = inv_total * np.array([sK, sD, sAI, sH]) / max(invest, 1e-6)
            C = Y - inv_total
            Cs.append(max(C, 1e-6)); Ks.append(K); Ds.append(D); AIs.append(AI); Hs.append(H); Ys.append(Y)
            K = (1 - dK) * K + IK
            D = (1 - dD) * D + ID / 100
            AI = (1 - dAI) * AI + IAI / 20
            H = H + thH * IH / 200 - mu * H
            A = A * (1 + p1 * D + p2 * AI + p3 * H) ** 0.001
        return np.array(Cs), np.array(Ks), np.array(Ds), np.array(AIs), np.array(Hs), np.array(Ys)

    def neg_welfare(shares):
        Cs = simulate(shares)[0]
        return -sum(rho**t * np.log(Cs[t]) for t in range(T))

    if strategy == "even":
        sh = np.array([0.15, 0.05, 0.05, 0.05])
    elif strategy == "frontload":
        sh = np.array([0.25, 0.08, 0.08, 0.08])
    else:
        res = minimize(neg_welfare, x0=[0.15, 0.05, 0.05, 0.05], method="SLSQP",
                       bounds=[(0.01, 0.4)] * 4)
        sh = res.x
    Cs, Ks, Ds, AIs, Hs, Ys = simulate(sh)
    W = float(sum(rho**t * np.log(Cs[t]) for t in range(T)))
    return sh, Cs, Ks, Ds, AIs, Hs, Ys, W


def render_bai8():
    st.title("⏳ Bài 8 · Tối ưu động phân bổ liên thời gian 2026-2035")
    policy_box("<b>Bối cảnh (Đại hội XIII):</b> thiết kế quỹ đạo phân bổ vốn 10 năm tối đa hoá phúc lợi "
               "xã hội chiết khấu, có động học tích luỹ K/D/AI/H và cập nhật TFP nội sinh.")
    st.latex(r"\max \sum_{t} \rho^{t}\,\ln C_t \quad s.t.\quad C_t+\sum I_{j,t}\le Y_t")

    st.sidebar.markdown("**⚙️ Tham số (Bài 8)**")
    rho = st.sidebar.slider("ρ — hệ số chiết khấu", 0.85, 0.99, 0.97, 0.01)
    shock = st.sidebar.checkbox("Cú sốc 2028 (−8% Y, kiểu bão Yagi)", False)

    sh, Cs, Ks, Ds, AIs, Hs, Ys, W = solve_bai8(rho, shock, "optimize")
    yrs = list(range(2026, 2036))
    c1, c2, c3 = st.columns(3)
    with c1: card("Phúc lợi tổng W*", f"{W:.3f}", "Σ ρᵗ ln Cₜ", "#2e7d32")
    with c2: card("Y 2035 (ngh.tỷ)", f"{Ys[-1]:,.0f}", f"từ {Ys[0]:,.0f} (2026)", INK)
    with c3: card("Tỷ lệ đầu tư K:H", f"{sh[0]/max(sh[3],1e-6):.1f}:1", "trung bình", ACCENT)

    section("8.3.2 · Quỹ đạo tối ưu K, D, AI, H, Y, C")
    fig = make_subplots(rows=2, cols=3, subplot_titles=("Vốn K", "Số hoá D", "Năng lực AI",
                        "Nhân lực H", "Sản lượng Y", "Tiêu dùng C"))
    series = [Ks, Ds, AIs, Hs, Ys, Cs]
    pos = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    for s, (r, cl), col in zip(series, pos, PALETTE):
        fig.add_trace(go.Scatter(x=yrs, y=s, mode="lines+markers",
                      line=dict(color=col, width=2.5), showlegend=False), row=r, col=cl)
    fig.update_layout(height=460, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    section("8.3.4 · So sánh chiến lược đầu tư")
    _, _, _, _, _, _, _, W_even = solve_bai8(rho, shock, "even")
    _, _, _, _, _, _, _, W_front = solve_bai8(rho, shock, "frontload")
    cmp = pd.DataFrame({"Chiến lược": ["Tối ưu (SLSQP)", "Trải đều", "Front-load"],
                        "Phúc lợi W": [W, W_even, W_front]})
    fig = px.bar(cmp, x="Chiến lược", y="Phúc lợi W", color="Chiến lược",
                 color_discrete_sequence=PALETTE)
    fig.update_layout(height=300, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    policy_box("<b>Thảo luận:</b> Quỹ đạo tối ưu thường 'front-load' nhẹ đầu tư D/AI để hưởng lợi tích luỹ "
               "TFP dài hạn. ρ cao (0,97) → chú trọng tương lai, tăng đầu tư R&D/H; ρ thấp (0,90) → tiêu dùng "
               "sớm, dưới đầu tư — lý giải vì sao chính phủ ngắn hạn hay dưới đầu tư vào R&D.")


# ============================================================================
# BÀI 9 — TÁC ĐỘNG AI TỚI THỊ TRƯỜNG LAO ĐỘNG
# ============================================================================
SEC9 = ["Nông-Lâm-TS", "CN chế biến", "Xây dựng", "Bán buôn-lẻ",
        "Tài chính-NH", "Logistics", "CNTT-TT", "Giáo dục"]
L9 = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
RISK9 = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
A1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
B1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
C1c = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
D1 = np.array([50, 32, 42, 38, 26, 36, 24, 62])


def solve_bai9(budget=30000, cap5pct=False):
    # biến: x_AI[8], x_H[8] => 16 biến
    n = 16
    # NetJob_i = a1*xAI + b1*xH - (c1*risk)*xAI ; maximize sum
    coef_AI = A1 - C1c * RISK9
    coef_H = B1
    c = -np.concatenate([coef_AI, coef_H])
    A_ub, b_ub = [], []
    # ngân sách
    row = np.ones(n); A_ub.append(row); b_ub.append(budget)
    for i in range(8):
        # NetJob_i >= 0 -> -(coefAI*xAI + b1*xH) <= 0
        row = np.zeros(n); row[i] = -coef_AI[i]; row[8 + i] = -coef_H[i]
        A_ub.append(row); b_ub.append(0)
        # Displaced <= RetrainCap : c1*risk*xAI - d1*xH <= 0
        row = np.zeros(n); row[i] = C1c[i] * RISK9[i]; row[8 + i] = -D1[i]
        A_ub.append(row); b_ub.append(0)
        if cap5pct:
            # Displaced <= 0.05*L (đơn vị: việc/tỷ * tỷ -> nghìn việc; L triệu)
            row = np.zeros(n); row[i] = C1c[i] * RISK9[i]
            A_ub.append(row); b_ub.append(0.05 * L9[i] * 1000)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n, method="highs")
    return res, coef_AI, coef_H


def render_bai9():
    st.title("👷 Bài 9 · Tác động AI tới thị trường lao động")
    policy_box("<b>Bối cảnh (ILO/OECD 2024):</b> 30-50% việc làm Việt Nam có nguy cơ tự động hoá nhưng AI "
               "cũng tạo việc mới. Phân bổ 30.000 tỷ cho 8 ngành (x_AI, x_H) tối đa hoá NetJob ròng, "
               "ràng buộc NetJobᵢ≥0 và Displacedᵢ ≤ RetrainCapᵢ.")
    st.latex(r"NetJob_i = a_{1i}x^{AI}_i + b_{1i}x^{H}_i - c_{1i}\,risk_i\,x^{AI}_i \ge 0")

    st.sidebar.markdown("**⚙️ Tham số (Bài 9)**")
    budget = st.sidebar.slider("Ngân sách (ngh.tỷ)", 20000, 50000, 30000, 5000)
    cap5 = st.sidebar.checkbox("Ràng buộc: mất ≤ 5% LĐ/ngành (9.4.4)", False)

    res, cAI, cH = solve_bai9(budget, cap5)
    if not res.success:
        st.error("❌ Không khả thi với ràng buộc hiện tại (vd: ràng buộc 5% quá chặt cho ngành rủi ro cao).")
        return
    xAI = res.x[:8]; xH = res.x[8:]
    NewJob = A1 * xAI; Upgrade = B1 * xH
    Displaced = C1c * RISK9 * xAI; NetJob = NewJob + Upgrade - Displaced
    c1, c2, c3 = st.columns(3)
    with c1: card("Tổng NetJob ròng", f"{NetJob.sum():,.0f}", "nghìn việc làm", "#2e7d32")
    with c2: card("Tổng việc dịch chuyển", f"{Displaced.sum():,.0f}", "nghìn việc (Displaced)", ACCENT)
    with c3: card("Phần ngân sách → đào tạo H", f"{xH.sum()/budget*100:.0f}%", "ưu tiên kỹ năng", INK)

    section("9.4.1 · Phân bổ tối ưu & NetJob theo ngành")
    df9 = pd.DataFrame({"Ngành": SEC9, "x_AI": np.round(xAI, 0), "x_H": np.round(xH, 0),
                        "NewJob": np.round(NewJob, 0), "Displaced": np.round(Displaced, 0),
                        "NetJob": np.round(NetJob, 0)})
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(df9, use_container_width=True, hide_index=True, height=340)
    with c2:
        fig = px.bar(df9, x="Ngành", y=["x_AI", "x_H"], barmode="stack",
                     color_discrete_sequence=[ACCENT, ACCENT2])
        fig.update_layout(height=340, template="plotly_white", title="Phân bổ x_AI vs x_H")
        st.plotly_chart(fig, use_container_width=True)

    section("9.4.3 · Luồng dịch chuyển lao động (Sankey)")
    src, tgt, val = [], [], []
    labels = SEC9 + ["Việc mới (AI)", "Nâng cấp (H)", "Dịch chuyển (mất)"]
    for i in range(8):
        src += [i, i, i]; tgt += [8, 9, 10]
        val += [max(NewJob[i], 0.1), max(Upgrade[i], 0.1), max(Displaced[i], 0.1)]
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=14, thickness=14,
                  color=PALETTE[:8] + ["#2e7d32", "#1565c0", "#d4001f"]),
        link=dict(source=src, target=tgt, value=val)))
    fig.update_layout(height=420, template="plotly_white", title="Luồng lao động 8 ngành")
    st.plotly_chart(fig, use_container_width=True)

    section("9.4.2 · Ngưỡng đào tạo tối thiểu ngành CN chế biến chế tạo")
    # x_H để NetJob>=0 khi xAI tối đa: b1*xH >= (c1*risk - a1)*xAI
    i = 1
    st.info(f"Với ngành **CN chế biến chế tạo** (rủi ro 42%): hệ số NetJob của AI là "
            f"a₁−c₁·risk = {cAI[i]:.2f} (>0 → AI vẫn tạo việc ròng). Khi đầu tư AI lớn, cần x_H ≥ "
            f"phần bù để giữ NetJob≥0. Mô hình tự cân đối qua ràng buộc Displaced ≤ RetrainCap.")

    if cap5:
        st.warning("⚠️ Ràng buộc 'mất ≤ 5% LĐ/ngành' siết mạnh các ngành rủi ro cao (Tài chính-NH, "
                   "CN chế biến) → giảm dư địa đầu tư AID, tổng NetJob thấp hơn nhưng an sinh tốt hơn.")

    policy_box("<b>Thảo luận:</b> Ngành cần đào tạo lại nhiều nhất là CN chế biến & Bán buôn-lẻ (rủi ro + "
               "lao động lớn). Tài chính-NH rủi ro 52% nhưng a₁ rất cao → chiến lược 'AI + tái đào tạo song "
               "song'. Nông nghiệp: a₁ thấp nhưng lao động khổng lồ → ưu tiên H/nâng cấp hơn AI thuần. "
               "Phát biểu 'tốc độ tự động hoá ≤ năng lực đào tạo lại' = ràng buộc Displaced ≤ RetrainCap.")


# ============================================================================
# BÀI 10 — QUY HOẠCH NGẪU NHIÊN HAI GIAI ĐOẠN (VSS & EVPI)
# ============================================================================
J10 = ["I", "D", "AI", "H"]
P10 = {"s1": 0.30, "s2": 0.45, "s3": 0.20, "s4": 0.05}
S10 = list(P10)
BETA0 = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}
BETA_S = {
    "s1": {"I": 1.25, "D": 1.35, "AI": 1.55, "H": 1.05},
    "s2": {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95},
    "s3": {"I": 0.75, "D": 0.85, "AI": 0.90, "H": 1.00},
    "s4": {"I": 0.40, "D": 0.50, "AI": 0.55, "H": 1.10},
}
CAP1 = 30000     # trần hấp thụ mỗi hạng mục GĐ1
B1_TOTAL = 65000
B2_TOTAL = 15000


def _stage2_value(x, s):
    # tối ưu second-stage cho 1 kịch bản, x cố định
    b = BETA_S[s]
    c = -np.array([b[j] for j in J10])
    A_ub = [np.ones(4)]; b_ub = [B2_TOTAL]
    # y_AI <= 0.5 x_H
    row = np.zeros(4); row[2] = 1; A_ub.append(row); b_ub.append(0.5 * x[3])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")
    return -res.fun if res.success else 0.0, (res.x if res.success else np.zeros(4))


def solve_SP():
    # x[4] + y[s,4]*4 = 20 biến
    nv = 20
    c = np.zeros(nv)
    for i, j in enumerate(J10): c[i] = BETA0[j]
    idx = lambda s, j: 4 + S10.index(s) * 4 + J10.index(j)
    for s in S10:
        for j in J10: c[idx(s, j)] = P10[s] * BETA_S[s][j]
    A_ub, b_ub = [], []
    row = np.zeros(nv); row[:4] = 1; A_ub.append(row); b_ub.append(B1_TOTAL)
    for s in S10:
        row = np.zeros(nv)
        for j in J10: row[idx(s, j)] = 1
        A_ub.append(row); b_ub.append(B2_TOTAL)
        row = np.zeros(nv); row[idx(s, "AI")] = 1; row[3] = -0.5; A_ub.append(row); b_ub.append(0)
    bounds = [(0, CAP1)] * 4 + [(0, None)] * 16
    res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    x = res.x[:4]
    return x, -res.fun


def solve_one_scenario(s):
    # first+second stage tối ưu khi biết trước kịch bản s (wait-and-see)
    nv = 8
    c = np.zeros(nv)
    for i, j in enumerate(J10): c[i] = BETA0[j]; c[4 + i] = BETA_S[s][j]
    A_ub, b_ub = [], []
    row = np.zeros(nv); row[:4] = 1; A_ub.append(row); b_ub.append(B1_TOTAL)
    row = np.zeros(nv); row[4:] = 1; A_ub.append(row); b_ub.append(B2_TOTAL)
    row = np.zeros(nv); row[6] = 1; row[3] = -0.5; A_ub.append(row); b_ub.append(0)
    bounds = [(0, CAP1)] * 4 + [(0, None)] * 4
    res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res.x[:4], -res.fun


def solve_EV():
    # kịch bản kỳ vọng: beta_s trung bình theo xác suất
    bbar = {j: sum(P10[s] * BETA_S[s][j] for s in S10) for j in J10}
    nv = 8
    c = np.zeros(nv)
    for i, j in enumerate(J10): c[i] = BETA0[j]; c[4 + i] = bbar[j]
    A_ub, b_ub = [], []
    row = np.zeros(nv); row[:4] = 1; A_ub.append(row); b_ub.append(B1_TOTAL)
    row = np.zeros(nv); row[4:] = 1; A_ub.append(row); b_ub.append(B2_TOTAL)
    row = np.zeros(nv); row[6] = 1; row[3] = -0.5; A_ub.append(row); b_ub.append(0)
    bounds = [(0, CAP1)] * 4 + [(0, None)] * 4
    res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res.x[:4]


@st.cache_data
def bai10_metrics():
    x_sp, RP = solve_SP()
    # WS
    WS = 0.0; x_ws = {}
    for s in S10:
        xs, vs = solve_one_scenario(s); WS += P10[s] * vs; x_ws[s] = xs
    # EEV: dùng x_EV, đánh giá kỳ vọng second-stage
    x_ev = solve_EV()
    first_ev = sum(BETA0[j] * x_ev[i] for i, j in enumerate(J10))
    EEV = first_ev + sum(P10[s] * _stage2_value(x_ev, s)[0] for s in S10)
    VSS = max(RP - EEV, 0.0)
    EVPI = max(WS - RP, 0.0)
    return x_sp, x_ev, RP, EEV, WS, VSS, EVPI


def render_bai10():
    st.title("🎲 Bài 10 · Quy hoạch ngẫu nhiên hai giai đoạn")
    policy_box("<b>Bối cảnh:</b> Việt Nam độ mở thương mại ~180% GDP → tăng trưởng phụ thuộc kịch bản toàn "
               "cầu. Quyết định GĐ1 'here-and-now' (≤65.000 tỷ, giữ 15.000 tỷ dự phòng) trước khi biết kịch "
               "bản; GĐ2 'recourse' điều chỉnh theo 4 kịch bản (lạc quan/cơ sở/bi quan/khủng hoảng).")

    st.dataframe(pd.DataFrame({
        "Kịch bản": ["Lạc quan", "Cơ sở", "Bi quan", "Khủng hoảng"],
        "Xác suất": [0.30, 0.45, 0.20, 0.05],
        "β_AI": [1.55, 1.25, 0.90, 0.55], "β_H": [1.05, 0.95, 1.00, 1.10]}),
        use_container_width=True, hide_index=True)

    x_sp, x_ev, RP, EEV, WS, VSS, EVPI = bai10_metrics()
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("RP (stochastic)", f"{RP:,.0f}", "lời giải SP", "#2e7d32")
    with c2: card("EEV (lời giải kỳ vọng)", f"{EEV:,.0f}", "dùng x_EV", INK)
    with c3: card("VSS = RP − EEV", f"{VSS:,.0f}", "giá trị tư duy xác suất", ACCENT)
    with c4: card("EVPI = WS − RP", f"{EVPI:,.0f}", "giá trị thông tin hoàn hảo", "#6a1b9a")

    section("10.5.1 & 10.5.2 · Quyết định GĐ1: SP vs EV")
    dfx = pd.DataFrame({"Hạng mục": J10,
                        "SP (stochastic)": np.round(x_sp, 0),
                        "EV (kỳ vọng)": np.round(x_ev, 0)})
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.dataframe(dfx, use_container_width=True, hide_index=True)
    with c2:
        fig = px.bar(dfx, x="Hạng mục", y=["SP (stochastic)", "EV (kỳ vọng)"],
                     barmode="group", color_discrete_sequence=[ACCENT, "#94a3b8"])
        fig.update_layout(height=320, template="plotly_white", title="Phân bổ GĐ1")
        st.plotly_chart(fig, use_container_width=True)

    dh = x_sp[3] - x_ev[3]
    st.info(f"💡 Lời giải SP đầu tư nhân lực H {'**nhiều hơn**' if dh>0 else 'khác'} EV "
            f"({dh:+,.0f} ngh.tỷ): H là 'hàng hoá bảo hiểm' — trong kịch bản khủng hoảng β_H cao nhất (1,10) "
            "và H còn mở khoá năng lực AI GĐ2 (ràng buộc y_AI ≤ 0,5·x_H).")

    section("10.5.3 · Phân rã giá trị: WS ≥ RP ≥ EEV")
    fig = go.Figure(go.Bar(x=["EEV", "RP (SP)", "WS"], y=[EEV, RP, WS],
                           marker_color=["#94a3b8", ACCENT, "#2e7d32"], text=[f"{v:,.0f}" for v in [EEV, RP, WS]]))
    fig.update_layout(height=320, template="plotly_white", title="EEV ≤ RP ≤ WS",
                      yaxis_title="GDP gain kỳ vọng")
    st.plotly_chart(fig, use_container_width=True)

    policy_box("<b>Thảo luận:</b> VSS dương khẳng định: cân nhắc bất định khi quyết định mang lại giá trị "
               "thực — bỏ qua rủi ro (EV) khiến phân bổ kém bền. EVPI là cận trên cho chi cho dự báo/thông "
               "tin. Bài học COVID-19 & bão Yagi: Việt Nam nên xem nhân lực số như khoản 'bảo hiểm' chống sốc.")


# ============================================================================
# BÀI 11 — Q-LEARNING CHO CHÍNH SÁCH KINH TẾ THÍCH NGHI
# ============================================================================
ALLOC11 = {0: [0.70, 0.10, 0.10, 0.10], 1: [0.40, 0.25, 0.15, 0.20],
           2: [0.25, 0.45, 0.15, 0.15], 3: [0.20, 0.20, 0.45, 0.15],
           4: [0.30, 0.20, 0.10, 0.40]}
ACT_NAME = ["a0 Truyền thống", "a1 Cân bằng", "a2 Số hoá nhanh", "a3 AI dẫn dắt", "a4 Bao trùm"]
W11 = np.array([0.40, 0.25, 0.20, 0.15])


def step_env(state, action, rng):
    a = np.array(ALLOC11[action])
    g, d, ai, u = state
    # ΔGDP ~ trọng đầu tư K + D + AI; thất nghiệp giảm theo H; cyber theo AI; phát thải theo K
    dGDP = 0.5 * a[0] + 0.8 * a[1] + 1.0 * a[2] + 0.3 * a[3] + 0.05 * rng.standard_normal()
    dUnemp = -0.6 * a[3] + 0.4 * a[2] - 0.1 * a[1]
    cyber = 0.7 * a[2] - 0.3 * a[3]
    emis = 0.6 * a[0] + 0.2 * a[2]
    reward = W11[0] * dGDP - W11[1] * dUnemp - W11[2] * cyber - W11[3] * emis
    # cập nhật trạng thái rời rạc (clip 0..2)
    def upd(v, x): return int(np.clip(v + (1 if x > 0.15 else (-1 if x < -0.05 else 0)), 0, 2))
    ns = np.array([upd(g, dGDP), upd(d, a[1]), upd(ai, a[2]), upd(u, -dUnemp)])
    return ns, reward


@st.cache_data
def train_qlearning(episodes=4000, alpha=0.1, gamma=0.95, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((3, 3, 3, 3, 5))
    curve = []
    for ep in range(episodes):
        s = np.array([1, 1, 0, 1]); tot = 0.0
        eps = max(0.05, 1.0 - ep / (episodes * 0.5))
        for t in range(10):
            a = rng.integers(5) if rng.random() < eps else int(np.argmax(Q[tuple(s)]))
            ns, r = step_env(s, a, rng)
            Q[tuple(s) + (a,)] += alpha * (r + gamma * Q[tuple(ns)].max() - Q[tuple(s) + (a,)])
            s = ns; tot += r
        curve.append(tot)
    # smoothing
    curve = np.convolve(curve, np.ones(50) / 50, mode="valid")
    return Q, curve


def eval_policy(Q, mode, seed=1, episodes=200):
    rng = np.random.default_rng(seed); tot = 0.0
    for _ in range(episodes):
        s = np.array([1, 1, 0, 1])
        for t in range(10):
            if mode == "q": a = int(np.argmax(Q[tuple(s)]))
            elif mode == "a1": a = 1
            elif mode == "a3": a = 3
            else: a = rng.integers(5)
            s, r = step_env(s, a, rng); tot += r
    return tot / episodes


def render_bai11():
    st.title("🤖 Bài 11 · Q-learning cho chính sách kinh tế thích nghi")
    policy_box("<b>Bối cảnh:</b> nền kinh tế = môi trường, chính sách = hành động, phần thưởng = phúc lợi xã "
               "hội. MDP 81 trạng thái (3⁴) × 5 hành động ngân sách. <i>Lưu ý: AI hỗ trợ chứ không thay thế "
               "trách nhiệm chính trị.</i>")
    st.latex(r"R_t = 0{,}40\,\Delta GDP - 0{,}25\,\Delta U - 0{,}20\,Cyber - 0{,}15\,Emission")

    st.sidebar.markdown("**⚙️ Tham số (Bài 11)**")
    episodes = st.sidebar.select_slider("Số episode huấn luyện", [2000, 4000, 8000], 4000)
    gamma = st.sidebar.slider("γ — discount", 0.80, 0.99, 0.95, 0.01)

    with st.spinner("Đang huấn luyện Q-learning..."):
        Q, curve = train_qlearning(episodes, 0.1, gamma)

    section("11.3.2 & 11.3.4 · Learning curve & so sánh chính sách")
    rq = eval_policy(Q, "q"); r1 = eval_policy(Q, "a1"); r3 = eval_policy(Q, "a3"); rr = eval_policy(Q, "rand")
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("π* (Q-learning)", f"{rq:.2f}", "reward TB/episode", "#2e7d32")
    with c2: card("Luôn a1 (cân bằng)", f"{r1:.2f}", "rule-based", INK)
    with c3: card("Luôn a3 (AI dẫn dắt)", f"{r3:.2f}", "rule-based", INK)
    with c4: card("Ngẫu nhiên", f"{rr:.2f}", "baseline", "#94a3b8")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = px.line(x=np.arange(len(curve)), y=curve,
                      labels={"x": "Episode", "y": "Reward (trượt 50)"})
        fig.update_traces(line_color=ACCENT, line_width=2)
        fig.update_layout(height=320, template="plotly_white", title="Learning curve Q-learning")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cmp = pd.DataFrame({"Chính sách": ["π*", "a1", "a3", "random"],
                            "Reward": [rq, r1, r3, rr]})
        fig = px.bar(cmp, x="Chính sách", y="Reward", color="Chính sách",
                     color_discrete_sequence=[ACCENT, "#1565c0", "#6a1b9a", "#94a3b8"])
        fig.update_layout(height=320, template="plotly_white", showlegend=False,
                          title="So sánh phần thưởng")
        st.plotly_chart(fig, use_container_width=True)

    section("11.3.3 · Chính sách tối ưu π*(s) tại các trạng thái tiêu biểu")
    states = {"VN 2026 (G=TB, D=TB, AI=thấp, U=TB)": (1, 1, 0, 1),
              "Suy giảm (G=thấp, D=thấp, U=cao)": (0, 0, 0, 2),
              "Bứt phá (G=cao, AI=cao, U=thấp)": (2, 2, 2, 0),
              "Quá nóng (G=cao, D=cao, U=thấp)": (2, 2, 1, 0)}
    rows = []
    for name, s in states.items():
        a = int(np.argmax(Q[tuple(s)]))
        rows.append({"Trạng thái": name, "π*(s)": ACT_NAME[a]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    policy_box("<b>Thảo luận:</b> Khi GDP thấp & thất nghiệp cao, π* thường chọn hành động ưu tiên H/D "
               "('quick win' tạo việc làm). Khi kinh tế nóng & AI cao, π* chuyển sang 'consolidation' cân "
               "bằng/bao trùm. Tích hợp π* vào quy trình: dùng làm <b>khuyến nghị</b> cho hội đồng chính "
               "sách, giữ quyết định cuối ở con người — đúng nguyên tắc của Mục 11.")


# ============================================================================
# BÀI 12 — ĐỒ ÁN TÍCH HỢP AIDEOM-VN (6 module → 4 tab)
# ============================================================================
SCENARIOS = {
    "S1 Truyền thống": [0.70, 0.10, 0.10, 0.10],
    "S2 Số hoá nhanh": [0.25, 0.45, 0.15, 0.15],
    "S3 AI dẫn dắt":   [0.20, 0.20, 0.45, 0.15],
    "S4 Bao trùm số":  [0.30, 0.20, 0.10, 0.40],
    "S5 Tối ưu cân bằng": [0.35, 0.25, 0.20, 0.20],
}


def m1_forecast():
    """M1 — Dự báo kinh tế Cobb-Douglas 2026-2030."""
    K, L, D, AI, H, Y, A, _, _ = cobb_compute(0.33, 0.42, 0.10, 0.08, 0.07)
    a, b, g, d, th = 0.33, 0.42, 0.10, 0.08, 0.07
    yrs = list(range(2026, 2031))
    Kn, Ln, Dn, AIn, Hn, An = K[-1], L[-1], D[-1], AI[-1], H[-1], A[-1]
    out = []
    for _ in yrs:
        Kn *= 1.06; Ln *= 1.005; Dn = min(Dn + 2.1, 30); AIn += 4; Hn = min(Hn + 1.3, 35); An *= 1.012
        out.append(An * Kn**a * Ln**b * Dn**g * AIn**d * Hn**th)
    return yrs, np.array(out), Y[-1]


def m2_readiness():
    """M2 — TOPSIS sẵn sàng số 6 vùng."""
    w = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
    C = topsis(w)
    return pd.DataFrame({"region_name_vi": REGIONS.region_name_vi, "C*": C}).sort_values("C*", ascending=False)


def m345_scenario(profile, budget=50000):
    """M3/M4/M5 — KPI tổng hợp cho 1 kịch bản phân bổ [K,D,AI,H]."""
    sK, sD, sAI, sH = profile
    # M3: GDP gain xấp xỉ qua hệ số hiệu quả biên trung bình
    gdp_gain = budget * (0.85 * sK + 1.10 * sD + 1.25 * sAI + 0.95 * sH) / 1000  # ngh.tỷ
    # M4: NetJob (đơn giản hoá) — H & AI tạo việc, AI có rủi ro dịch chuyển
    netjob = budget * (35 * sH + 20 * sAI - 18 * sAI) / 100   # nghìn việc
    # M5: rủi ro tổng hợp (cyber theo AI, phát thải theo K, bất bình đẳng nếu thiếu H)
    risk = 100 * (0.5 * sAI + 0.4 * sK - 0.3 * sH + 0.2)
    digital = 100 * (0.6 * sD + 0.4 * sAI)
    inclusion = 100 * (0.7 * sH + 0.3 * sD)
    return dict(GDP_gain=gdp_gain, NetJob=netjob, Risk=max(risk, 0),
                Digital=digital, Inclusion=inclusion)


def render_bai12():
    st.title("🧠 Bài 12 · Đồ án tích hợp AIDEOM-VN")
    policy_box("<b>Đồ án tổng kết:</b> tích hợp <b>6 module</b> (M1 Dự báo · M2 Sẵn sàng số · M3 Phân bổ · "
               "M4 Lao động · M5 Rủi ro · M6 Dashboard) thành hệ hỗ trợ ra quyết định, so sánh "
               "<b>5 kịch bản chính sách</b> S1-S5. Tổ chức theo 4 tab dưới đây.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Tổng quan (M1+M2)", "🗺️ Phân bổ (M3)",
         "⚖️ So sánh kịch bản (M4)", "🚨 Cảnh báo rủi ro (M5+M6)"])

    # ---------- TAB 1: Tổng quan = M1 + M2 ----------
    with tab1:
        st.subheader("Module M1 — Dự báo kinh tế 2026-2030")
        yrs, Yf, Y0 = m1_forecast()
        c1, c2, c3 = st.columns(3)
        with c1: card("GDP 2030 dự báo (ngh.tỷ)", f"{Yf[-1]:,.0f}", f"CAGR {(np.log(Yf[-1]/Y0)/5)*100:.2f}%", "#2e7d32")
        with c2: card("Tăng so với 2025", f"+{(Yf[-1]/Y0-1)*100:.1f}%", "5 năm", INK)
        with c3: card("GDP 2030 (tỷ USD)", f"{Yf[-1]*1000/25.5:,.0f}", "~25.500 VND/USD", ACCENT)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[2025] + yrs, y=[Y0] + list(Yf), mode="lines+markers",
                      line=dict(color=ACCENT, width=3), name="GDP dự báo"))
        fig.update_layout(height=320, template="plotly_white", title="Quỹ đạo GDP 2025-2030")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Module M2 — Bản đồ mức sẵn sàng số (TOPSIS)")
        m2 = m2_readiness()
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.dataframe(m2.rename(columns={"region_name_vi": "Vùng"}).round(4),
                         use_container_width=True, hide_index=True)
        with c2:
            fig = px.bar(m2.sort_values("C*"), x="C*", y="region_name_vi", orientation="h",
                         color="C*", color_continuous_scale="Reds")
            fig.update_layout(height=320, template="plotly_white", yaxis_title="",
                              title="Xếp hạng sẵn sàng AI 6 vùng")
            st.plotly_chart(fig, use_container_width=True)

    # ---------- TAB 2: Phân bổ = M3 ----------
    with tab2:
        st.subheader("Module M3 — Tối ưu phân bổ ngân sách ngành-vùng")
        res = solve_bai4(True, 0.70)
        if res.success:
            X = res.x[:24].reshape(6, 4)
            c1, c2 = st.columns([1, 1.2])
            with c1:
                card("Z* GDP gain", f"{-res.fun:,.0f}", "tỷ VND", "#2e7d32")
                st.dataframe(pd.DataFrame(X, columns=["I", "D", "AI", "H"],
                             index=REGION_SHORT).style.format("{:,.0f}"), use_container_width=True)
            with c2:
                fig = px.imshow(X, x=["I", "D", "AI", "H"], y=REGION_SHORT, text_auto=".0f",
                                color_continuous_scale="Reds", aspect="auto")
                fig.update_layout(height=360, template="plotly_white", title="Heatmap phân bổ tối ưu (M3)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Cấu hình công bằng vùng hiện vô nghiệm — xem Bài 4.")

    # ---------- TAB 3: So sánh 5 kịch bản = M4 + KPI ----------
    with tab3:
        st.subheader("Module M4 — Mô phỏng lao động & So sánh 5 kịch bản chính sách")
        rows = []
        for name, prof in SCENARIOS.items():
            kpi = m345_scenario(prof)
            rows.append({"Kịch bản": name, "Cơ cấu (K/D/AI/H)": "/".join(f"{int(p*100)}" for p in prof),
                         **{k: round(v, 1) for k, v in kpi.items()}})
        dfk = pd.DataFrame(rows)
        st.dataframe(dfk, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(dfk, x="Kịch bản", y="GDP_gain", color="Kịch bản",
                         color_discrete_sequence=PALETTE, title="GDP gain theo kịch bản")
            fig.update_layout(height=320, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(dfk, x="Kịch bản", y="NetJob", color="Kịch bản",
                         color_discrete_sequence=PALETTE, title="NetJob ròng theo kịch bản")
            fig.update_layout(height=320, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        # radar so sánh đa chiều
        cats = ["GDP_gain", "Digital", "Inclusion", "NetJob"]
        fig = go.Figure()
        for i, (name, prof) in enumerate(SCENARIOS.items()):
            kpi = m345_scenario(prof)
            vals = [kpi["GDP_gain"], kpi["Digital"], kpi["Inclusion"], kpi["NetJob"] / 10]
            fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]],
                          name=name, line=dict(color=PALETTE[i])))
        fig.update_layout(height=420, template="plotly_white",
                          title="Radar đa chiều 5 kịch bản (NetJob/10 để cùng thang)")
        st.plotly_chart(fig, use_container_width=True)

    # ---------- TAB 4: Cảnh báo rủi ro = M5 + M6 ----------
    with tab4:
        st.subheader("Module M5 — Đánh giá rủi ro (Cyber · Môi trường · Bất bình đẳng)")
        risk_rows = []
        for name, prof in SCENARIOS.items():
            kpi = m345_scenario(prof)
            level = "🔴 Cao" if kpi["Risk"] > 50 else ("🟡 Trung bình" if kpi["Risk"] > 30 else "🟢 Thấp")
            risk_rows.append({"Kịch bản": name, "Chỉ số rủi ro": round(kpi["Risk"], 1),
                              "Bao trùm": round(kpi["Inclusion"], 1), "Cảnh báo": level})
        dfr = pd.DataFrame(risk_rows)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(dfr, use_container_width=True, hide_index=True)
        with c2:
            fig = px.scatter(dfr, x="Chỉ số rủi ro", y="Bao trùm", text="Kịch bản",
                             color="Chỉ số rủi ro", color_continuous_scale="RdYlGn_r", size_max=20)
            fig.update_traces(textposition="top center", marker_size=16)
            fig.update_layout(height=340, template="plotly_white", title="Rủi ro × Bao trùm")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Module M6 — Dashboard khuyến nghị chính sách")
        best = max(SCENARIOS, key=lambda n: m345_scenario(SCENARIOS[n])["GDP_gain"]
                   - 0.5 * m345_scenario(SCENARIOS[n])["Risk"] + m345_scenario(SCENARIOS[n])["Inclusion"])
        st.success(f"🏆 **Khuyến nghị AIDEOM-VN:** kịch bản **{best}** đạt cân bằng tốt nhất giữa "
                   "tăng trưởng GDP, mức rủi ro chấp nhận được và tính bao trùm xã hội.")
        policy_box("<b>Cảnh báo & khuyến nghị tự động:</b> "
                   "• S1 (Truyền thống) — rủi ro thấp nhưng động lực số yếu, khó đạt 30% KTS/GDP 2030. "
                   "• S3 (AI dẫn dắt) — GDP & số hoá cao nhưng rủi ro cyber/bất bình đẳng cao → cần đệm H. "
                   "• S5 (Tối ưu cân bằng) — bám sát Nghị quyết 57-NQ/TW: tăng trưởng đi cùng bao trùm & an toàn.")
        st.caption("Tiêu chí đánh giá đồ án (Phụ lục F2): mô hình 20% · mã nguồn 20% · dữ liệu VN 15% · "
                   "phân tích chính sách 20% · dashboard 15% · báo cáo 10%.")


# ============================================================================
# BỘ ĐỊNH TUYẾN
# ============================================================================
ROUTER = {
    PAGES[0]: render_home, PAGES[1]: render_bai1, PAGES[2]: render_bai2,
    PAGES[3]: render_bai3, PAGES[4]: render_bai4, PAGES[5]: render_bai5,
    PAGES[6]: render_bai6, PAGES[7]: render_bai7, PAGES[8]: render_bai8,
    PAGES[9]: render_bai9, PAGES[10]: render_bai10, PAGES[11]: render_bai11,
    PAGES[12]: render_bai12,
}
ROUTER.get(choice, render_home)()

st.divider()
st.caption("AIDEOM-VN · Bài tập lớn Các mô hình ra quyết định · Tạ Tuấn Chinh — 23051191 · "
           "Dữ liệu Việt Nam 2020-2025 · github.com/anoreo07/AIDEOMVN")
