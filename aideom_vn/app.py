import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pages import (
    bai1_cobb_douglas,
    bai2_lp_budget,
    bai3_priority_sectors,
    bai4_lp_region_budget,
    bai5_mip_project_selection,
    bai6_topsis_ai_regions,
    bai7_multi_objective_nsga2,
    bai8_dynamic_forecast,
    bai9_labor_simulation,
    bai10_stochastic_budget,
    bai11_rl_policy,
    bai12_integrated
)
# Note: Other modules will be added as they are implemented

st.set_page_config(
    page_title="AIDEOM-VN Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Simplified from original)
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🇻🇳 AIDEOM-VN")
st.sidebar.caption("Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI")

menu = st.sidebar.radio(
    "Chọn bài phân tích:",
    [
        "🏠 Trang chủ",
        "🌱 Bài 1 — Cobb-Douglas + AI",
        "💰 Bài 2 — LP ngân sách số",
        "📊 Bài 3 — Priority 10 ngành",
        "🗺️ Bài 4 — LP ngành-vùng",
        "🎯 Bài 5 — MIP 15 dự án",
        "🏆 Bài 6 — TOPSIS 6 vùng",
        "🌐 Bài 7 — NSGA-II Pareto",
        "⏳ Bài 8 — Động 2026-2035",
        "👷 Bài 9 — Lao động & AI",
        "🎲 Bài 10 — Stochastic SP",
        "🤖 Bài 11 — Q-learning RL",
        "🧠 Bài 12 — AIDEOM tích hợp",
    ]
)

if menu == "🏠 Trang chủ":
    st.title("AIDEOM-VN: Vietnam Economic Decision Support System")
    st.image("https://img.freepik.com/free-vector/digital-economy-concept-illustration_114360-7856.jpg", width=600)
    st.markdown("""
    Chào mừng bạn đến với hệ thống hỗ trợ ra quyết định kinh tế số Việt Nam.
    Dựa trên 12 bài thực hành tối ưu hóa và học máy trong kinh tế.
    
    ### Hướng dẫn sử dụng:
    1. Chọn một **Bài phân tích** từ thanh bên trái.
    2. Mỗi bài được chia thành các phần: Bối cảnh, Lý thuyết, Mô hình, Kết quả và Thảo luận.
    3. Bạn có thể thay đổi các tham số trong thanh bên để quan sát sự thay đổi của kết quả.
    """)

elif menu == "🌱 Bài 1 — Cobb-Douglas + AI":
    bai1_cobb_douglas.render()

elif menu == "💰 Bài 2 — LP ngân sách số":
    bai2_lp_budget.render()

elif menu == "📊 Bài 3 — Priority 10 ngành":
    bai3_priority_sectors.render()

elif menu == "🗺️ Bài 4 — LP ngành-vùng":
    bai4_lp_region_budget.render()

elif menu == "🎯 Bài 5 — MIP 15 dự án":
    bai5_mip_project_selection.render()

elif menu == "🏆 Bài 6 — TOPSIS 6 vùng":
    bai6_topsis_ai_regions.render()

elif menu == "🌐 Bài 7 — NSGA-II Pareto":
    bai7_multi_objective_nsga2.render()

elif menu == "⏳ Bài 8 — Động 2026-2035":
    bai8_dynamic_forecast.render()

elif menu == "👷 Bài 9 — Lao động & AI":
    bai9_labor_simulation.render()

elif menu == "🎲 Bài 10 — Stochastic SP":
    bai10_stochastic_budget.render()

elif menu == "🤖 Bài 11 — Q-learning RL":
    bai11_rl_policy.render()

elif menu == "🧠 Bài 12 — AIDEOM tích hợp":
    bai12_integrated.render()

else:
    st.warning(f"Module '{menu}' đang được phát triển nội dung dựa trên tài liệu bài tập.")
    st.info("Vui lòng chọn Bài 1 hoặc Bài 12 để xem bản demo.")

# Footer
st.sidebar.divider()
st.sidebar.caption("© 2026 AIDEOM-VN Project")
