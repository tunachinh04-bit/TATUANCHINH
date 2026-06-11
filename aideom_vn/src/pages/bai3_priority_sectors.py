import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def norm_good(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

def norm_bad(x):
    return (x.max() - x) / (x.max() - x.min() + 1e-12)

def render():
    st.title("📊 Bài 3 — Tính chỉ số ưu tiên ngành Priority cho 10 ngành Việt Nam")
    
    st.markdown("""
    **Mục tiêu học tập:** Sinh viên nắm vững phương pháp chuẩn hóa dữ liệu đa tiêu chí, 
    xây dựng được chỉ số tổng hợp có trọng số ($Priority$) để hỗ trợ ra quyết định lựa chọn ngành 
    ưu tiên trong chiến lược chuyển đổi số quốc gia.
    """)

    tabs = st.tabs([
        "📖 Bối cảnh & Vấn đề", 
        "🔬 Lý thuyết tiêu chí", 
        "🛠️ Phương pháp chuẩn hóa", 
        "📊 Dữ liệu 10 ngành", 
        "📈 Xếp hạng ưu tiên", 
        "🔥 Phân tích độ nhạy",
        "💡 Thảo luận chính sách",
        "📚 Tham khảo"
    ])
    
    with tabs[0]:
        st.header("1. Bối cảnh & Vấn đề")
        st.markdown("""
        Theo cơ cấu kinh tế năm 2024, Việt Nam đối diện câu hỏi: trong số các ngành lớn, ngành nào nên được ưu tiên 
        đẩy mạnh chuyển đổi số và ứng dụng AI trước để tạo hiệu ứng lan tỏa tối đa? 
        
        Với ngân sách có hạn, Chính phủ không thể đầu tư dàn trải. Cần một chỉ số định lượng ($Priority$) để xác định 
        danh sách "mũi nhọn".
        """)
        st.info("💡 **Vấn đề:** Ngành nào xứng đáng nhận 100 tỷ VND đầu tư AI đầu tiên?")
        
    with tabs[1]:
        st.header("2. Chỉ số Ưu tiên Ngành ($Priority_i$)")
        st.latex(r"Priority_i = \sum_{j=1}^{6} a_j \cdot \tilde{x}_{ij} - a_7 \cdot \tilde{Risk}_i")
        
        st.markdown("""
        **Các tiêu chí đánh giá ($x_j$):**
        1. **Growth** ($x_1$): Tốc độ tăng trưởng GDP của ngành (%).
        2. **Productivity** ($x_2$): Năng suất lao động (tr.VND/LĐ).
        3. **Spillover** ($x_3$): Hệ số lan tỏa tới các ngành khác (0-1).
        4. **Export** ($x_4$): Kim ngạch xuất khẩu (tỷ USD).
        5. **Employment** ($x_5$): Số lượng việc làm thu hút (tr.LĐ).
        6. **AI Readiness** ($x_6$): Mức độ sẵn sàng ứng dụng AI (0-100).
        7. **Risk** ($x_7$): Rủi ro bị thay thế bởi tự động hóa - Đây là **tiêu chí âm** (Cost criterion).
        """)
        
    with tabs[2]:
        st.header("3. Phương pháp chuẩn hóa Min-Max")
        st.markdown("""
        Do các tiêu chí có đơn vị đo khác nhau (%, tỷ USD, triệu người), ta cần chuẩn hóa về cùng thang [0, 1].
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Tiêu chí Lợi ích (Benefit):**")
            st.latex(r"\tilde{x}_{ij} = \frac{x_{ij} - \min(x_j)}{\max(x_j) - \min(x_j)}")
            st.caption("Áp dụng cho x1, x2, x3, x4, x5, x6")
            
        with col2:
            st.markdown("**Tiêu chí Rủi ro/Chi phí (Cost):**")
            st.latex(r"\tilde{x}_{ij} = \frac{\max(x_j) - x_{ij}}{\max(x_j) - \min(x_j)}")
            st.caption("Áp dụng cho x7 (Càng thấp càng ưu tiên)")

    with tabs[3]:
        st.header("4. Dữ liệu 10 ngành Việt Nam (2024)")
        sectors_data = {
            "Ngành": ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Khai khoáng", "Bán buôn-bán lẻ", "Tài chính-Ngân hàng", "Logistics-Vận tải", "CNTT-Truyền thông", "Giáo dục-Đào tạo", "Y tế"],
            "Growth": [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
            "Productivity": [103.4, 241.2, 168.8, 1290.5, 145.3, 1072.4, 321.4, 713.8, 205.7, 437.1],
            "Spillover": [0.35, 0.78, 0.42, 0.30, 0.55, 0.85, 0.72, 0.92, 0.65, 0.60],
            "Export": [40.5, 290.9, 2.5, 8.2, 5.5, 1.2, 3.1, 178.0, 0.0, 0.0],
            "Employment": [13.20, 11.50, 4.80, 0.30, 7.80, 0.55, 1.95, 0.62, 2.15, 0.75],
            "AI_Ready": [15, 55, 20, 30, 48, 72, 42, 88, 38, 45],
            "Risk": [18, 42, 25, 55, 38, 52, 35, 28, 22, 18]
        }
        df_sec = pd.DataFrame(sectors_data)
        st.dataframe(df_sec.style.format(precision=2), use_container_width=True)
        st.caption("Dữ liệu mô phỏng dựa trên Niên giám Thống kê 2024")

    with tabs[4]:
        st.header("5. Xếp hạng ưu tiên theo trọng số chính sách")
        
        st.sidebar.markdown("### Trọng số Tiêu chí (a_j)")
        a = [
            st.sidebar.slider("a1: Growth", 0.0, 0.5, 0.15),
            st.sidebar.slider("a2: Productivity", 0.0, 0.5, 0.15),
            st.sidebar.slider("a3: Spillover", 0.0, 0.5, 0.20),
            st.sidebar.slider("a4: Export", 0.0, 0.5, 0.15),
            st.sidebar.slider("a5: Employment", 0.0, 0.5, 0.10),
            st.sidebar.slider("a6: AI Readiness", 0.0, 0.5, 0.20),
            st.sidebar.slider("a7: Risk (Penalty)", 0.0, 0.5, 0.15)
        ]
        
        # Calculate scores
        cols_benefit = ["Growth", "Productivity", "Spillover", "Export", "Employment", "AI_Ready"]
        df_norm = df_sec.copy()
        for c in cols_benefit:
            df_norm[c] = (df_sec[c] - df_sec[c].min()) / (df_sec[c].max() - df_sec[c].min() + 1e-9)
        df_norm["Risk"] = (df_sec["Risk"].max() - df_sec["Risk"]) / (df_sec["Risk"].max() - df_sec["Risk"].min() + 1e-9)
        
        w = np.array(a) / sum(a)
        df_sec['Priority_Score'] = df_norm[cols_benefit + ["Risk"]].values @ w
        df_sec = df_sec.sort_values('Priority_Score', ascending=False)
        
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.metric("Top 1", df_sec['Ngành'].iloc[0])
            st.dataframe(df_sec[['Ngành', 'Priority_Score']].style.format(subset=['Priority_Score'], precision=3), use_container_width=True)
            
        with col2:
            fig = px.bar(df_sec, x='Priority_Score', y='Ngành', orientation='h', color='Priority_Score',
                         color_continuous_scale='Viridis', title="Bản đồ Ưu tiên Ngành")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.header("6. Phân tích độ nhạy (Weight Sensitivity)")
        st.markdown("""
        Phân tích độ nhạy giúp đánh giá tính vững chãi (robustness) của quyết định chính sách. 
        Tại đây, chúng ta biến đổi trọng số AI Readiness ($a_6$) từ 0 đến 0.5 (renormalize tổng bằng 1) 
        để quan sát sự thay đổi thứ hạng của 10 ngành kinh tế.
        """)
        
        ai_weights = np.linspace(0, 0.5, 11)
        rankings = []
        for aw in ai_weights:
            a_temp = list(a)
            a_temp[5] = aw
            w_temp = np.array(a_temp) / sum(a_temp)
            scores = df_norm[cols_benefit + ["Risk"]].values @ w_temp
            rankings.append(scores)
            
        rank_df = pd.DataFrame(np.array(rankings).T, columns=[f"{aw:.2f}" for aw in ai_weights])
        rank_df.index = df_sec.sort_index()['Ngành'] # Use original index for matching order
        
        fig_heat = px.imshow(rank_df, labels=dict(x="Trọng số AI", y="Ngành", color="Điểm"),
                             title="Sự thay đổi điểm số khi thay đổi ưu tiên AI")
        st.plotly_chart(fig_heat, use_container_width=True)

    with tabs[6]:
        st.header("7. Thảo luận chính sách")
        st.markdown(r"""
        Phân tích kết quả xếp hạng đa tiêu chí gợi mở một số hàm ý chính sách quan trọng:
        
        - **Ngành CNTT-Truyền thông & Công nghiệp chế biến chế tạo**: Luôn nằm trong top ưu tiên đầu tư số và AI. Kết quả này hoàn toàn đồng nhất với định hướng của **Nghị quyết số 52-NQ/TW** ngày 27/9/2019 của Bộ Chính trị về chủ động tham gia Cách mạng công nghiệp lần thứ tư, trong đó xác định hạ tầng số và phát triển các ngành công nghiệp công nghệ cao là nhiệm vụ mũi nhọn.
        - **Sự nghịch lý của ngành Khai khoáng**: Ngành Khai khoáng sở hữu năng suất lao động danh nghĩa rất cao (do tính chất thâm dụng vốn và tài nguyên), nhưng điểm ưu tiên thấp do hệ số lan tỏa kinh tế thấp và rủi ro tự động hóa cao. Điều này chứng minh rằng nếu chỉ nhìn vào các chỉ số đơn lẻ như năng suất lao động, chính sách phân bổ ngân sách dễ bị lệch lạc.
        - **Vấn đề an sinh xã hội**: Các ngành có tỷ trọng lao động lớn nhưng sẵn sàng công nghệ thấp như Nông nghiệp đòi hỏi bộ trọng số chính sách hướng tới "tính bao trùm" ($a_5$ cao hơn) để tránh phân kỳ phát triển xã hội.
        - **Quản trị và Governance**: Trọng số chính sách không nên là kết quả áp đặt chủ quan của chuyên gia kỹ thuật, mà cần được thảo luận đa bên giữa các bộ ngành để phản ánh đúng định hướng của Chính phủ.
        """)
        
    with tabs[7]:
        st.header("Tham khảo")
        st.markdown(r"""
        - Niên giám thống kê Việt Nam 2024 (Cục Thống kê quốc gia).
        - **Nghị quyết số 52-NQ/TW** ngày 27/9/2019 của Bộ Chính trị về một số chủ trương, chính sách chủ động tham gia cuộc Cách mạng công nghiệp lần thứ tư.
        - **Nghị quyết số 57-NQ/TW** về định hướng chiến lược phát triển công nghiệp quốc gia đến năm 2030, tầm nhìn đến năm 2045.
        - **Quyết định số 749/QĐ-TTg** phê duyệt "Chương trình Chuyển đổi số quốc gia đến năm 2025, định hướng đến năm 2030".
        - Oxford Insights (2024): Government AI Readiness Index.
        """)

if __name__ == "__main__":
    render()
