import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from data_loader import load_regions
except ImportError:
    def load_regions():
        return pd.DataFrame({
            "region_id": [1, 2, 3, 4, 5, 6],
            "region_name_vi": [
                "Trung du miền núi phía Bắc",
                "Đồng bằng sông Hồng",
                "Bắc Trung Bộ và Duyên hải miền Trung",
                "Tây Nguyên",
                "Đông Nam Bộ",
                "Đồng bằng sông Cửu Long"
            ],
            "grdp_per_capita_million_VND": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
            "fdi_registered_billion_USD": [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
            "digital_index_0_100": [38, 78, 55, 32, 82, 48],
            "ai_readiness_0_100": [22, 68, 40, 18, 75, 30],
            "trained_labor_pct": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
            "rd_intensity_pct": [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
            "internet_penetration_pct": [72, 92, 84, 68, 94, 78],
            "gini_coef": [0.405, 0.358, 0.372, 0.412, 0.385, 0.392]
        })

# Implementation of TOPSIS from scratch
def run_topsis(X, w, is_benefit):
    # Vector Normalization
    norm_denom = np.sqrt(np.sum(X ** 2, axis=0))
    norm_denom = np.where(norm_denom == 0, 1e-12, norm_denom)
    R = X / norm_denom
    
    # Weighted normalization
    V = R * w
    
    # Positive and Negative Ideal Solutions
    pis = np.zeros(X.shape[1])
    nis = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        if is_benefit[j]:
            pis[j] = np.max(V[:, j])
            nis[j] = np.min(V[:, j])
        else:
            pis[j] = np.min(V[:, j])
            nis[j] = np.max(V[:, j])
            
    # Euclidean distances
    d_plus = np.sqrt(np.sum((V - pis) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((V - nis) ** 2, axis=1))
    
    # Closeness coefficient
    scores = d_minus / (d_plus + d_minus + 1e-12)
    ranks = np.argsort(np.argsort(-scores)) + 1
    return scores, ranks

# Objective Entropy Weights
def calculate_entropy_weights(X):
    P = X / (X.sum(axis=0) + 1e-12)
    k = 1.0 / np.log(X.shape[0]) if X.shape[0] > 1 else 1.0
    E = -k * np.sum(P * np.log(P + 1e-12), axis=0)
    d = 1.0 - E
    w = d / (np.sum(d) + 1e-12)
    return w

def render():
    st.title("🏆 Bài 6 — TOPSIS: Xếp hạng 6 vùng kinh tế Việt Nam theo mức độ ưu tiên đầu tư AI")
    
    st.markdown("""
    **Mục tiêu học tập:** Sinh viên thành thạo kỹ thuật ra quyết định đa tiêu chí (MCDM) - TOPSIS, 
    biết cách chuẩn hóa ma trận quyết định, tính toán khoảng cách đến phương án lý tưởng tốt/xấu, 
    và áp dụng phương pháp Entropy để xác định trọng số khách quan từ dữ liệu.
    """)

    tabs = st.tabs([
        "📖 Bối cảnh & Vấn đề", 
        "🔬 Lý thuyết TOPSIS", 
        "🛠️ Thuật toán Entropy", 
        "📊 Dữ liệu 6 Vùng", 
        "📈 Kết quả xếp hạng", 
        "🔥 Phân tích độ nhạy",
        "💡 Thảo luận chính sách",
        "📚 Tham khảo"
    ])
    
    with tabs[0]:
        st.header("1. Bối cảnh & Vấn đề")
        st.markdown("""
        Theo Quyết định số 127/QĐ-TTg ngày 26/01/2021 về Chiến lược quốc gia về nghiên cứu, phát triển và ứng dụng AI đến năm 2030, 
        Việt Nam đặt mục tiêu trở thành trung tâm AI của ASEAN. Tuy nhiên, ngân sách nhà nước đầu tư công nghệ luôn giới hạn. 
        
        **Câu hỏi đặt ra:** Nên lựa chọn vùng kinh tế nào để ưu tiên đặt các trung tâm tính toán hiệu năng cao và sandbox dữ liệu trước? 
        
        Bài tập này hướng dẫn áp dụng phương pháp TOPSIS để xếp hạng **6 vùng kinh tế - xã hội** của Việt Nam dựa trên 8 chỉ số kinh tế số.
        """)
        st.info("💡 **Phương pháp**: TOPSIS tìm phương án có khoảng cách hình học ngắn nhất đến giải pháp lý tưởng tốt nhất, và xa nhất khỏi giải pháp tồi nhất.")
        
    with tabs[1]:
        st.header("2. Lý thuyết toán học TOPSIS")
        st.markdown("Quy trình áp dụng phương pháp TOPSIS bao gồm 5 bước chính:")
        
        st.latex(r"r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^n x_{ij}^2}} \quad \text{(Bước 1: Chuẩn hóa vector)}")
        st.latex(r"v_{ij} = w_j \cdot r_{ij} \quad \text{(Bước 2: Xây dựng ma trận có trọng số)}")
        st.latex(r"A^* = \{v_1^*, \dots, v_m^*\}, \quad A^- = \{v_1^-, \dots, v_m^-\} \quad \text{(Bước 3: Xác định lý tưởng tốt/xấu)}")
        st.latex(r"S_i^* = \sqrt{\sum_{j=1}^m (v_{ij} - v_j^*)^2}, \quad S_i^- = \sqrt{\sum_{j=1}^m (v_{ij} - v_j^-)^2} \quad \text{(Bước 4: Tính khoảng cách Euclide)}")
        st.latex(r"C_i^* = \frac{S_i^-}{S_i^* + S_i^-} \quad \text{(Bước 5: Hệ số gần gũi tương đối)}")
        
        st.markdown("Thứ hạng các phương án được xác định bằng cách sắp xếp $C_i^*$ giảm dần (càng gần 1 càng ưu tiên).")

    with tabs[2]:
        st.header("3. Phương pháp trọng số Entropy")
        st.markdown("""
        Bên cạnh trọng số chuyên gia (chủ quan), phương pháp **Entropy** cho phép xác định trọng số khách quan 
        dựa trên mức độ biến động (độ phân kỳ) của dữ liệu. 
        Nếu dữ liệu của một tiêu chí phân tán mạnh, tiêu chí đó chứa nhiều thông tin và cần được gán trọng số cao hơn.
        """)
        st.latex(r"p_{ij} = \frac{x_{ij}}{\sum_{i=1}^n x_{ij}} \quad \text{(Chuẩn hóa tỉ lệ)}")
        st.latex(r"E_j = -\frac{1}{\ln(n)} \sum_{i=1}^n p_{ij} \ln(p_{ij}) \quad \text{(Entropy của tiêu chí j)}")
        st.latex(r"d_j = 1 - E_j \quad \text{(Độ phân kỳ)}")
        st.latex(r"w_j = \frac{d_j}{\sum_{k=1}^m d_k} \quad \text{(Trọng số Entropy)}")

    with tabs[3]:
        st.header("4. Dữ liệu kinh tế - xã hội 6 Vùng Việt Nam")
        st.markdown("Dữ liệu nạp từ tệp `vietnam_regions_2024.csv`:")
        
        df_regions = load_regions()
        st.dataframe(df_regions.style.highlight_max(axis=0, color='#10b981'), use_container_width=True)
        st.caption("Nguồn: vietnam_regions_2024.csv. Tất cả 7 chỉ số đầu là Lợi ích (+), chỉ số Gini là Chi phí (-) (càng thấp càng tốt).")

    with tabs[4]:
        st.header("5. Kết quả xếp hạng TOPSIS")
        
        criteria = [
            'grdp_per_capita_million_VND', 'fdi_registered_billion_USD',
            'digital_index_0_100', 'ai_readiness_0_100',
            'trained_labor_pct', 'rd_intensity_pct',
            'internet_penetration_pct', 'gini_coef'
        ]
        is_benefit = [True, True, True, True, True, True, True, False]
        X = df_regions[criteria].values.astype(float)
        
        w_expert = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
        w_entropy = calculate_entropy_weights(X)
        
        w_mode = st.radio("Lựa chọn phương thức xác định trọng số:", ["Trọng số Chuyên gia (Đề xuất)", "Trọng số Khách quan (Entropy)"])
        
        if w_mode == "Trọng số Chuyên gia (Đề xuất)":
            w = w_expert
        else:
            w = w_entropy
            
        scores, ranks = run_topsis(X, w, is_benefit)
        df_res = df_regions.copy()
        df_res['Điểm TOPSIS (C*)'] = scores
        df_res['Thứ hạng'] = ranks
        df_res = df_res.sort_values('Thứ hạng')
        
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.dataframe(df_res[['Thứ hạng', 'region_name_vi', 'Điểm TOPSIS (C*)']].style.format({'Điểm TOPSIS (C*)': '{:.4f}'}), use_container_width=True)
            
            # Show weights used
            df_w = pd.DataFrame({
                "Tiêu chí": [
                    "GRDP/người", "FDI đăng ký", "Digital Index", "AI Readiness",
                    "Lao động qua ĐT", "Cường độ R&D", "Tỷ lệ Internet", "Hệ số Gini"
                ],
                "Trọng số": w
            })
            st.markdown("**Bộ trọng số áp dụng:**")
            st.dataframe(df_w.style.format({'Trọng số': '{:.2%}'}), use_container_width=True)
            
        with col2:
            fig = px.bar(df_res, x='Điểm TOPSIS (C*)', y='region_name_vi', orientation='h', color='Điểm TOPSIS (C*)',
                         color_continuous_scale='Viridis', title="Kết quả xếp hạng các Vùng")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.header("6. Phân tích độ nhạy của trọng số AI ($w_{AI}$)")
        st.markdown("""
        Tại đây, chúng ta biến đổi trọng số AI Readiness ($w_{AI}$) từ 0.10 đến 0.40 để kiểm tra 
        tính ổn định của vị trí xếp hạng top-3. Các trọng số tiêu chí khác sẽ được chuẩn hóa tỷ lệ lại tự động.
        """)
        
        w_ai_range = np.linspace(0.10, 0.40, 7)
        rank_matrix = []
        for w_ai in w_ai_range:
            w_new = w_expert.copy()
            w_new[3] = w_ai
            other_indices = [i for i in range(8) if i != 3]
            other_sum = w_expert[other_indices].sum()
            w_new[other_indices] = w_expert[other_indices] * ((1.0 - w_ai) / other_sum)
            w_new = w_new / w_new.sum()
            
            _, r_temp = run_topsis(X, w_new, is_benefit)
            rank_matrix.append(r_temp)
            
        df_sens = pd.DataFrame(np.array(rank_matrix).T, columns=[f"{val:.2f}" for val in w_ai_range])
        df_sens.index = df_regions['region_name_vi']
        
        fig_heat = px.imshow(df_sens, labels=dict(x="Trọng số AI", y="Vùng", color="Thứ hạng"),
                             x=[f"{val:.2f}" for val in w_ai_range],
                             y=df_regions['region_name_vi'],
                             color_continuous_scale="Reds_r",
                             title="Sự biến động thứ hạng các Vùng theo trọng số AI")
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Chú thích: Màu đậm hơn tương ứng với thứ hạng cao hơn (hạng 1, 2).")

    with tabs[6]:
        st.header("7. Thảo luận chính sách")
        st.markdown(r"""
Học sinh thảo luận về các kết quả tính toán định lượng:

- **Sự bứt phá của Đồng bằng sông Hồng và Đông Nam Bộ**: Đây là hai đầu tàu kinh tế dẫn đầu bảng xếp hạng trong mọi kịch bản trọng số nhờ quy mô kinh tế lớn, FDI vượt trội, hạ tầng công nghệ số và tỷ lệ nhân lực qua đào tạo rất cao. Quyết định đặt 3 trung tâm AI lớn của quốc gia tại miền Bắc (Hà Nội), miền Nam (TP. HCM) và miền Trung (Đà Nẵng thuộc Bắc Trung Bộ và Duyên hải miền Trung) hoàn toàn đồng nhất với kết quả xếp hạng.
- **Sự chuyển dịch của trọng số Entropy**: Trọng số Entropy gán ưu tiên rất cao cho FDI và cường độ R&D do đây là hai tiêu chí có độ phân tán dữ liệu lớn nhất giữa các vùng miền. Tuy nhiên, việc phụ thuộc hoàn toàn vào trọng số khách quan có thể dẫn đến việc "bỏ quên" các khía cạnh an sinh như bình đẳng thu nhập (Gini) hay hạ tầng cơ sở ở vùng sâu vùng xa.
- **Tính tương quan tiêu chí**: Phương pháp TOPSIS giả định các tiêu chí độc lập. Trên thực tế, hạ tầng số (Digital Index) và AI Readiness tương quan cực kỳ cao. Để khắc phục, có thể áp dụng phương pháp khoảng cách Mahalanobis thay cho khoảng cách Euclide hoặc phân tích nhân tố (PCA) trước khi xếp hạng.
        """)
        
    with tabs[7]:
        st.header("8. Tham khảo")
        st.markdown(r"""
        - Quyết định số 127/QĐ-TTg ngày 26/01/2021 về Chiến lược quốc gia về nghiên cứu, phát triển và ứng dụng AI đến năm 2030.
        - **Hwang, C. L., & Yoon, K. (1981):** Multiple Attribute Decision Making: Methods and Applications.
        - Báo cáo chỉ số chuyển đổi số DTI các tỉnh thành năm 2024 (Bộ Thông tin - Truyền thông).
        - **Shannon, C. E. (1948):** A Mathematical Theory of Communication.
        """)

if __name__ == "__main__":
    render()
