import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def render():
    st.title("🌐 Bài 7 — Tối ưu hóa đa mục tiêu Pareto với NSGA-II / Weighted-Sum LP")
    
    st.markdown("""
    **Mục tiêu học tập:** Sinh viên hiểu sự khác biệt giữa tối ưu đơn mục tiêu và đa mục tiêu, 
    xây dựng được bài toán 4 mục tiêu xung đột (tăng trưởng GDP, công bằng vùng miền, phát thải CO2, và an ninh dữ liệu), 
    giải quyết để trích xuất mặt trận Pareto, và sử dụng phương pháp TOPSIS để chọn nghiệm thỏa hiệp chính sách.
    """)

    tabs = st.tabs([
        "📖 Bối cảnh & Vấn đề", 
        "🔬 Lý thuyết Pareto", 
        "🧠 Cấu trúc 4 Mục tiêu", 
        "📈 Không gian Pareto 3D", 
        "🔥 So sánh các kịch bản",
        "💡 Thảo luận chính sách",
        "📚 Tham khảo"
    ])
    
    with tabs[0]:
        st.header("1. Bối cảnh & Vấn đề")
        st.markdown("""
        Mục 8.2 của bài báo nguồn nhấn mạnh rằng kết quả tối ưu hóa phát triển kinh tế vĩ mô trong kỷ nguyên số 
        không nhất thiết phải là một nghiệm duy nhất, mà là một tập nghiệm **Pareto**. 
        Các mục tiêu quốc gia luôn có sự đánh đổi (trade-offs). Ví dụ:
        
        - Để tối đa hóa GDP nhanh, ta tập trung đầu tư AI vào hai vùng đầu tàu là Đông Nam Bộ và ĐBS Hồng.
        - Nhưng để giảm bất bình đẳng (bao trùm), ta phải phân bổ về các vùng khó khăn hơn (Tây Nguyên, Miền núi phía Bắc).
        - Ngoài ra, các siêu trung tâm dữ liệu AI tiêu thụ năng lượng lớn làm tăng phát thải khí nhà kính (xung đột mục tiêu Net Zero).
        """)
        st.info("💡 **Thách thức**: Thiết kế chính sách phân bổ ngân sách 50.000 tỷ VND (Bài 4) để cân bằng cả 4 mục tiêu kinh tế - xã hội - môi trường - an ninh.")
        
    with tabs[1]:
        st.header("2. Lý thuyết Tối ưu hóa Pareto")
        st.markdown("""
        Trong tối ưu hóa đa mục tiêu, một phương án $x$ được gọi là **lấn át** (dominate) phương án $y$ nếu nó tốt hơn $y$ ở ít nhất một mục tiêu 
        và không kém hơn $y$ ở tất cả các mục tiêu khác.
        
        **Mặt trận Pareto (Pareto Front)** là tập hợp các phương án không bị lấn át bởi bất kỳ phương án nào khác trong không gian tìm kiếm.
        """)
        
        st.latex(r"\min F(x) = [f_1(x), f_2(x), f_3(x), f_4(x)]^T \quad \text{s.t.} \quad x \in \Omega")
        
        st.markdown("""
        Để giải bài toán này và trích xuất mặt trận Pareto mà không cần cài đặt các thư viện tiến hóa nặng nề như `pymoo`, 
        chúng ta sử dụng phương pháp **Tổng trọng số (Weighted-Sum Method)** giải liên tiếp các bài toán Quy hoạch tuyến tính (LP) bằng PuLP:
        """)
        st.latex(r"\min_{x \in \Omega} \sum_{k=1}^4 W_k \cdot \bar{f}_k(x) \quad \text{với} \quad W_k \ge 0, \sum W_k = 1")

    with tabs[2]:
        st.header("3. Đặc tả 4 Hàm mục tiêu")
        st.markdown("Quyết định $x_{j,r}$ là phân bổ ngân sách cho hạng mục $j \\in \\{I, D, AI, H\\}$ tại vùng $r \\in \\{1..6\\}$.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**1. GDP Gain ($f_1$ - Tối đa hóa):**")
            st.latex(r"f_1(x) = \sum_{r=1}^6 \sum_{j=1}^4 \beta_{j,r} \cdot x_{j,r}")
            st.caption("β là hệ số tác động biên của Bài 4.")
            
            st.markdown("**2. Bất bình đẳng vùng ($f_2$ - Tối thiểu hóa):**")
            st.latex(r"f_2(x) = \frac{1}{6} \sum_{r=1}^6 |S_r - S_{mean}|")
            st.caption("S_r là tổng ngân sách vùng r. Được tuyến tính hóa trong LP.")
            
        with col2:
            st.markdown("**3. Phát thải CO2 gián tiếp ($f_3$ - Tối thiểu hóa):**")
            st.latex(r"f_3(x) = \sum_{r=1}^6 e_r \cdot (x_{I,r} + x_{AI,r})")
            st.caption("e_r là hệ số phát thải năng lượng vùng r.")
            
            st.markdown("**4. Rủi ro An ninh mạng ròng ($f_4$ - Tối thiểu hóa):**")
            st.latex(r"f_4(x) = \sum_{r=1}^6 \rho_r \cdot x_{AI,r} - \sum_{r=1}^6 \sigma_r \cdot x_{H,r}")
            st.caption("ρ_r là hệ số rủi ro AI, σ_r là hệ số giảm thiểu của H.")

    with tabs[3]:
        st.header("4. Trực quan hóa Không gian Pareto 3D (Đã tối ưu hóa)")
        
        # We solve the LP for many weights to generate the true Pareto front
        try:
            import pulp
            
            # Param từ đề bài
            beta_vals = [
                [1.15, 0.85, 0.55, 1.30],  # NMM
                [0.95, 1.25, 1.40, 1.05],  # RRD
                [1.05, 0.95, 0.85, 1.15],  # NCC
                [1.20, 0.75, 0.45, 1.35],  # CH
                [0.90, 1.30, 1.55, 1.00],  # SE
                [1.10, 0.85, 0.65, 1.25],  # MD
            ]
            d0 = [38, 78, 55, 32, 82, 48]
            e_vals = [0.42, 0.55, 0.48, 0.32, 0.62, 0.38]
            rho_vals = [0.18, 0.45, 0.28, 0.12, 0.52, 0.22]
            sig_vals = [0.32, 0.28, 0.30, 0.35, 0.25, 0.30]
            
            # Generate weights on the simplex (200 points)
            np.random.seed(42)
            n_samples = 120
            W_samples = np.random.dirichlet([1.0, 1.0, 1.0, 1.0], size=n_samples)
            
            pareto_data = []
            
            # Solve LP loop
            for idx, W in enumerate(W_samples):
                prob = pulp.LpProblem(f"Pareto_LP_{idx}", pulp.LpMinimize)
                
                # Variables
                x = pulp.LpVariable.dicts("x", (range(6), range(4)), lowBound=0)
                d = pulp.LpVariable.dicts("d", range(6), lowBound=0) # auxiliary for MAD
                
                # S_mean = 50000 / 6
                S_mean = 8333.33
                
                # Constraints
                # C1: Budget
                prob += pulp.lpSum(x[r][j] for r in range(6) for j in range(4)) <= 50000
                for r in range(6):
                    # C2: Sàn
                    prob += pulp.lpSum(x[r][j] for j in range(4)) >= 5000
                    # C3: Trần
                    prob += pulp.lpSum(x[r][j] for j in range(4)) <= 12000
                # C4: Sàn nhân lực
                prob += pulp.lpSum(x[r][3] for r in range(6)) >= 12000
                
                # C5: Ràng buộc công bằng vùng (digital index)
                gamma = 0.002
                for r in range(6):
                    d_final_r = d0[r] + gamma * x[r][1]
                    for k in range(6):
                        d_final_k = d0[k] + gamma * x[k][1]
                        prob += d_final_r >= 0.7 * d_final_k
                        
                # MAD linearization
                for r in range(6):
                    prob += d[r] >= pulp.lpSum(x[r][j] for j in range(4)) - S_mean
                    prob += d[r] >= S_mean - pulp.lpSum(x[r][j] for j in range(4))
                    
                # Objectives
                f1_val = pulp.lpSum(beta_vals[r][j] * x[r][j] for r in range(6) for j in range(4))
                f2_val = pulp.lpSum(d[r] for r in range(6)) / 6
                f3_val = pulp.lpSum(e_vals[r] * (x[r][0] + x[r][2]) for r in range(6))
                f4_val = pulp.lpSum(rho_vals[r] * x[r][2] for r in range(6)) - pulp.lpSum(sig_vals[r] * x[r][3] for r in range(6))
                
                # Normalization scalars
                n_f1, n_f2, n_f3, n_f4 = 60000.0, 3000.0, 20000.0, 15000.0
                
                # Objective
                prob += W[0] * (-f1_val / n_f1) + W[1] * (f2_val / n_f2) + W[2] * (f3_val / n_f3) + W[3] * (f4_val / n_f4)
                
                prob.solve(pulp.PULP_CBC_CMD(msg=0))
                
                if pulp.LpStatus[prob.status] == 'Optimal':
                    pareto_data.append({
                        'GDP_Gain': pulp.value(f1_val),
                        'Inequality': pulp.value(f2_val),
                        'Emission': pulp.value(f3_val),
                        'Net_Risk': pulp.value(f4_val),
                        'W1': W[0], 'W2': W[1], 'W3': W[2], 'W4': W[3]
                    })
            
            df_p = pd.DataFrame(pareto_data)
            
            # Apply TOPSIS on Pareto set (7.4.3 weights: 0.40, 0.25, 0.20, 0.15)
            # Normalize Pareto objectives to [0, 1]
            Y_pareto = df_p[['GDP_Gain', 'Inequality', 'Emission', 'Net_Risk']].values.copy()
            # GDP_Gain is benefit (+), others are cost (-)
            # Standard vector normalization
            Y_norm = Y_pareto / np.sqrt(np.sum(Y_pareto ** 2, axis=0))
            w_topsis = np.array([0.40, 0.25, 0.20, 0.15])
            
            # Weight matrix
            V = Y_norm * w_topsis
            
            # Ideal solutions
            A_star = [np.max(V[:, 0]), np.min(V[:, 1]), np.min(V[:, 2]), np.min(V[:, 3])]
            A_neg = [np.min(V[:, 0]), np.max(V[:, 1]), np.max(V[:, 2]), np.max(V[:, 3])]
            
            S_star = np.sqrt(np.sum((V - A_star)**2, axis=1))
            S_neg = np.sqrt(np.sum((V - A_neg)**2, axis=1))
            C_star = S_neg / (S_star + S_neg)
            
            df_p['TOPSIS_Score'] = C_star
            df_p['Is_Compromise'] = (C_star == C_star.max())
            
            # Max GDP option
            df_p['Is_Max_GDP'] = (df_p['GDP_Gain'] == df_p['GDP_Gain'].max())
            
            # Plot 3D Scatter
            fig_3d = px.scatter_3d(df_p, x='GDP_Gain', y='Inequality', z='Emission',
                                   color='TOPSIS_Score', opacity=0.8,
                                   title="Mặt trận Pareto kinh tế vĩ mô Việt Nam (LP thực tế)",
                                   color_continuous_scale='Viridis',
                                   labels={'GDP_Gain': 'GDP Gain (Tỷ VND)', 'Inequality': 'Bao trùm (MAD)', 'Emission': 'Phát thải CO2'})
            st.plotly_chart(fig_3d, use_container_width=True)
            st.caption("Biểu đồ thể hiện mối quan hệ đánh đổi: Để tăng GDP Gain (trục X), ta phải chấp nhận phát thải cao hơn (trục Z) hoặc độ bất bình đẳng lớn hơn (trục Y).")
            
        except Exception as e:
            st.error(f"Lỗi tối ưu hóa đa mục tiêu: {e}")

    with tabs[4]:
        st.header("5. So sánh kịch bản và Phân tích chi phí cơ hội")
        
        # Parallel coordinates plot
        fig_par = px.parallel_coordinates(df_p, color="TOPSIS_Score",
                             dimensions=['GDP_Gain', 'Inequality', 'Emission', 'Net_Risk'],
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             labels={'GDP_Gain': 'GDP Gain', 'Inequality': 'Bất bình đẳng', 'Emission': 'Phát thải', 'Net_Risk': 'Rủi ro ròng'})
        st.plotly_chart(fig_par, use_container_width=True)
        
        # Opportunity cost analysis
        opt_compromise = df_p[df_p['Is_Compromise']].iloc[0]
        opt_max_gdp = df_p[df_p['Is_Max_GDP']].iloc[0]
        
        st.subheader("📊 Phân tích định lượng sự đánh đổi (Trade-off Analysis)")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Nghị quyết thỏa hiệp (Compromise Solution):**")
            st.write(f"- GDP Gain: **{opt_compromise['GDP_Gain']:,.1f}** tỷ VND")
            st.write(f"- Bất bình đẳng (MAD): **{opt_compromise['Inequality']:,.1f}** tỷ VND")
            st.write(f"- Phát thải CO2: **{opt_compromise['Emission']:,.1f}**")
            st.write(f"- Rủi ro dữ liệu ròng: **{opt_compromise['Net_Risk']:,.1f}**")
            
        with col_c2:
            st.markdown("**Nghiệm tăng trưởng tối đa (Max GDP Solution):**")
            st.write(f"- GDP Gain: **{opt_max_gdp['GDP_Gain']:,.1f}** tỷ VND")
            st.write(f"- Bất bình đẳng (MAD): **{opt_max_gdp['Inequality']:,.1f}** tỷ VND")
            st.write(f"- Phát thải CO2: **{opt_max_gdp['Emission']:,.1f}**")
            st.write(f"- Rủi ro dữ liệu ròng: **{opt_max_gdp['Net_Risk']:,.1f}**")
            
        st.divider()
        
        # Calculate percent changes
        gdp_diff_pct = (opt_max_gdp['GDP_Gain'] - opt_compromise['GDP_Gain']) / opt_compromise['GDP_Gain'] * 100
        ineq_diff_pct = (opt_max_gdp['Inequality'] - opt_compromise['Inequality']) / opt_compromise['Inequality'] * 100
        emis_diff_pct = (opt_max_gdp['Emission'] - opt_compromise['Emission']) / opt_compromise['Emission'] * 100
        
        st.markdown(f"""
        > [!IMPORTANT]
        > **Báo cáo chi phí cơ hội (Opportunity Cost Analysis):**
        > So với nghiệm thỏa hiệp, việc đuổi theo kịch bản tăng trưởng tối đa (Max GDP) giúp quốc gia tăng thêm **{gdp_diff_pct:.2f}%** về GDP Gain. 
        > Tuy nhiên, cái giá phải trả là sự gia tăng mạnh mẽ của bất bình đẳng vùng miền lên tới **{ineq_diff_pct:.2f}%** và lượng phát thải khí nhà kính tăng thêm **{emis_diff_pct:.2f}%**.
        """)

    with tabs[5]:
        st.header("6. Thảo luận chính sách")
        st.markdown(r"""
        Các kết quả tính toán định lượng trên biên Pareto cho thấy:
        
        - **Tính phi duy nhất của chính sách công**: Không tồn tại một nghiệm tối ưu duy nhất bao trùm. Mọi chính sách công bố trí ngân sách đều là một lựa chọn điểm trên biên Pareto, đòi hỏi sự đồng thuận giữa các bộ ban ngành kinh tế, xã hội và tài nguyên môi trường.
        - **Hành động thích ứng với cam kết COP26**: Quyết định ưu tiên tăng trưởng bằng mọi giá sẽ trực tiếp đe dọa cam kết phát thải ròng bằng 0 (Net Zero) vào năm 2050. Việc áp dụng cơ cấu thỏa hiệp giúp cân bằng tăng trưởng công nghệ số đồng thời giữ mức phát thải trong tầm kiểm soát.
        - **Vai trò của các mô hình đa mục tiêu**: Thay vì chỉ cung cấp một con số, thuật toán giúp vẽ ra toàn bộ "bản đồ lựa chọn" (Pareto Front), lượng hóa cụ thể chi phí cơ hội của từng kịch bản để hỗ trợ quá trình thương thảo chính sách một cách công khai, minh bạch.
        """)
        
    with tabs[6]:
        st.header("7. Tham khảo")
        st.markdown("""
        - **Deb, K. (2001).** Multi-Objective Optimization using Evolutionary Algorithms.
        - Văn kiện Đại hội đại biểu toàn quốc lần thứ XIII của Đảng.
        - Quyết định số 127/QĐ-TTg phê duyệt Chiến lược quốc gia về nghiên cứu, phát triển và ứng dụng AI đến năm 2030.
        - Báo cáo đánh giá tác động COP26 đối với kinh tế năng lượng Việt Nam (Bộ Công thương).
        """)

if __name__ == "__main__":
    render()
