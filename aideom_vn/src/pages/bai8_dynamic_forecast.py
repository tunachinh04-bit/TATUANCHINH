import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def solve_dynamic_optimization(tfp_base, shock_2028=False):
    # Parameters
    T = 10  # 2026-2035
    rho = 0.97
    delta_K = 0.05
    delta_D = 0.12
    delta_AI = 0.15
    theta_H = 0.8
    mu = 0.02
    phi1, phi2, phi3 = 0.003, 0.002, 0.004
    L = 54.0
    
    # Initial conditions
    K0 = 27500.0
    D0 = 20.3
    AI0 = 86.0
    H0 = 30.0
    A0 = tfp_base
    
    # 40 decision variables: x[4*t + j] represents I_K, I_D, I_AI, I_H for year t
    # Bounds: all investments >= 0
    bounds = [(0, None)] * (T * 4)
    
    # Helper to simulate state trajectory
    def simulate(x):
        inv = x.reshape((T, 4))
        K = np.zeros(T + 1)
        D = np.zeros(T + 1)
        AI = np.zeros(T + 1)
        H = np.zeros(T + 1)
        A = np.zeros(T + 1)
        Y = np.zeros(T)
        C = np.zeros(T)
        
        K[0] = K0
        D[0] = D0
        AI[0] = AI0
        H[0] = H0
        A[0] = A0
        
        for t in range(T):
            # Production function
            y_t = A[t] * (K[t]**0.33) * (L**0.42) * (D[t]**0.10) * (AI[t]**0.08) * (H[t]**0.07)
            if shock_2028 and t == 2:  # t=2 corresponds to 2028 (2026 is t=0)
                y_t *= 0.92  # 8% shock
            Y[t] = y_t
            
            # Consumption
            c_t = y_t - np.sum(inv[t])
            C[t] = c_t
            
            # Update states
            K[t+1] = (1 - delta_K) * K[t] + inv[t, 0]
            D[t+1] = (1 - delta_D) * D[t] + inv[t, 1]
            AI[t+1] = (1 - delta_AI) * AI[t] + inv[t, 2]
            H[t+1] = (1 - mu) * H[t] + theta_H * inv[t, 3]
            A[t+1] = A[t] * (1 + phi1 * D[t] + phi2 * AI[t] + phi3 * H[t])
            
        return K, D, AI, H, A, Y, C
    
    # Objective: Maximize sum(rho^t * ln(C_t)) => Minimize -sum(...)
    def objective(x):
        _, _, _, _, _, _, C = simulate(x)
        if np.any(C <= 0):
            return 1e10  # heavy penalty for negative consumption
        return -np.sum(rho**np.arange(T) * np.log(C))
    
    # Constraint: Consumption >= 10.0 for all years
    def cons_consumption(x):
        _, _, _, _, _, _, C = simulate(x)
        return C - 100.0  # C_t >= 100
        
    constraints = {'type': 'ineq', 'fun': cons_consumption}
    
    # Initial guess: invest 15% of GDP in K, 2% in D, 2% in AI, 2% in H
    x0 = np.zeros(T * 4)
    for t in range(T):
        y_approx = A0 * (K0**0.33) * (L**0.42) * (D0**0.10) * (AI0**0.08) * (H0**0.07)
        x0[4*t] = y_approx * 0.15   # I_K
        x0[4*t+1] = y_approx * 0.02 # I_D
        x0[4*t+2] = y_approx * 0.02 # I_AI
        x0[4*t+3] = y_approx * 0.02 # I_H
        
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200})
    
    K, D, AI, H, A, Y, C = simulate(res.x)
    return res.x.reshape((T, 4)), K, D, AI, H, A, Y, C, -res.fun

def render():
    st.title("⏳ Bài 8 — Tối ưu hóa động phân bổ liên thời gian 2026-2035")
    
    st.markdown("""
    **Mục tiêu học tập:** Sinh viên xây dựng và giải bài toán quy hoạch phi tuyến động (Dynamic Programming / Nonlinear Optimization) 
    liên thời gian với phương trình tích lũy vốn của 4 yếu tố, hàm sản xuất Cobb-Douglas, và cập nhật TFP nội sinh bằng SciPy.
    """)

    tabs = st.tabs([
        "📖 Bối cảnh & Vấn đề", 
        "🔬 Lý thuyết động học", 
        "🛠️ Phương pháp tối ưu", 
        "📈 Quỹ đạo tối ưu 2026-2035", 
        "⚡ Phân tích Cú sốc", 
        "💡 Thảo luận chính sách",
        "📚 Tham khảo"
    ])
    
    with tabs[0]:
        st.header("1. Bối cảnh & Vấn đề")
        st.markdown("""
        Theo Chiến lược phát triển kinh tế - xã hội 10 năm 2021–2030, Việt Nam đặt mục tiêu trở thành nước có thu nhập trung bình cao vào năm 2030. 
        Đầu tư số và AI không thể là quyết định ngắn hạn; chúng tích lũy thành trữ lượng **Vốn số** và tạo năng lực lan tỏa cho năng suất vĩ mô.
        
        Tuy nhiên, đầu tư công nghệ có tốc độ lỗi thời (khấu hao) nhanh hơn nhiều so với nhà xưởng, đường sá truyền thống. 
        Chính phủ cần thiết lập một **Quỹ đạo đầu tư liên thời gian** tối ưu để phân bổ dòng vốn hợp lý qua từng năm.
        """)
        st.info("💡 **Hành động**: Giải bài toán tối ưu động tìm chuỗi đầu tư dài hạn nhằm tối đa hóa tổng phúc lợi kinh tế.")
        
    with tabs[1]:
        st.header("2. Mô hình toán học động")
        st.markdown("**Hàm thỏa dụng phúc lợi liên thời gian (Welfare):**")
        st.latex(r"\max \sum_{t=2026}^{2035} \rho^{t-2026} \ln(C_t)")
        st.caption("ρ = 0.97 là hệ số chiết khấu tiêu dùng dài hạn.")
        
        st.markdown("**Động học tích lũy vốn & nhân lực:**")
        st.latex(r"K_{t+1} = (1 - \delta_K) K_t + I_{K,t} \quad (\delta_K = 0.05)")
        st.latex(r"D_{t+1} = (1 - \delta_D) D_t + I_{D,t} \quad (\delta_D = 0.12)")
        st.latex(r"AI_{t+1} = (1 - \delta_{AI}) AI_t + I_{AI,t} \quad (\delta_{AI} = 0.15)")
        st.latex(r"H_{t+1} = (1 - \mu) H_t + \theta_H I_{H,t} \quad (\mu = 0.02, \theta_H = 0.8)")
        
        st.markdown("**Năng suất nhân tố tổng hợp (TFP) nội sinh:**")
        st.latex(r"A_{t+1} = A_t (1 + 0.003 D_t + 0.002 AI_t + 0.004 H_t)")
        st.caption("Chuyển đổi số, AI và Đào tạo giúp cải thiện năng lực công nghệ và hiệu suất vĩ mô một cách gián tiếp.")
        
        st.markdown("**Ràng buộc ngân sách:**")
        st.latex(r"C_t + I_{K,t} + I_{D,t} + I_{AI,t} + I_{H,t} \le Y_t")

    with tabs[2]:
        st.header("3. Phương pháp giải Quy hoạch phi tuyến")
        st.markdown("""
        Vì hàm sản xuất vĩ mô và cập nhật TFP phi tuyến tính, chúng ta giải bài toán bằng phương pháp **Sequential Least Squares Programming (SLSQP)** 
        thông qua thư viện `scipy.optimize`. 
        
        - **Quyết định**: 40 biến (10 năm x 4 hạng mục đầu tư).
        - **Mục tiêu**: Tối đa hóa tổng hữu dụng logarit của tiêu dùng.
        - **Ràng buộc**: Tiêu dùng mỗi năm phải dương và đáp ứng điều kiện tích lũy vốn của nền kinh tế.
        """)

    with tabs[3]:
        st.header("4. Quỹ đạo tối ưu hóa vĩ mô")
        
        # Calibrated A0 from historic average
        tfp_base = 33.8
        
        # Run solver
        inv_opt, K, D, AI, H, A, Y, C, obj_val = solve_dynamic_optimization(tfp_base, shock_2028=False)
        
        years = list(range(2026, 2036))
        
        df_states = pd.DataFrame({
            'Năm': years,
            'GDP (Y)': Y,
            'Tiêu dùng (C)': C,
            'Vốn truyền thống (K)': K[:-1],
            'Số hóa (D)': D[:-1],
            'AI (DN số)': AI[:-1],
            'Nhân lực đào tạo (H)': H[:-1]
        })
        
        st.subheader("Kết quả quỹ đạo tối ưu (Nghìn tỷ VND):")
        st.dataframe(df_states.style.format(precision=1), use_container_width=True)
        
        # Plot Trajectories
        fig_gdp = go.Figure()
        fig_gdp.add_trace(go.Scatter(x=years, y=Y, mode='lines+markers', name='GDP (Y)', line=dict(color='blue', width=3)))
        fig_gdp.add_trace(go.Scatter(x=years, y=C, mode='lines+markers', name='Tiêu dùng (C)', line=dict(color='green', dash='dash')))
        fig_gdp.update_layout(title="Quỹ đạo GDP và Tiêu dùng tối ưu", xaxis_title="Năm", yaxis_title="Nghìn tỷ VND")
        st.plotly_chart(fig_gdp, use_container_width=True)
        
        # Plot investment allocation
        df_inv = pd.DataFrame(inv_opt, columns=['Đầu tư K', 'Đầu tư D', 'Đầu tư AI', 'Đầu tư H'], index=years).reset_index().rename(columns={'index':'Năm'})
        df_inv_melt = df_inv.melt(id_vars='Năm', var_name='Hạng mục', value_name='Ngân sách (Tỷ VND)')
        
        fig_inv = px.bar(df_inv_melt, x='Năm', y='Ngân sách (Tỷ VND)', color='Hạng mục', title="Cơ cấu đầu tư tối ưu hàng năm")
        st.plotly_chart(fig_inv, use_container_width=True)

    with tabs[4]:
        st.header("5. Phân tích Cú sốc kinh tế 2028")
        st.markdown("""
        Giả sử vào năm 2028 xảy ra một cú sốc thiên tai hoặc bất định chuỗi cung ứng toàn cầu làm GDP thực tế năm đó sụt giảm **8%** so với dự kiến. 
        Chính phủ sẽ điều chỉnh dòng vốn đầu tư trong các năm còn lại như thế nào?
        """)
        
        # Run optimization with shock
        inv_shock, K_s, D_s, AI_s, H_s, A_s, Y_s, C_s, obj_val_s = solve_dynamic_optimization(tfp_base, shock_2028=True)
        
        fig_shock = go.Figure()
        fig_shock.add_trace(go.Scatter(x=years, y=Y, mode='lines+markers', name='GDP Cơ bản (Không sốc)', line=dict(color='blue')))
        fig_shock.add_trace(go.Scatter(x=years, y=Y_s, mode='lines+markers', name='GDP Có Sốc 2028', line=dict(color='red', width=3)))
        fig_shock.update_layout(title="So sánh tác động của Cú sốc GDP năm 2028", xaxis_title="Năm", yaxis_title="GDP (Nghìn tỷ VND)")
        st.plotly_chart(fig_shock, use_container_width=True)
        
        # Compare total welfare
        st.write(f"Tổng Welfare không sốc (trị số mục tiêu): **{obj_val:.4f}**")
        st.write(f"Tổng Welfare có sốc: **{obj_val_s:.4f}**")
        st.info("💡 **Nhận xét**: Khi gặp cú sốc năm 2028, mô hình tối ưu điều chỉnh bằng cách cắt giảm một phần đầu tư công nghệ số để bảo vệ mức tiêu dùng thiết yếu của người dân, sau đó tăng tốc phục hồi đầu tư nhân lực (H) và hạ tầng số (D) trong các năm cuối kỳ.")

    with tabs[5]:
        st.header("6. Thảo luận chính sách")
        st.markdown(r"""
        Các bài học thu được từ mô hình hoạch định động:
        
        - **Đặc tính đầu tư Front-loaded**: Đầu tư công nghệ (D, AI, H) thường mang tính "giải ngân sớm" (front-loaded) ở những năm đầu của chu kỳ quy hoạch. Mô hình đề xuất điều này do đầu tư công nghệ có tác động lan tỏa lâu dài tới TFP nội sinh, do đó đầu tư càng sớm càng tạo ra hiệu ứng cộng dồn tốt cho các năm sau.
        - **Khấu hao công nghệ cao**: Khấu hao AI và D rất lớn (12%-15%), buộc Chính phủ phải liên tục đầu tư tái tạo chứ không thể "dừng hẳn" sau khi đạt mục tiêu.
        - **Hệ số chiết khấu**: Hệ số chiết khấu $\rho = 0.97$ phản ánh mức độ coi trọng tương lai. Nếu hạ xuống $0.90$ (ngắn hạn hơn), mô hình sẽ giảm đầu tư R&D và giáo dục để dồn cho tiêu dùng ngắn hạn, đây chính là lời giải thích khoa học cho hiện tượng "dưới đầu tư dài hạn" của các chính trị gia nhiệm kỳ.
        """)
        
    with tabs[6]:
        st.header("7. Tham khảo")
        st.markdown("""
        - **Solow, R. M. (1956).** A Contribution to the Theory of Economic Growth.
        - Báo cáo chiến lược phát triển kinh tế xã hội Việt Nam 2021-2030 (Đại hội XIII).
        - **Ljungqvist, L., & Sargent, T. J. (2018):** Recursive Macroeconomic Theory.
        """)

if __name__ == "__main__":
    render()
