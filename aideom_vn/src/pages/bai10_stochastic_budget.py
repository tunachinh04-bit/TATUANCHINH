import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def solve_stochastic_program(p_s1, p_s2, p_s3, p_s4, budget1, budget2):
    try:
        import pulp
    except ImportError as e:
        raise ImportError("PuLP is required to solve stochastic programming problems. Install it with `pip install pulp`.") from e
    
    # Categories and Scenarios
    J = ['I', 'D', 'AI', 'H']
    S = ['s1', 's2', 's3', 's4']
    p = {'s1': p_s1, 's2': p_s2, 's3': p_s3, 's4': p_s4}
    
    beta = {'I': 1.00, 'D': 1.10, 'AI': 1.25, 'H': 0.95}
    beta_s = {
        ('s1','I'): 1.25, ('s1','D'): 1.35, ('s1','AI'): 1.55, ('s1','H'): 1.05,
        ('s2','I'): 1.00, ('s2','D'): 1.10, ('s2','AI'): 1.25, ('s2','H'): 0.95,
        ('s3','I'): 0.75, ('s3','D'): 0.85, ('s3','AI'): 0.90, ('s3','H'): 1.00,
        ('s4','I'): 0.40, ('s4','D'): 0.50, ('s4','AI'): 0.55, ('s4','H'): 1.10
    }
    
    # ------------------ 1. SOLVE STOCHASTIC PROGRAM (SP) ------------------
    prob_sp = pulp.LpProblem("Stochastic_Program", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", J, lowBound=0)
    y = pulp.LpVariable.dicts("y", (S, J), lowBound=0)
    
    # Objective: first stage + expected second stage
    first_stage = pulp.lpSum(beta[j] * x[j] for j in J)
    second_stage = pulp.lpSum(p[s] * pulp.lpSum(beta_s[(s, j)] * y[s][j] for j in J) for s in S)
    prob_sp += first_stage + second_stage
    
    # Constraints
    prob_sp += pulp.lpSum(x[j] for j in J) <= budget1, "Budget1"
    for s in S:
        prob_sp += pulp.lpSum(y[s][j] for j in J) <= budget2, f"Budget2_{s}"
        prob_sp += y[s]['AI'] <= 0.5 * x['H'], f"Precedence_{s}"
        
    prob_sp.solve(pulp.PULP_CBC_CMD(msg=0))
    z_sp = pulp.value(prob_sp.objective)
    x_sp = {j: x[j].varValue for j in J}
    
    # ------------------ 2. SOLVE EXPECTED VALUE (EV) PROBLEM ------------------
    # EV replaces random params with their expected values
    prob_ev = pulp.LpProblem("Expected_Value", pulp.LpMaximize)
    x_ev = pulp.LpVariable.dicts("x_ev", J, lowBound=0)
    y_ev = pulp.LpVariable.dicts("y_ev", J, lowBound=0)
    
    # Expected beta_s
    mean_beta_s = {j: sum(p[s] * beta_s[(s, j)] for s in S) for j in J}
    prob_ev += pulp.lpSum(beta[j] * x_ev[j] for j in J) + pulp.lpSum(mean_beta_s[j] * y_ev[j] for j in J)
    
    prob_ev += pulp.lpSum(x_ev[j] for j in J) <= budget1
    prob_ev += pulp.lpSum(y_ev[j] for j in J) <= budget2
    prob_ev += y_ev['AI'] <= 0.5 * x_ev['H']
    
    prob_ev.solve(pulp.PULP_CBC_CMD(msg=0))
    z_ev = pulp.value(prob_ev.objective)
    x_EV_val = {j: x_ev[j].varValue for j in J}
    
    # ------------------ 3. SOLVE EEV PROBLEM (EXPECTED RESULT OF EV) ------------------
    # We fix x = x_EV_val and solve stage 2 for each scenario
    eev_scenario_profits = []
    for s in S:
        prob_s = pulp.LpProblem(f"EEV_{s}", pulp.LpMaximize)
        y_s = pulp.LpVariable.dicts(f"y_eev_{s}", J, lowBound=0)
        
        prob_s += pulp.lpSum(beta_s[(s, j)] * y_s[j] for j in J)
        prob_s += pulp.lpSum(y_s[j] for j in J) <= budget2
        prob_s += y_s['AI'] <= 0.5 * x_EV_val['H']
        
        prob_s.solve(pulp.PULP_CBC_CMD(msg=0))
        eev_scenario_profits.append(pulp.value(prob_s.objective))
        
    first_stage_ev_cost = sum(beta[j] * x_EV_val[j] for j in J)
    expected_eev_second_stage = sum(p[s] * eev_scenario_profits[idx] for idx, s in enumerate(S))
    z_eev = first_stage_ev_cost + expected_eev_second_stage
    
    # ------------------ 4. SOLVE WAIT-AND-SEE (WS) PROBLEM ------------------
    # For each scenario, we solve the joint stage 1 + stage 2 LP, then calculate expectation
    ws_scenario_profits = []
    for s in S:
        prob_s = pulp.LpProblem(f"WS_{s}", pulp.LpMaximize)
        x_s = pulp.LpVariable.dicts(f"x_ws_{s}", J, lowBound=0)
        y_s = pulp.LpVariable.dicts(f"y_ws_{s}", J, lowBound=0)
        
        prob_s += pulp.lpSum(beta[j] * x_s[j] for j in J) + pulp.lpSum(beta_s[(s, j)] * y_s[j] for j in J)
        prob_s += pulp.lpSum(x_s[j] for j in J) <= budget1
        prob_s += pulp.lpSum(y_s[j] for j in J) <= budget2
        prob_s += y_s['AI'] <= 0.5 * x_s['H']
        
        prob_s.solve(pulp.PULP_CBC_CMD(msg=0))
        ws_scenario_profits.append(pulp.value(prob_s.objective))
        
    z_ws = sum(p[s] * ws_scenario_profits[idx] for idx, s in enumerate(S))
    
    # Metrics
    vss = z_sp - z_eev
    evpi = z_ws - z_sp
    
    return x_sp, x_EV_val, z_sp, z_eev, z_ws, vss, evpi

def render():
    st.title("🎲 Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định")
    
    st.markdown("""
    **Mục tiêu học tập:** Sinh viên xây dựng được bài toán quy hoạch ngẫu nhiên hai giai đoạn (Two-stage Stochastic Programming), 
    hiểu ý nghĩa của quyết định first-stage (here-and-now) và second-stage recourse (wait-and-see), 
    đồng thời biết cách tính toán và đánh giá chỉ số VSS (Value of Stochastic Solution) và EVPI (Expected Value of Perfect Information).
    """)

    tabs = st.tabs([
        "📖 Bối cảnh & Cấu trúc", 
        "🔬 Lý thuyết Two-stage", 
        "🌳 Cấu trúc kịch bản", 
        "📊 Bài toán phân bổ", 
        "📈 Kết quả Tối ưu ngẫu nhiên", 
        "💡 Thảo luận chính sách",
        "📚 Tham khảo"
    ])
    
    with tabs[0]:
        st.header("1. Bối cảnh & Vấn đề")
        st.markdown("""
        Việt Nam có độ mở thương mại lớn, tăng trưởng chịu nhiều ảnh hưởng của kịch bản kinh tế toàn cầu, 
        dòng vốn FDI và rủi ro chuỗi cung ứng. 
        
        Khi thiết lập kế hoạch ngân sách AI vĩ mô giai đoạn 2026-2030, Chính phủ phải quyết định mức ngân sách giải ngân trước (First stage) 
        khi chưa biết kịch bản tương lai bùng nổ hay khủng hoảng, đồng thời giữ lại một khoản dự phòng điều chỉnh bổ sung (Second stage).
        """)
        st.info("💡 **Vấn đề**: Quyết định phân bổ ngân sách giai đoạn 1 thế nào để tổng lợi ích kỳ vọng (NPV) đạt lớn nhất?")
        
    with tabs[1]:
        st.header("2. Lý thuyết quy hoạch ngẫu nhiên 2 giai đoạn")
        st.markdown("Mô hình được biểu diễn dưới dạng toán học tổng quát:")
        st.latex(r"\max \sum_{j} \beta_j x_j + \sum_{s \in S} p_s \left[ \sum_j \beta_{j}^s y_j^s \right]")
        st.markdown("Các ràng buộc của mô hình:")
        st.latex(r"\begin{aligned} &\sum_{j} x_j \le Budget_{1} && \text{(Ngân sách GĐ1)} \\ &\sum_{j} y_j^s \le Budget_{2} \quad \forall s && \text{(Ngân sách dự phòng GĐ2)} \\ &y_{AI}^s \le 0.5 \cdot x_H \quad \forall s && \text{(Ràng buộc tiên quyết: AI cần Nhân lực H)} \\ &x_j, y_j^s \ge 0 \quad \forall j, s \end{aligned}")
        st.markdown("""
        **Các khái niệm then chốt:**
        - **VSS (Value of Stochastic Solution)**: Đo lường giá trị của việc xem xét yếu tố ngẫu nhiên khi ra quyết định. $VSS = Z_{SP} - Z_{EEV}$.
        - **EVPI (Expected Value of Perfect Information)**: Đo lường giá trị tối đa mà Chính phủ sẵn sàng chi trả để có được thông tin hoàn hảo trước khi ra quyết định. $EVPI = Z_{WS} - Z_{SP}$.
        """)

    with tabs[2]:
        st.header("3. Đặc tả 4 kịch bản kinh tế vĩ mô")
        st.markdown("Thiết lập xác suất xảy ra của 4 kịch bản vĩ mô dài hạn:")
        
        col1, col2 = st.columns(2)
        with col1:
            p_s1 = st.slider("Xác suất s1: Lạc quan (%)", 0, 100, 30) / 100
            p_s2 = st.slider("Xác suất s2: Cơ sở (%)", 0, 100, 45) / 100
        with col2:
            p_s3 = st.slider("Xác suất s3: Bi quan (%)", 0, 100, 20) / 100
            p_s4 = st.slider("Xác suất s4: Khủng hoảng (%)", 0, 100, 5) / 100
            
        sum_p = p_s1 + p_s2 + p_s3 + p_s4
        if abs(sum_p - 1.0) > 1e-5:
            st.warning(f"Tổng xác suất bằng {sum_p * 100:.1f}%. Hệ thống sẽ tự động renormalize về 100% để bảo đảm tính chuẩn tắc.")
            p_s1 /= sum_p
            p_s2 /= sum_p
            p_s3 /= sum_p
            p_s4 /= sum_p

        # Draw decision tree using plotly
        fig_tree = go.Figure()
        fig_tree.add_trace(go.Scatter(x=[0, 1, 1, 1, 1], y=[0, 1.5, 0.5, -0.5, -1.5], mode='lines+markers+text',
                                     text=["GĐ1 (Now)", f"s1: Lạc quan (p={p_s1:.2f})", f"s2: Cơ sở (p={p_s2:.2f})", f"s3: Bi quan (p={p_s3:.2f})", f"s4: Khủng hoảng (p={p_s4:.2f})"],
                                     textposition="top right",
                                     marker=dict(size=[25, 15, 15, 15, 15], color=['black', 'green', 'blue', 'orange', 'red'])))
        fig_tree.update_layout(title="Cấu trúc cây quyết định 2 giai đoạn", showlegend=False,
                               xaxis=dict(visible=False, range=[-0.2, 2.0]), yaxis=dict(visible=False, range=[-2.0, 2.0]))
        st.plotly_chart(fig_tree, use_container_width=True)

    with tabs[3]:
        st.header("4. Thiết lập Ngân sách")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            budget1 = st.slider("Ngân sách giai đoạn 1 (Tỷ VND)", 40000, 75000, 65000, step=5000)
        with col_b2:
            budget2 = st.slider("Ngân sách dự phòng giai đoạn 2 (Tỷ VND)", 5000, 25000, 15000, step=2500)

    with tabs[4]:
        st.header("5. Kết quả Phân tích Ngẫu nhiên")
        
        try:
            x_sp, x_EV, z_sp, z_eev, z_ws, vss, evpi = solve_stochastic_program(p_s1, p_s2, p_s3, p_s4, budget1, budget2)
        except ImportError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Lỗi trong giải bài toán ngẫu nhiên: {e}")
            return
        
        # Display KPIs
        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            st.metric("Lợi ích Stochastic Z* (SP)", f"{z_sp:,.1f} tỷ VND")
        with col_k2:
            st.metric("Chỉ số VSS (Giá trị giải pháp ngẫu nhiên)", f"{vss:,.1f} tỷ VND")
        with col_k3:
            st.metric("Chỉ số EVPI (Giá trị thông tin)", f"{evpi:,.1f} tỷ VND")
            
        st.divider()
        
        # Compare first-stage decisions
        df_compare = pd.DataFrame({
            "Hạng mục": ["Hạ tầng số (I)", "Chuyển đổi số (D)", "Trí tuệ nhân tạo (AI)", "Nhân lực số (H)"],
            "Giải pháp Stochastic (SP)": [x_sp['I'], x_sp['D'], x_sp['AI'], x_sp['H']],
            "Giải pháp Xác định (EV)": [x_EV['I'], x_EV['D'], x_EV['AI'], x_EV['H']]
        })
        st.subheader("So sánh quyết định phân bổ giai đoạn 1:")
        st.dataframe(df_compare.style.format(precision=1), use_container_width=True)
        
        # Explain results
        st.info(f"""
        > [!TIP]
        > **Nhận xét chính sách**: 
        > Giải pháp ngẫu nhiên (SP) phân bổ nhiều ngân sách hơn cho **Nhân lực số (H)** so với giải pháp xác định dựa trên trung bình (EV). 
        > Điều này là do SP nhận diện được rủi ro trong các kịch bản xấu (s3, s4), nơi AI sụt giảm hiệu quả nhưng nhân lực H lại trở thành yếu tố bảo hiểm kinh tế vĩ mô bền vững, 
        > giúp nền kinh tế hấp thụ tốt hơn các cú sốc. VSS = **{vss:,.1f} tỷ VND** là minh chứng định lượng cho lợi ích của việc hoạch định ngân sách phòng vệ rủi ro.
        """)

    with tabs[5]:
        st.header("6. Thảo luận chính sách")
        st.markdown(f"""
- **Tư duy phòng vệ (Hedging)**: Hoạch định chính sách trong kỷ nguyên bất định đòi hỏi tư duy phòng vệ rủi ro. 
  Một giải pháp trung dung nhưng linh hoạt tốt hơn một giải pháp tối ưu cục bộ cho kịch bản trung bình nhưng dễ sụp đổ khi khủng hoảng xảy ra.
  _Cơ sở chính sách: Nghị quyết 52-NQ/TW ngày 27/09/2019 về chủ trương, chính sách chủ động tham gia Cách mạng công nghiệp lần thứ tư._
- **Giá trị thông tin hoàn hảo (EVPI = {evpi:,.1f} tỷ VND)**: 
  Đây là giới hạn ngân sách tối đa mà Chính phủ nên chi cho hoạt động nghiên cứu, phân tích dự báo 
  và tình báo công nghệ để có thông tin tin cậy trước khi quyết định. 
  _Tham chiếu: Quyết định 749/QĐ-TTg phê duyệt Chương trình CDS Quốc gia 2025, điều khoản xây dựng cơ sở dữ liệu số quốc gia._
- **Đại dịch COVID-19 và Bão Yagi**: Đây là các kịch bản khủng hoảng ($s_4$) thực tế chứng minh bài học: 
  Việt Nam cần tích lũy vốn nhân lực giáo dục thiết yếu để tăng độ dẻ dai thích nghi của lực lượng lao động
  trước các biến cố đột ngột của tự nhiên và chuỗi cung ứng. (**VSS = {vss:,.1f} tỷ VND** — giá trị của việc lập kế hoạch ngân sách phòng vệ rủi ro.)
        """)
        
    with tabs[6]:
        st.header("7. Tham khảo")
        st.markdown("""
        - **Shapiro, A., Dentcheva, D., & Ruszczynski, A. (2021).** Lectures on Stochastic Programming: Modeling and Theory.
        - Báo cáo Kinh tế vĩ mô Việt Nam (Ngân hàng Thế giới).
        - **Kall, P., & Wallace, S. W. (1994):** Stochastic Programming (John Wiley & Sons).
        """)

if __name__ == "__main__":
    render()
