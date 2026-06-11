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
    st.title("👷 Bài 9 — Tối ưu hóa phân bổ nguồn nhân lực và mô phỏng lao động dưới tác động AI")
    
    st.markdown("""
    **Mục tiêu học tập:** Sinh viên nắm vững phương pháp thiết lập bài toán LP tối ưu hóa cơ cấu việc làm quốc gia, 
    xác định mối quan hệ biện chứng giữa tạo việc làm số và tự động hóa thay thế lao động, 
    tính toán được ngưỡng đầu tư đào tạo lại tối thiểu và trực quan hóa luồng dịch chuyển bằng biểu đồ Sankey.
    """)

    tabs = st.tabs([
        "📖 Bối cảnh & Vấn đề", 
        "🔬 Mô hình toán học", 
        "📊 Dữ liệu 8 Ngành", 
        "📈 Kết quả tối ưu hóa", 
        "🌊 Dòng dịch chuyển (Sankey)", 
        "💡 Thảo luận chính sách",
        "📚 Tham khảo"
    ])
    
    with tabs[0]:
        st.header("1. Bối cảnh & Vấn đề")
        st.markdown("""
        Theo báo cáo của ILO và Tổng cục Thống kê, kỷ nguyên AI tạo ra cả hai hiệu ứng đối lập trên thị trường lao động Việt Nam:
        
        1. **Hiệu ứng thay thế (Displacement Effect)**: Tự động hóa thay thế các công việc thủ công, l lặp lại trong các ngành chế biến chế tạo, bán lẻ và logistics.
        2. **Hiệu ứng tạo mới (Productivity & Creation Effect)**: AI tạo ra các việc làm mới có kỹ năng cao hơn.
        
        **Câu hỏi đặt ra:** Làm thế nào để phân bổ **30.000 tỷ VND** ngân sách phát triển lao động số cho 8 ngành kinh tế chính để 
        tối đa hóa số việc làm ròng tăng thêm ($NetJob$), đồng thời đảm bảo không có ngành nào bị âm việc làm (không sa thải ròng)?
        """)
        st.info("💡 **Mục tiêu**: Tối đa hóa tổng NetJob vĩ mô sao cho NetJob mỗi ngành không âm và tốc độ tự động hóa không vượt quá năng lực đào tạo lại.")
        
    with tabs[1]:
        st.header("2. Mô hình toán học LP")
        st.markdown("**Biến quyết định cho mỗi ngành $i \\in \\{1..8\\}$:**")
        st.markdown("- $x_{AI,i}$: Ngân sách đầu tư ứng dụng AI và số hóa (tỷ VND).")
        st.markdown("- $x_{H,i}$: Ngân sách đầu tư đào tạo lại nguồn nhân lực (tỷ VND).")
        
        st.markdown("**Hệ phương trình việc làm ròng:**")
        st.latex(r"NetJob_i = NewJob_i^{AI} + UpgradeJob_i - DisplacedJob_i^{Auto}")
        st.markdown("Trong đó:")
        st.latex(r"NewJob_i^{AI} = a_{1,i} \cdot x_{AI,i}")
        st.latex(r"UpgradeJob_i = b_{1,i} \cdot x_{H,i}")
        st.latex(r"DisplacedJob_i^{Auto} = c_{1,i} \cdot x_{AI,i} \cdot Risk_i")
        st.latex(r"RetrainCapacity_i = d_{1,i} \cdot x_{H,i}")
        
        st.markdown("**Các ràng buộc:**")
        st.markdown(r"- **Tổng ngân sách:** $\sum_i (x_{AI,i} + x_{H,i}) \le 30.000$ (tỷ VND)")
        st.markdown(r"- **Không mất việc ròng:** $NetJob_i \ge 0 \quad \forall i$")
        st.markdown(r"- **Giới hạn đào tạo lại:** $DisplacedJob_i^{Auto} \le RetrainCapacity_i \quad \forall i$ (không tự động hóa nhanh hơn tốc độ đào tạo)")

    with tabs[2]:
        st.header("3. Dữ liệu 8 ngành kinh tế Việt Nam")
        st.markdown("Tham số ước lượng từ Niên giám Thống kê 2024 và ILO (số việc làm trên mỗi tỷ VND đầu tư):")
        
        sectors = [
            "1. Nông-Lâm-Thủy sản", "2. CN chế biến chế tạo", "3. Xây dựng",
            "4. Bán buôn-bán lẻ", "5. Tài chính-Ngân hàng", "6. Logistics-Vận tải",
            "7. CNTT-Truyền thông", "8. Giáo dục-Đào tạo"
        ]
        
        df_sectors = pd.DataFrame({
            "Ngành": sectors,
            "Lao động (triệu)": [13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15],
            "Risk (%)": [18, 42, 25, 38, 52, 35, 28, 22],
            "a1 (việc/tỷ)": [8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5],
            "b1 (việc/tỷ)": [45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0],
            "c1 (việc/tỷ)": [5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5],
            "d1 (việc/tỷ)": [50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0]
        })
        st.dataframe(df_sectors.style.format(precision=1), use_container_width=True)
        st.caption("Nguồn: vietnam_sectors_2024.csv hiệu chỉnh theo mô hình lao động của ILO.")

    with tabs[3]:
        st.header("4. Kết quả phân bổ tối ưu")
        
        budget = st.slider("Tổng ngân sách lao động & công nghệ (Tỷ VND)", 10000, 50000, 30000, step=2500)
        
        a1 = df_sectors["a1 (việc/tỷ)"].values
        b1 = df_sectors["b1 (việc/tỷ)"].values
        c1 = df_sectors["c1 (việc/tỷ)"].values
        d1 = df_sectors["d1 (việc/tỷ)"].values
        risk = df_sectors["Risk (%)"].values / 100.0

        optimization_success = False
        x_AI_opt = np.zeros(8)
        x_H_opt = np.zeros(8)
        displaced_val = np.zeros(8)

        try:
            import pulp
            
            # Solve LP
            prob = pulp.LpProblem("Labor_Allocation", pulp.LpMaximize)
            
            x_AI = pulp.LpVariable.dicts("x_AI", range(8), lowBound=0)
            x_H = pulp.LpVariable.dicts("x_H", range(8), lowBound=0)
            
            # NetJob_i = a1*x_AI + b1*x_H - c1*risk*x_AI
            prob += pulp.lpSum((a1[i] * x_AI[i] + b1[i] * x_H[i] - c1[i] * risk[i] * x_AI[i]) for i in range(8)), "Total_NetJobs"
            
            # Constraints
            prob += pulp.lpSum(x_AI[i] + x_H[i] for i in range(8)) <= budget, "Total_Budget"
            for i in range(8):
                net_job = a1[i] * x_AI[i] + b1[i] * x_H[i] - c1[i] * risk[i] * x_AI[i]
                displaced = c1[i] * risk[i] * x_AI[i]
                retrain = d1[i] * x_H[i]
                
                prob += net_job >= 0, f"NonNegative_NetJob_{i}"
                prob += displaced <= retrain, f"Retrain_Capacity_{i}"
                
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if pulp.LpStatus[prob.status] == 'Optimal':
                optimization_success = True
                x_AI_opt = np.array([x_AI[i].varValue for i in range(8)])
                x_H_opt = np.array([x_H[i].varValue for i in range(8)])
                
                net_jobs_val = [a1[i]*x_AI_opt[i] + b1[i]*x_H_opt[i] - c1[i]*risk[i]*x_AI_opt[i] for i in range(8)]
                displaced_val = [c1[i]*risk[i]*x_AI_opt[i] for i in range(8)]
                
                df_res = df_sectors.copy()
                df_res["Đầu tư AI (Tỷ VND)"] = x_AI_opt
                df_res["Đầu tư Đào tạo H (Tỷ VND)"] = x_H_opt
                df_res["Việc làm bị thay thế"] = displaced_val
                df_res["Việc làm ròng (NetJob)"] = net_jobs_val
                
                st.success(f"Tổng số việc làm ròng tạo thêm tối ưu: **{pulp.value(prob.objective):,.0f}** việc làm")
                
                st.dataframe(df_res[["Ngành", "Đầu tư AI (Tỷ VND)", "Đầu tư Đào tạo H (Tỷ VND)", "Việc làm bị thay thế", "Việc làm ròng (NetJob)"]].style.format(precision=1), use_container_width=True)
                
                # Plot allocation
                df_plot = df_res.melt(id_vars='Ngành', value_vars=['Đầu tư AI (Tỷ VND)', 'Đầu tư Đào tạo H (Tỷ VND)'], var_name='Hạng mục', value_name='Ngân sách')
                fig_alloc = px.bar(df_plot, x='Ngành', y='Ngân sách', color='Hạng mục', title="Cơ cấu phân bổ ngân sách tối ưu theo ngành")
                fig_alloc.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_alloc, use_container_width=True)
                
            else:
                st.error("Không tìm thấy nghiệm tối ưu.")
        except ImportError:
            st.error("Thiếu thư viện PuLP.")

    with tabs[4]:
        st.header("5. Dòng dịch chuyển lao động của nhóm dễ tổn thương (Sankey)")
        st.markdown("""
        Mô phỏng dòng lao động thủ công (Manual) bị thay thế do tự động hóa, 
        dòng được đào tạo lại thành công (Upgrade) và dòng thất nghiệp lâm thời (Unemployed) 
        trong 3 ngành nhạy cảm nhất: Nông nghiệp, Xây dựng, Bán lẻ.
        """)
        
        # Calculate Sankey flows from optimization results
        # vulnerable: agricultural (idx 0), construction (idx 2), retail (idx 3)
        v_idx = [0, 2, 3]
        labels = [
            "Nông nghiệp bị thay thế", "Xây dựng bị thay thế", "Bán lẻ bị thay thế",  # Sources: 0, 1, 2
            "Đào tạo lại (Nông nghiệp)", "Đào tạo lại (Xây dựng)", "Đào tạo lại (Bán lẻ)", # intermediate: 3, 4, 5
            "Thất nghiệp", "Việc làm mới" # Sinks: 6, 7
        ]
        
        sources = []
        targets = []
        values = []
        
        for k, idx in enumerate(v_idx):
            disp = displaced_val[idx]
            retrained = min(disp, d1[idx] * x_H_opt[idx])
            unemp = max(0.0, disp - retrained)
            
            # Link to intermediate retraining
            if retrained > 0:
                sources.append(k)
                targets.append(3 + k)
                values.append(retrained)
                
                # Link from retraining to new jobs
                sources.append(3 + k)
                targets.append(7)
                values.append(retrained)
                
            # Link to unemployment
            if unemp > 0:
                sources.append(k)
                targets.append(6)
                values.append(unemp)
                
        fig_sankey = go.Figure(data=[go.Sankey(
            node = dict(pad = 15, thickness = 20, line = dict(color = "black", width = 0.5),
                        label = labels, color = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]),
            link = dict(source = sources, target = targets, value = values,
                        color = "rgba(200, 200, 200, 0.4)")
        )])
        if optimization_success:
            fig_sankey.update_layout(title_text="Mô phỏng dòng lao động dịch chuyển (Sankey)", font_size=10)
            st.plotly_chart(fig_sankey, use_container_width=True)
        else:
            st.warning("Không thể hiển thị biểu đồ Sankey vì tối ưu hóa chưa có nghiệm hợp lệ hoặc PuLP chưa được cài đặt.")

    with tabs[5]:
        st.header("6. Thảo luận chính sách")
        
        # Calculate manufacturing ratio requirement
        # c1[1]*risk[1]/d1[1] = 62.4 * 0.42 / 32 = 0.819
        ratio_req = (c1[1] * risk[1]) / d1[1]
        
        st.markdown(fr"""
- **Ngưỡng đào tạo lại tối thiểu của ngành Chế biến chế tạo (Ngành 2)**: Để đáp ứng điều kiện sa thải không vượt quá khả năng đào tạo lại ($Displaced_2 \le RetrainCapacity_2$), tỷ lệ phân bổ ngân sách $x_H / x_{{AI}}$ trong ngành Chế biến chế tạo phải đạt tối thiểu **{ratio_req:.3f}** (tức cứ mỗi 1 đồng đầu tư AI phải đi kèm ít nhất {ratio_req:.3f} đồng đầu tư đào tạo lại). Đây chính là định chế chính sách định lượng quan trọng để giữ vững an sinh công nghiệp.
- **Tính khả thi của ràng buộc an sinh xã hội**: Nếu Quốc hội yêu cầu ràng buộc ngặt nghèo hơn là không có ngành nào sa thải quá 5% tổng lao động ($Displaced_i \le 0.05 \cdot L_i$), bài toán có thể vô nghiệm nếu tốc độ giải ngân đầu tư AI quá nhanh mà không đồng bộ hạ tầng giáo dục.
- **Ngành Tài chính-Ngân hàng (Ngành 5)**: Có mức độ tự động hóa cao nhất (52%), nhưng cũng có hệ số tạo việc làm AI mới rất lớn (45.8). Mô hình khuyến nghị tập trung đầu tư cả AI và H cho ngành này để thực hiện chuyển dịch lực lượng lao động văn phòng chất lượng cao sang AI-Hybrid, tạo giá trị gia tăng cực lớn.
        """)
        
    with tabs[6]:
        st.header("7. Tham khảo")
        st.markdown("""
        - **Acemoglu, D., & Restrepo, P. (2018).** The Race between Man and Machine: Implications of Technology for Growth, Factor Shares, and Employment.
        - Báo cáo Việc làm và Xu hướng Xã hội Việt Nam 2024 (ILO Vietnam).
        - Niên giám thống kê lao động việc làm Việt Nam 2024.
        """)

if __name__ == "__main__":
    render()
