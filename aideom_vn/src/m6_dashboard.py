import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from aideom_vn.src.data_loader import load_macro, load_sectors, load_regions, load_priorities
from aideom_vn.src.m1_production import compute_tfp, growth_decomposition, scenario_2030
from aideom_vn.src.m2_readiness import topsis, entropy_weights
from aideom_vn.src.m3_allocation import solve_lp_pulp, solve_allocation_pulp, solve_project_selection
from aideom_vn.src.m4_labor import simulate_labor_dynamics, solve_netjob_maximization
from aideom_vn.src.m5_risk import calculate_risk_metrics, solve_minimax_regret_pulp

def render_tab1_overview():
    st.header("📈 Tổng quan Kinh tế vĩ mô & Năng lực số hóa vùng miền")
    st.markdown("""
    Tab này hiển thị các chỉ số kinh tế vĩ mô cốt lõi của Việt Nam giai đoạn 2020-2025 và bản đồ so sánh mức độ sẵn sàng công nghệ AI / Chuyển đổi số giữa 6 vùng kinh tế - xã hội.
    """)
    
    # KPIs vĩ mô
    df_macro = load_macro()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="GDP thực tế 2025", value=f"{df_macro.iloc[-1]['GDP_trillion_VND']:,} tỷ VND", delta="8.02% (2025)")
    with col2:
        st.metric(label="Vốn lũy kế 2025", value=f"{df_macro.iloc[-1]['K_trillion_VND']:,} tỷ VND")
    with col3:
        st.metric(label="Lực lượng lao động", value=f"{df_macro.iloc[-1]['L_million']:.1f} triệu người")
    with col4:
        st.metric(label="Tỷ lệ số hóa (D)", value=f"{df_macro.iloc[-1]['D_digital_pct']:.1f}%")
        
    st.subheader("📊 Xu hướng tăng trưởng kinh tế & TFP Việt Nam")
    # Biểu đồ tăng trưởng GDP vs TFP
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_macro['year'], y=df_macro['GDP_trillion_VND'], name="GDP (tỷ VND)", yaxis="y1", line=dict(color='royalblue', width=3)))
    
    # Tính TFP
    tfp_vals = compute_tfp(
        df_macro['GDP_trillion_VND'].values, df_macro['K_trillion_VND'].values, df_macro['L_million'].values,
        df_macro['D_digital_pct'].values, df_macro['AI_tech_firms_thousand'].values, df_macro['H_trained_pct'].values
    )
    fig.add_trace(go.Scatter(x=df_macro['year'], y=tfp_vals, name="TFP", yaxis="y2", line=dict(color='darkorange', width=3, dash='dash')))
    
    fig.update_layout(
        title="Tương quan GDP và Năng suất nhân tố tổng hợp (TFP) 2020-2025",
        xaxis=dict(title="Năm"),
        yaxis=dict(title="GDP (nghìn tỷ VND)", titlefont=dict(color="royalblue"), tickfont=dict(color="royalblue")),
        yaxis2=dict(title="Chỉ số TFP", titlefont=dict(color="darkorange"), tickfont=dict(color="darkorange"), anchor="x", overlaying="y", side="right"),
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🗺️ Bản đồ nhiệt sẵn sàng AI & Số hóa 6 Vùng kinh tế")
    df_regions = load_regions()
    fig_region = px.bar(
        df_regions,
        x='region_name_vi',
        y=['digital_index_0_100', 'ai_readiness_0_100', 'trained_labor_pct'],
        barmode='group',
        title="So sánh Chỉ số Số hóa, Sẵn sàng AI & Nhân lực qua đào tạo 6 Vùng miền",
        labels={'value': 'Điểm số / Tỷ lệ (%)', 'region_name_vi': 'Vùng kinh tế'},
        template="plotly_dark"
    )
    st.plotly_chart(fig_region, use_container_width=True)

def render_tab2_allocation():
    st.header("🎯 Tối ưu hóa phân bổ Ngân sách vĩ mô (LP)")
    st.markdown("""
    Sử dụng thuật toán Quy hoạch tuyến tính (LP) để tối ưu hóa việc phân chia ngân sách công giữa 4 hạng mục chính nhằm cực đại hóa hiệu quả kinh tế bổ sung.
    """)
    
    budget = st.slider("Tổng ngân sách đầu tư công nghệ (nghìn tỷ VND)", min_value=80.0, max_value=250.0, value=100.0, step=10.0)
    
    x_opt, Z_opt, shadow_prices = solve_lp_pulp(budget)
    
    # Hiển thị kết quả
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("💡 Kết quả phân bổ vốn tối ưu")
        categories = ['Hạ tầng vật chất (x1)', 'Công nghệ AI (x2)', 'Nhân lực đào tạo (x3)', 'Nghiên cứu R&D (x4)']
        df_res = pd.DataFrame({
            'Hạng mục': categories,
            'Phân bổ (tỷ VND)': x_opt,
            'Tỷ lệ (%)': (x_opt / budget) * 100
        })
        st.dataframe(df_res.style.format({'Phân bổ (tỷ VND)': '{:,.2f}', 'Tỷ lệ (%)': '{:.2f}%'}))
        st.success(f"**GDP tăng thêm cực đại (Z*):** {Z_opt:.2f} nghìn tỷ VND")
        
    with col2:
        st.subheader("📊 Trực quan hóa cơ cấu đầu tư")
        fig = px.pie(df_res, values='Phân bổ (tỷ VND)', names='Hạng mục', title="Cơ cấu phân bổ ngân sách tối ưu", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("🔍 Phân tích Giá đối ngẫu (Shadow Prices)")
    st.markdown("Giá đối ngẫu cho biết mức tăng trưởng GDP biên khi nới lỏng thêm một đơn vị ràng buộc tương ứng:")
    for constraint, sp in shadow_prices.items():
        st.info(f"👉 **Ràng buộc {constraint}:** {sp:.4f} (GDP tăng thêm trên mỗi nghìn tỷ VND đầu tư thêm)")

def render_tab3_scenarios():
    st.header("🎭 So sánh kịch bản phát triển & Biên tối ưu Pareto")
    st.markdown("""
    Tab này hiển thị kết quả phân tích đa tiêu chí (MCDM) xếp hạng 10 ngành kinh tế và so sánh 3 kịch bản: Cơ sở, Tăng trưởng nhanh và Bao trùm xã hội.
    """)
    
    df_sectors = load_sectors()
    df_priorities = load_priorities()
    
    # Chạy TOPSIS cho các kịch bản
    criteria_cols = ['growth_rate_2024_pct', 'productivity_million_VND_per_worker', 
                     'spillover_coef_0_1', 'export_billion_USD', 'labor_million', 
                     'ai_readiness_0_100', 'automation_risk_pct']
    X = df_sectors[criteria_cols].values
    is_benefit = [True, True, True, True, True, True, False]
    
    # 1. Mặc định
    w_def = df_priorities.iloc[0, 1:8].values.astype(float)
    scores_def, _ = topsis(X, w_def, is_benefit)
    # 2. Tăng trưởng
    w_growth = df_priorities.iloc[1, 1:8].values.astype(float)
    scores_growth, _ = topsis(X, w_growth, is_benefit)
    # 3. Bao trùm
    w_inc = df_priorities.iloc[2, 1:8].values.astype(float)
    scores_inc, _ = topsis(X, w_inc, is_benefit)
    
    df_compare = pd.DataFrame({
        'Ngành kinh tế': df_sectors['sector_name_vi'],
        'Điểm Mặc định': scores_def,
        'Điểm Tăng trưởng': scores_growth,
        'Điểm Bao trùm': scores_inc
    })
    
    st.subheader("📊 Xếp hạng ưu tiên ngành kinh tế theo kịch bản")
    st.dataframe(df_compare.style.format({'Điểm Mặc định': '{:.4f}', 'Điểm Tăng trưởng': '{:.4f}', 'Điểm Bao trùm': '{:.4f}'}))
    
    fig = px.bar(
        df_compare,
        x='Ngành kinh tế',
        y=['Điểm Mặc định', 'Điểm Tăng trưởng', 'Điểm Bao trùm'],
        barmode='group',
        title="So sánh Điểm ưu tiên phát triển ngành giữa các kịch bản chính sách",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_tab4_labor():
    st.header("👷 Mô phỏng luồng Lao động & Tối ưu việc làm ròng")
    st.markdown("""
    Tab này hiển thị sự biến động của thị trường lao động dưới tác động của AI: Tạo việc làm số mới và Thay thế/Sa thải lao động tự động hóa.
    """)
    
    budget = st.slider("Ngân sách hỗ trợ đào tạo lại & chuyển đổi nghề nghiệp (tỷ VND)", 30000, 100000, 60000, 5000)
    
    # Giải bài toán tối ưu việc làm ròng
    x_opt, Z_net = solve_netjob_maximization(budget)
    df_sectors = load_sectors()
    
    df_labor_res = simulate_labor_dynamics(df_sectors, x_opt)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("💼 Việc làm mới và việc làm bị thay thế theo ngành")
        st.dataframe(df_labor_res[['sector_name_vi', 'jobs_created_thousand', 'jobs_displaced_thousand', 'net_jobs_thousand']].style.format({
            'jobs_created_thousand': '{:,.1f}',
            'jobs_displaced_thousand': '{:,.1f}',
            'net_jobs_thousand': '{:,.1f}'
        }))
        st.success(f"**Tổng số việc làm ròng (Net Job) tạo thêm:** {Z_net:.2f} nghìn lao động")
        
    with col2:
        st.subheader("📊 Biểu đồ việc làm ròng theo từng ngành")
        fig = px.bar(
            df_labor_res,
            x='sector_name_vi',
            y='net_jobs_thousand',
            color='net_jobs_thousand',
            color_continuous_scale='RdBu',
            title="Số việc làm ròng tạo thêm (nghìn người) của từng ngành kinh tế",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

def render_tab5_risk():
    st.header("🎲 Quản trị rủi ro vĩ mô & Lập kế hoạch ngẫu nhiên vững chắc")
    st.markdown("""
    Sử dụng Quy hoạch ngẫu nhiên hai giai đoạn để giải bài toán phân bổ ngân sách dưới rủi ro bất định thị trường toàn cầu.
    """)
    
    metrics = calculate_risk_metrics()
    
    # KPIs đo lường rủi ro
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Lợi nhuận ngẫu nhiên kỳ vọng (RP)", value=f"{metrics['Z_RP']:.2f} tỷ VND")
    with col2:
        st.metric(label="Giá trị giải pháp ngẫu nhiên (VSS)", value=f"{metrics['VSS']:.2f} tỷ VND", help="Giá trị thu được khi chủ động quản trị rủi ro")
    with col3:
        st.metric(label="Giá trị thông tin hoàn hảo (EVPI)", value=f"{metrics['EVPI']:.2f} tỷ VND", help="Số ngân sách tối đa nên chi cho hoạt động dự báo")
        
    # Minimax Regret
    x_regret, max_regret = solve_minimax_regret_pulp()
    
    st.subheader("🛡️ Lựa chọn đầu tư vững chắc (Minimax Regret)")
    st.markdown(f"Quyết định đầu tư vững chắc giúp giảm thiểu tối đa sự hối tiếc lớn nhất trong mọi tình huống xuống còn **{max_regret:.2f} tỷ VND**:")
    
    df_regret = pd.DataFrame({
        'Hạng mục đầu tư': ['Hạ tầng vật lý (x1)', 'Công nghệ AI (x2)', 'Đào tạo nhân lực (x3)'],
        'Quyết định RP (tỷ VND)': metrics['x_RP'],
        'Quyết định EV tất định (tỷ VND)': metrics['x_EV'],
        'Quyết định Vững chắc (tỷ VND)': x_regret
    })
    st.dataframe(df_regret.style.format({
        'Quyết định RP (tỷ VND)': '{:,.2f}',
        'Quyết định EV tất định (tỷ VND)': '{:,.2f}',
        'Quyết định Vững chắc (tỷ VND)': '{:,.2f}'
    }))

def render_tab6_policy():
    st.header("📋 Khuyến nghị chính sách & Tải báo cáo")
    st.markdown("""
    Dựa trên kết quả chạy mô hình quyết định **AIDEOM-VN**, chúng tôi đề xuất các khuyến nghị chính sách cụ thể cho Việt Nam đến năm 2030:
    """)
    
    st.warning("⚠️ **Khuyến nghị 1 (Đào tạo nhân lực là ưu tiên số 1):** Bắt buộc phải duy trì ngưỡng đầu tạo lại tối thiểu cho ngành Chế tạo để bảo đảm việc làm ròng không bị âm trước làn sóng tự động hóa.")
    st.info("ℹ️ **Khuyến nghị 2 (Nâng cao TFP bền vững):** Việc chuyển đổi số sâu rộng (D) đóng góp hiệu quả biên cao nhất cho GDP dài hạn. Cần ưu tiên phát triển 100 nghìn doanh nghiệp công nghệ số theo đúng Quyết định 749.")
    st.success("✅ **Khuyến nghị 3 (Phát triển cân bằng vùng miền):** Áp dụng ràng buộc công bằng số hóa (lambda = 0.7) để tránh hố sâu khoảng cách số giữa vùng Đông Nam Bộ và Tây Nguyên.")
    
    st.subheader("📥 Xuất báo cáo dữ liệu")
    st.markdown("Bạn có thể tải các bộ dữ liệu mô phỏng đã được chạy dưới định dạng CSV:")
    
    # Tạo mock CSV để tải
    df_macro = load_macro()
    csv_macro = df_macro.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Tải báo cáo Vĩ mô Việt Nam (CSV)",
        data=csv_macro,
        file_name='vietnam_macro_report.csv',
        mime='text/csv',
    )
