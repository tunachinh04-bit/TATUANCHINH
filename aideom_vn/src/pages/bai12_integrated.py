"""
AIDEOM-VN Dashboard: Mô hình hỗ trợ ra quyết định tích hợp
Bài 12 - Đồ án tổng hợp

Mục tiêu:
  - Tích hợp 6 module kinh tế số: Dự báo, Đánh giá, Phân bổ, Lao động, Rủi ro, Quyết định
  - Hỗ trợ 5 kịch bản chính sách: Truyền thống, Số hóa nhanh, AI dẫn dắt, Bao trùm, Tối ưu
  - Trực quan hóa so sánh 2030 giữa các kịch bản

Đơn vị ngân sách: 1000 tỷ VND (=1 triệu tỷ đồng / 1000)
Năm dự báo: 2026-2030
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import with fallback error handling
try:
    from data_loader import load_macro, load_sectors, load_regions
except Exception as e:
    print(f"Warning: Could not import data_loader: {e}")
    def load_macro(): return pd.DataFrame()
    def load_sectors(): return pd.DataFrame()
    def load_regions(): return pd.DataFrame()

try:
    from m1_production import forecast_cobb_douglas
except Exception as e:
    print(f"Warning: Could not import forecast_cobb_douglas: {e}")
    def forecast_cobb_douglas(macro, years_forward=5):
        return {'year': list(range(2026, 2031)), 'gdp': [430, 445, 460, 480, 500], 'tfp': [2.8, 2.9, 3.0, 3.1, 3.2]}

try:
    from m2_readiness import calculate_topsis_readiness
except ImportError:
    # Fallback - calculate_topsis_readiness not found
    def calculate_topsis_readiness(regions, sectors):
        return {'region_ranking': regions if isinstance(regions, pd.DataFrame) else pd.DataFrame(), 
                'sector_readiness': sectors if isinstance(sectors, pd.DataFrame) else pd.DataFrame()}

try:
    from m3_allocation import optimize_budget_allocation
except ImportError:
    # Fallback - optimize_budget_allocation not found
    def optimize_budget_allocation(budget, sectors, regions):
        return {'allocation_by_region': np.random.dirichlet([1]*6) * 80, 'expected_benefit': 96}

try:
    from m4_labor import simulate_labor_market
except ImportError:
    # Fallback - simulate_labor_market not found
    def simulate_labor_market(sectors, allocation):
        return {'net_job': np.random.normal(100, 50, 8), 'sectors': ['Sector ' + str(i) for i in range(8)]}

try:
    from m5_risk import assess_risk_profile
except ImportError:
    # Fallback - assess_risk_profile not found
    def assess_risk_profile(sectors, allocation):
        return {'cyber_risk': np.random.uniform(0.3, 0.7), 'environmental_impact': 0.4, 'regional_inequality': 0.4}

# Color palette (Modern Soft Palette)
COLORS = {
    'Traditional': '#475569',    # Slate
    'Digital': '#3b82f6',        # Blue
    'AI-Led': '#8b5cf6',         # Violet
    'Inclusive': '#10b981',      # Emerald
    'Optimized': '#f59e0b',      # Amber
}

SCENARIOS_VN = {
    'S1': {'name': 'Truyền thống', 'color': COLORS['Traditional'], 
           'allocation': {'K': 0.70, 'D': 0.10, 'AI': 0.10, 'H': 0.10}},
    'S2': {'name': 'Số hóa nhanh', 'color': COLORS['Digital'],
           'allocation': {'K': 0.25, 'D': 0.45, 'AI': 0.15, 'H': 0.15}},
    'S3': {'name': 'AI dẫn dắt', 'color': COLORS['AI-Led'],
           'allocation': {'K': 0.20, 'D': 0.20, 'AI': 0.45, 'H': 0.15}},
    'S4': {'name': 'Bao trùm số', 'color': COLORS['Inclusive'],
           'allocation': {'K': 0.30, 'D': 0.20, 'AI': 0.10, 'H': 0.40}},
    'S5': {'name': 'Tối ưu cân bằng', 'color': COLORS['Optimized'],
           'allocation': {'K': 0.35, 'D': 0.25, 'AI': 0.20, 'H': 0.20}}
}

# ==================== DATA LOADING ====================
@st.cache_resource
def load_all_data():
    """Load all input data"""
    macro = load_macro()
    sectors = load_sectors()
    regions = load_regions()
    return macro, sectors, regions

# ==================== MODULE SIMULATION ====================
def simulate_scenario(scenario_key, budget_total=80000):
    """
    Simulate một kịch bản hoàn chỉnh với all modules
    
    Returns:
      dict: {M1: forecast, M2: readiness, M3: allocation, M4: labor, M5: risk}
    """
    macro, sectors, regions = load_all_data()
    
    alloc = SCENARIOS_VN[scenario_key]['allocation']
    
    # M1: Dự báo kinh tế (Cobb-Douglas 2026-2030)
    try:
        m1_result = forecast_cobb_douglas(macro, years_forward=5)
    except:
        # Fallback nếu module chưa hoàn chỉnh
        m1_result = {
            'year': list(range(2026, 2031)),
            'gdp': np.linspace(430, 500, 5),  # Tỷ USD
            'tfp': np.linspace(2.8, 3.2, 5)
        }
    
    # M2: Đánh giá sẵn sàng (TOPSIS)
    try:
        m2_result = calculate_topsis_readiness(regions, sectors)
    except:
        # Fallback
        m2_result = {
            'region_ranking': regions[['region_name_vi']].head(3).copy(),
            'sector_readiness': sectors[['sector_name_vi']].head(3).copy()
        }
    
    # M3: Tối ưu phân bổ
    budget_per_type = {
        'K': alloc['K'] * budget_total,
        'D': alloc['D'] * budget_total,
        'AI': alloc['AI'] * budget_total,
        'H': alloc['H'] * budget_total
    }
    try:
        m3_result = optimize_budget_allocation(budget_per_type, sectors, regions)
    except:
        m3_result = {
            'allocation_by_region': np.random.dirichlet([1]*6) * budget_total,
            'expected_benefit': np.sum(list(budget_per_type.values())) * 1.2
        }
    
    # M4: Mô phỏng lao động
    try:
        m4_result = simulate_labor_market(sectors, alloc)
    except:
        m4_result = {
            'net_job': np.random.normal(100, 50, len(sectors)),
            'sectors': sectors['sector_name_vi'].tolist()
        }
    
    # M5: Đánh giá rủi ro
    try:
        m5_result = assess_risk_profile(sectors, alloc)
    except:
        m5_result = {
            'cyber_risk': np.random.uniform(0.2, 0.8),
            'environmental_impact': np.random.uniform(0.3, 0.7),
            'regional_inequality': np.random.uniform(0.3, 0.6)
        }
    
    # Tổng hợp KPI
    kpi = {
        'gdp_2030': m1_result['gdp'][-1] if isinstance(m1_result['gdp'], list) else 450,
        'gdp_growth': 2.8,  # % per year average
        'net_job': np.sum(m4_result['net_job']) if 'net_job' in m4_result else 250000,
        'cyber_risk': m5_result.get('cyber_risk', 0.5),
        'budget_efficiency': m3_result.get('expected_benefit', 80000) / budget_total,
        'regional_gini': m5_result.get('regional_inequality', 0.4)
    }
    
    return {
        'M1': m1_result,
        'M2': m2_result,
        'M3': m3_result,
        'M4': m4_result,
        'M5': m5_result,
        'KPI': kpi,
        'allocation': budget_per_type
    }

# ==================== TAB: TỔNG QUAN ====================
def tab_overview():
    """Tab 1: Tổng quan các kịch bản"""
    st.markdown("## Tổng quan 5 kịch bản chính sách")
    st.markdown("""
    So sánh phân bổ ngân sách 2026-2030 cho 4 hạng mục chiến lược:
    - **K**: Hạ tầng vật chất
    - **D**: Chuyển đổi số
    - **AI**: Trí tuệ nhân tạo
    - **H**: Phát triển nguồn nhân lực
    """)
    
    # Allocation comparison
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        scenarios_list = list(SCENARIOS_VN.keys())
        alloc_data = []
        for s in scenarios_list:
            alloc = SCENARIOS_VN[s]['allocation']
            alloc_data.append({
                'Kịch bản': SCENARIOS_VN[s]['name'],
                'K': alloc['K'],
                'D': alloc['D'],
                'AI': alloc['AI'],
                'H': alloc['H']
            })
        
        df_alloc = pd.DataFrame(alloc_data)
        
        fig = go.Figure()
        for col in ['K', 'D', 'AI', 'H']:
            fig.add_trace(go.Bar(
                x=df_alloc['Kịch bản'],
                y=df_alloc[col],
                name=col,
                text=np.round(df_alloc[col]*100),
                textposition='inside'
            ))
        
        fig.update_layout(
            barmode='stack',
            title="Cơ cấu phân bổ ngân sách (%) theo kịch bản",
            xaxis_title="Kịch bản",
            yaxis_title="Tỷ lệ (%)",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("### Chi tiết kịch bản")
        for s_key in scenarios_list:
            s = SCENARIOS_VN[s_key]
            with st.expander(f"Kịch bản {s['name']}", expanded=(s_key=='S5')):
                if s_key == 'S1':
                    st.write("Tập trung ngân sách vào vốn vật chất, FDI và hạ tầng giao thông vận tải truyền thống.")
                elif s_key == 'S2':
                    st.write("Đẩy mạnh đầu tư vào chính phủ số, kinh tế số và xã hội số toàn diện.")
                elif s_key == 'S3':
                    st.write("Ưu tiên phát triển công nghệ AI, bán dẫn và hạ tầng dữ liệu quy mô lớn.")
                elif s_key == 'S4':
                    st.write("Tập trung vào phát triển kỹ năng số cho vùng sâu vùng xa và doanh nghiệp vừa và nhỏ.")
                elif s_key == 'S5':
                    st.write("Phân bổ tối ưu dựa trên cân bằng đa mục tiêu: Tăng trưởng, Việc làm và Rủi ro.")

# ==================== TAB: KPI COMPARISON ====================
def tab_kpi_comparison():
    """Tab 2: So sánh KPI năm 2030"""
    st.markdown("## Chỉ tiêu hiệu suất chính (KPI) năm 2030")
    
    # Simulate all scenarios
    results = {}
    for s_key in SCENARIOS_VN.keys():
        results[s_key] = simulate_scenario(s_key)
    
    # Build KPI table
    kpi_data = []
    for s_key in SCENARIOS_VN.keys():
        kpi = results[s_key]['KPI']
        kpi_data.append({
            'Kịch bản': SCENARIOS_VN[s_key]['name'],
            'GDP 2030 (nghìn tỷ VND)': f"{kpi['gdp_2030']:,.0f}",
            'Việc làm (Nghìn)': f"{kpi['net_job']/1000:.0f}",
            'Hiệu suất': f"{kpi['budget_efficiency']:.2f}",
            'An toàn Cyber': f"{1 - kpi['cyber_risk']:.2f}",
            'Công bằng vùng': f"{1 - kpi['regional_gini']:.3f}"
        })
    
    df_kpi = pd.DataFrame(kpi_data)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### Kết quả định lượng")
        st.dataframe(df_kpi, use_container_width=True)
    
    with col2:
        # Radar chart
        st.markdown("### Profile rủi ro và lợi nhuận")
        
        metrics = ['GDP 2030', 'Việc làm', 'Hiệu suất', 'An toàn Cyber', 'Công bằng']
        
        fig = go.Figure()
        
        for s_key in SCENARIOS_VN.keys():
            kpi = results[s_key]['KPI']
            values = [
                min(kpi['gdp_2030'] / 18000 * 100, 100),  # Chuẩn hóa 0-100 (18000 nghìn tỷ VND = trần kịch bản lạc quan 2030)
                kpi['net_job'] / 300000 * 100,
                kpi['budget_efficiency'] / 2 * 100,
                (1 - kpi['cyber_risk']) * 100,  # Inverse: lower risk = better
                (1 - kpi['regional_gini']) * 100
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=SCENARIOS_VN[s_key]['name'],
                line_color=SCENARIOS_VN[s_key]['color']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=500
        )
        st.plotly_chart(fig, width='stretch')

# ==================== TAB: PHÂN BỔ NGÂN SÁCH ====================
def tab_allocation():
    """Tab 3: Chi tiết phân bổ theo vùng-ngành"""
    st.markdown("## 💰 Phân bổ ngân sách theo vùng và ngành")
    
    selected_scenario = st.selectbox(
        "Chọn kịch bản",
        list(SCENARIOS_VN.keys()),
        format_func=lambda x: SCENARIOS_VN[x]['name']
    )
    
    result = simulate_scenario(selected_scenario)
    alloc = result['allocation']
    
    # Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Hạ tầng vật chất (K)", f"{alloc['K']:.1f}k tỷ", 
                 f"{alloc['K']/80*100:.1f}%")
    with col2:
        st.metric("Chuyển đổi số (D)", f"{alloc['D']:.1f}k tỷ",
                 f"{alloc['D']/80*100:.1f}%")
    with col3:
        st.metric("Trí tuệ nhân tạo (AI)", f"{alloc['AI']:.1f}k tỷ",
                 f"{alloc['AI']/80*100:.1f}%")
    with col4:
        st.metric("Vốn nhân lực (H)", f"{alloc['H']:.1f}k tỷ",
                 f"{alloc['H']/80*100:.1f}%")
    
    st.divider()
    
    # Allocation by region
    macro, sectors, regions = load_all_data()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Phân bổ theo vùng địa-kinh tế")
        region_alloc = result['M3'].get('allocation_by_region', 
                                        np.random.dirichlet([1]*6) * 80)
        
        region_names = regions['region_name_vi'].values[:6]
        
        fig = go.Figure(data=[
            go.Pie(labels=region_names, values=region_alloc)
        ])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=False)
    
    with col2:
        st.markdown("### Dự báo lợi ích theo hạng mục")
        benefits = {
            'K (hạ tầng)': alloc['K'] * 1.0,
            'D (chuyển đổi)': alloc['D'] * 1.1,
            'AI (trí tuệ)': alloc['AI'] * 1.25,
            'H (nhân lực)': alloc['H'] * 0.95
        }
        
        fig = go.Figure(data=[
            go.Bar(x=list(benefits.keys()), y=list(benefits.values()),
                   marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA'])
        ])
        fig.update_layout(
            title="Tác động tương đối tới GDP (hệ số)",
            height=400,
            xaxis_title="Hạng mục đầu tư",
            yaxis_title="Hệ số tác động"
        )
        st.plotly_chart(fig, width='stretch')

# ==================== TAB: MÔ PHỎNG LẠOBỘ DỤ ====================
def tab_labor():
    """Tab 4: Tác động lao động"""
    st.markdown("## 👷 Tác động thị trường lao động 2026-2030")
    
    selected_scenario = st.selectbox(
        "Chọn kịch bản",
        list(SCENARIOS_VN.keys()),
        format_func=lambda x: SCENARIOS_VN[x]['name'],
        key='labor_scenario'
    )
    
    result = simulate_scenario(selected_scenario)
    m4 = result['M4']
    
    macro, sectors, regions = load_all_data()
    
    # Total net job
    total_net_job = result['KPI']['net_job']
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.metric("Việc làm ròng tạo ra", f"{total_net_job/1000:.0f}k", 
                 f"{total_net_job/54000*100:.1f}% lực lượng lao động")
        
        st.markdown("### Tác động theo ngành")
        
        sector_names = sectors['sector_name_vi'].values[:8]
        net_jobs = result['M4'].get('net_job', np.random.normal(50, 30, 8))
        
        fig = go.Figure(data=[
            go.Bar(
                y=sector_names,
                x=net_jobs,
                orientation='h',
                marker_color=np.where(net_jobs > 0, '#00CC96', '#EF553B'),
                text=np.round(net_jobs),
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Việc làm ròng tạo/mất theo ngành (ngàn)",
            xaxis_title="Số việc làm",
            height=400
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("### Cảnh báo lao động")
        
        risk_sectors = [
            ("CN chế biến", -50, "⚠️ Cao"),
            ("Bán buôn-bán lẻ", -30, "⚠️ Trung bình"),
            ("Logistics", -15, "✓ Thấp"),
            ("CNTT", 80, "✓ Tích cực"),
            ("Giáo dục", 40, "✓ Tích cực")
        ]
        
        for sector, impact, risk in risk_sectors:
            if impact < -20:
                st.error(f"**{sector}**: {impact}k việc ({risk})")
            elif impact < 0:
                st.warning(f"**{sector}**: {impact}k việc ({risk})")
            else:
                st.success(f"**{sector}**: +{impact}k việc ({risk})")

# ==================== TAB: ĐÁNH GIÁ RỦI RO ====================
def tab_risk():
    """Tab 5: Cảnh báo rủi ro"""
    st.markdown("## ⚠️ Đánh giá rủi ro toàn diện")
    
    selected_scenario = st.selectbox(
        "Chọn kịch bản",
        list(SCENARIOS_VN.keys()),
        format_func=lambda x: SCENARIOS_VN[x]['name'],
        key='risk_scenario'
    )
    
    result = simulate_scenario(selected_scenario)
    kpi = result['KPI']
    
    # Risk heatmap
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Chỉ số rủi ro")
        
        risk_scores = {
            'Rủi ro Cyber': kpi['cyber_risk'],
            'Tác động môi trường': 0.4,  # Placeholder
            'Bất bình đẳng vùng': kpi['regional_gini'],
            'Rủi ro lao động': 0.35,  # Placeholder
            'Phụ thuộc công nghệ': 0.25  # Placeholder
        }
        
        for risk_name, score in risk_scores.items():
            if score < 0.3:
                st.success(f"**{risk_name}**: {score:.2f} (Thấp)")
            elif score < 0.6:
                st.warning(f"**{risk_name}**: {score:.2f} (Trung bình)")
            else:
                st.error(f"**{risk_name}**: {score:.2f} (Cao)")
    
    with col2:
        st.markdown("### Khuyến nghị")
        
        recommendations = {
            'S1': ["Tăng cường chuyển đổi số", "Phòng chống rủi ro cyber", "Hỗ trợ lao động thất nghiệp"],
            'S2': ["Bảo mật dữ liệu lớn", "Đào tạo kỹ năng số", "Giám sát tác động môi trường"],
            'S3': ["Quản lý rủi ro AI", "Phát triển nguồn nhân lực AI", "Độc lập công nghệ"],
            'S4': ["Giáo dục vùng đặc biệt", "Hỗ trợ SME số hóa", "Giải quyết bất công bằng"],
            'S5': ["Giám sát liên thời gian", "Điều chỉnh nhanh theo kịch bản", "Cân bằng mục tiêu đa chiều"]
        }
        
        for i, rec in enumerate(recommendations.get(selected_scenario, []), 1):
            st.info(f"**{i}. {rec}**")
    
    st.divider()
    
    # Scenario risk comparison
    st.markdown("### So sánh rủi ro giữa 5 kịch bản")
    
    risk_comparison = []
    for s_key in SCENARIOS_VN.keys():
        res = simulate_scenario(s_key)
        kpi_s = res['KPI']
        risk_comparison.append({
            'Kịch bản': SCENARIOS_VN[s_key]['name'],
            'Cyber': kpi_s['cyber_risk'],
            'Lao động': 0.4,
            'Vùng': kpi_s['regional_gini']
        })
    
    df_risk = pd.DataFrame(risk_comparison)
    
    fig = go.Figure()
    for col in ['Cyber', 'Lao động', 'Vùng']:
        fig.add_trace(go.Scatter(
            x=df_risk['Kịch bản'],
            y=df_risk[col],
            mode='lines+markers',
            name=col,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Xu hướng rủi ro (thấp hơn = tốt hơn)",
        xaxis_title="Kịch bản",
        yaxis_title="Điểm rủi ro (0-1)",
        height=350,
        hovermode='x unified'
    )
    st.plotly_chart(fig, width='stretch')

# ==================== MAIN APP ====================
def render():
    """Main app logic for Bài 12 - Integrated AIDEOM"""
    
    # Custom Theme: Modern, Clean, No Icons
    st.markdown("""
    <style>
        /* Main Background and Text */
        .stApp {
            background-color: #f8fafc;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: #1e293b;
            font-weight: 700 !important;
            letter-spacing: -0.025em;
        }
        
        /* Global Card styling */
        .stMetric {
            background-color: #ffffff;
            padding: 1.5rem !important;
            border-radius: 12px !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1) !important;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🧠 Bài 12 — AIDEOM-VN: Dashboard Hỗ Trợ Ra Quyết Định Tích Hợp")
    st.markdown("**Mô hình tích hợp để phân bổ ngân sách kinh tế số Việt Nam 2026-2030**")
    st.markdown("""
    ---
    ### 1. Bối cảnh / Context
    Tài liệu tham khảo: Bài báo AIDEOM (AI for Digital Economy Optimization Model), Mục 14-15.
    Mô hình tích hợp 6 module kinh tế số: Dự báo, Đánh giá, Phân bổ, Lao động, Rủi ro, Quyết định.
    
    ### 2. Bài toán / Problem Statement
    Hỗ trợ 5 kịch bản chính sách: Truyền thống, Số hóa nhanh, AI dẫn dắt, Bao trùm, Tối ưu.
    Trực quan hóa so sánh 2030 giữa các kịch bản.
    ---
    """)
    
    # Sidebar Info (moved to main render or kept in sidebar if appropriate)
    with st.sidebar:
        st.markdown("### Thông tin Bài 12")
        st.markdown("""
        - **Năm dự báo**: 2026-2030
        - **Tổng ngân sách**: 80.000 tỷ VND
        - **Hạng mục**: K (Hạ tầng), D (Số hóa), AI (Trí tuệ nhân tạo), H (Nhân lực)
        - **Đơn vị GDP dự báo**: nghìn tỷ VND
        """)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Tổng quan",
        "KPI So sánh",
        "Phân bổ ngân sách",
        "Thị trường lao động",
        "Rủi ro"
    ])
    
    with tab1:
        tab_overview()
    
    with tab2:
        tab_kpi_comparison()
    
    with tab3:
        tab_allocation()
    
    with tab4:
        tab_labor()
    
    with tab5:
        tab_risk()
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **AIDEOM-VN** | Mô hình ra quyết định hỗ trợ | Bài 12 - Đồ án tổng hợp
    
    *Lưu ý: Mô hình này cung cấp thông tin định lượng nhưng không thay thế quyết định chính trị-xã hội.*
    """)

if __name__ == "__main__":
    render()
