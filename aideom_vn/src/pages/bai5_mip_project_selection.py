import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def render():
    st.title("🎯 Bài 5 — Tối ưu hóa danh mục dự án (MIP Project Selection)")
    
    st.markdown(r"""
    **Mục tiêu học tập:** Tiếp cận mô hình quy hoạch nguyên hỗn hợp (Mixed-Integer Programming - MIP). 
    Sinh viên hiểu cách đưa các điều kiện logic (Nếu-Thì, Loại trừ nhau, Ràng buộc phụ thuộc) 
    vào mô hình toán học thông qua các biến nhị phân $y_i \in \{0, 1\}$.
    """)

    tabs = st.tabs([
        "📖 Bối cảnh & Vấn đề", 
        "🔬 Mô hình MIP", 
        "🧠 Ràng buộc Logic", 
        "📊 Danh mục 15 dự án", 
        "📈 Kết quả tối ưu", 
        "💡 Thảo luận chính sách",
        "📚 Tham khảo"
    ])
    
    with tabs[0]:
        st.header("1. Bối cảnh & Vấn đề")
        st.markdown("""
        Chính phủ có danh sách 15 dự án đầu tư công nghệ tiềm năng (từ Hạ tầng dữ liệu, Trung tâm AI, 
        đến Đào tạo nhân lực bán dẫn). Mỗi dự án có chi phí đầu tư và lợi ích kỳ vọng (NPV) khác nhau.
        
        **Thách thức:** Ngân sách có hạn, và các dự án có mối quan hệ ràng buộc lẫn nhau. 
        Ví dụ: Không thể xây dựng 2 trung tâm dữ liệu quốc gia cùng lúc ở hai miền (Loại trừ), 
        hoặc phải đầu tư vào Đào tạo nhân lực số trước khi có thể triển khai Siêu máy tính hay bán dẫn (Phụ thuộc).
        """)
        st.info("💡 **Mấu chốt:** Làm sao chọn được 'rổ' dự án có tổng NPV cao nhất mà không vi phạm các logic này?")
        
    with tabs[1]:
        st.header("2. Mô hình toán học MIP")
        st.markdown(r"""
        Gọi $y_i$ là biến nhị phân đại diện cho việc chọn dự án $i$:
        - $y_i = 1$: Nếu dự án $i$ được chọn.
        - $y_i = 0$: Nếu dự án $i$ bị loại.
        
        **Hàm mục tiêu:** Tối đa hóa tổng giá trị hiện tại ròng (NPV):
        """)
        st.latex(r"\max Z = \sum_{i=1}^{15} B_i \cdot y_i")
        st.markdown(r"""
        **Ràng buộc ngân sách tổng:**
        """)
        st.latex(r"\sum_{i=1}^{15} C_i \cdot y_i \le Budget_{Total}")
        st.markdown(r"""
        **Ràng buộc ngân sách Năm 1-2:**
        """)
        st.latex(r"\sum_{i=1}^{15} C_{1,i} \cdot y_i \le Budget_{1-2}")

    with tabs[2]:
        st.header("3. Chuyển đổi Logic sang Phép toán")
        st.markdown("Trong bài tập này, chúng ta xử lý 4 loại ràng buộc logic phổ biến:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Loại trừ (Mutually Exclusive)")
            st.write("Chỉ chọn tối đa 1 trong 2 trung tâm dữ liệu quốc gia (P1 hoặc P2):")
            st.latex(r"y_1 + y_2 \le 1")
            
            st.subheader("Phụ thuộc (Precedence)")
            st.write("Dự án AI (P8) và Bán dẫn (P13) chỉ được làm nếu đã đầu tư đào tạo nhân lực (P12):")
            st.latex(r"y_8 \le y_{12} \quad \text{và} \quad y_{13} \le y_{12}")
        
        with col2:
            st.subheader("Bắt buộc (Mandatory)")
            st.write("An ninh mạng SOC (P14) là bắt buộc quốc gia:")
            st.latex(r"y_{14} = 1")
            
            st.subheader("Giới hạn số lượng (Cardinality)")
            st.write("Chọn tối thiểu 7 dự án và tối đa 11 dự án:")
            st.latex(r"7 \le \sum_{i=1}^{15} y_i \le 11")

    with tabs[3]:
        st.header("4. Danh mục 15 dự án chuyển đổi số chiến lược")
        st.markdown("Số liệu chi phí và lợi ích kỳ vọng (NPV) thực tế theo Bảng 5.2 của đề bài (đơn vị: tỷ VND):")
        
        data = {
            'ID': range(1, 16),
            'Tên dự án': [
                'P1: Trung tâm DLQG Hòa Lạc',
                'P2: Trung tâm DLQG phía Nam',
                'P3: Hệ thống 5G phủ sóng toàn quốc',
                'P4: Hệ thống định danh VNeID 2.0',
                'P5: Cổng dịch vụ công quốc gia v3',
                'P6: Y tế số quốc gia (hồ sơ sức khỏe)',
                'P7: Giáo dục số K-12 toàn quốc',
                'P8: Trung tâm AI quốc gia + supercomputing',
                'P9: Sandbox tài chính số (fintech)',
                'P10: Logistics thông minh + cảng biển số',
                'P11: Nông nghiệp số ĐBSCL',
                'P12: Đào tạo 50.000 kỹ sư AI/bán dẫn',
                'P13: Khu CN bán dẫn Bắc Ninh - Bắc Giang',
                'P14: An ninh mạng quốc gia (SOC)',
                'P15: Open Data + dữ liệu mở quốc gia'
            ],
            'Lĩnh vực': [
                'Hạ tầng', 'Hạ tầng', 'Hạ tầng', 'Chính phủ số', 'Chính phủ số', 
                'Y tế số', 'Giáo dục', 'AI', 'Tài chính số', 'Logistics', 
                'Nông nghiệp', 'Nhân lực', 'Bán dẫn', 'An ninh', 'Dữ liệu'
            ],
            'Cost': [12000, 11500, 18000, 4500, 3200, 5800, 6500, 15000, 2500, 7200, 4800, 8500, 20000, 3800, 1500],
            'Cost_1_2': [8500, 7500, 12000, 3500, 2500, 4000, 4500, 9000, 1800, 5000, 3500, 5500, 13000, 2800, 1200],
            'NPV': [21500, 20800, 32500, 9200, 6800, 11400, 12200, 28500, 5800, 13800, 8500, 16200, 35000, 7500, 3800]
        }
        df_p = pd.DataFrame(data)
        df_p['B/C Ratio'] = (df_p['NPV'] / df_p['Cost']).round(2)
        st.dataframe(df_p.style.highlight_max(subset=['B/C Ratio'], color='#10b981'), use_container_width=True)

    with tabs[4]:
        st.header("5. Kết quả giải bằng PuLP Solver")
        
        selected_ids = []
        
        try:
            import pulp
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                budget = st.slider("Tổng ngân sách 5 năm (Tỷ VND)", 50000, 100000, 80000, step=5000)
            with col_b2:
                budget_12 = st.slider("Ngân sách Năm 1-2 (Tỷ VND)", 25000, 55000, 40000, step=2500)
                
            force_redundancy = st.checkbox("Yêu cầu Redundancy (Chọn cả P1 và P2)", value=False)
            use_risk = st.checkbox("Xét rủi ro dự án (Tối đa hóa lợi ích kỳ vọng E[Z] - Câu 5.4.4)", value=False)
            
            # Solver implementation
            prob = pulp.LpProblem("Project_Selection", pulp.LpMaximize)
            y = pulp.LpVariable.dicts("y", range(1, 16), cat='Binary')
            
            # Objective
            if use_risk:
                p = {
                    1:0.85, 2:0.85, 3:0.85, # Hạ tầng
                    4:0.75, 5:0.75,         # Chính phủ số
                    8:0.65, 12:0.65, 13:0.65, # AI/Bán dẫn
                    6:0.80, 7:0.80, 9:0.80, 10:0.80, 11:0.80, 14:0.80, 15:0.80 # Còn lại
                }
                prob += pulp.lpSum(p[i] * df_p.loc[i-1, 'NPV'] * y[i] for i in range(1, 16))
            else:
                prob += pulp.lpSum(df_p.loc[i-1, 'NPV'] * y[i] for i in range(1, 16))
            
            # Constraints
            prob += pulp.lpSum(df_p.loc[i-1, 'Cost'] * y[i] for i in range(1, 16)) <= budget
            prob += pulp.lpSum(df_p.loc[i-1, 'Cost_1_2'] * y[i] for i in range(1, 16)) <= budget_12
            
            if force_redundancy:
                prob += y[1] == 1
                prob += y[2] == 1
            else:
                prob += y[1] + y[2] <= 1
                
            prob += y[8] <= y[12]   # Precedence AI -> Edu
            prob += y[13] <= y[12]  # Precedence Semi -> Edu
            prob += y[4] + y[5] >= 1
            prob += y[14] == 1      # Mandatory CyberSecurity SOC
            prob += pulp.lpSum(y[i] for i in range(1, 16)) >= 7 # Min projects
            prob += pulp.lpSum(y[i] for i in range(1, 16)) <= 11 # Max projects
            
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if pulp.LpStatus[prob.status] == 'Optimal':
                selected_ids = [i for i in range(1, 16) if pulp.value(y[i]) == 1]
                total_npv = sum(df_p.loc[i-1, 'NPV'] for i in selected_ids)
                total_cost = sum(df_p.loc[i-1, 'Cost'] for i in selected_ids)
                total_cost_12 = sum(df_p.loc[i-1, 'Cost_1_2'] for i in selected_ids)
                
                st.success(f"Tìm thấy phương án tối ưu! Lợi ích mục tiêu tối ưu Z*: **{pulp.value(prob.objective):,.0f}** tỷ VND")
                
                df_selected = df_p[df_p['ID'].isin(selected_ids)].copy()
                df_selected['Status'] = 'Selected'
                
                col1, col2 = st.columns([1, 1.2])
                with col1:
                    st.metric("Số dự án chọn", len(selected_ids))
                    st.metric("Tổng chi phí", f"{total_cost:,.0f} tỷ VND")
                    st.metric("Chi phí năm 1-2", f"{total_cost_12:,.0f} tỷ VND")
                    st.write("**Danh mục được chọn:**")
                    st.write(df_selected[['Tên dự án', 'Cost', 'NPV']])
                    
                with col2:
                    fig = px.scatter(df_p, x='Cost', y='NPV', text='Tên dự án',
                                     color=df_p['ID'].isin(selected_ids).map({True:'Selected', False:'Rejected'}),
                                     color_discrete_map={'Selected':'#10b981', 'Rejected':'#ef4444'},
                                     title="Không gian dự án: Lợi ích vs Chi phí")
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("⚠️ **MÔ HÌNH VÔ NGHIỆM**: Không tìm thấy phương án tối ưu thỏa mãn tất cả các ràng buộc.")
                st.info("💡 **Gợi ý**: Hãy thử nới lỏng ngân sách năm 1-2 hoặc bỏ tùy chọn Redundancy (P1 + P2).")
        except ImportError:
            st.error("⚠️ **Xác định thiếu thư viện 'PuLP'**")
            st.markdown(r"""
            Để chạy mô hình tối ưu hóa này, vui lòng cài đặt thư viện PuLP bằng lệnh sau trong terminal:
            ```bash
            pip install pulp
            ```
            """)
            st.info("Hiển thị dữ liệu danh mục dự án thô (không tối ưu hóa):")
            st.dataframe(df_p)
                
            # Waterfall-like budget consumption
            df_p_sorted = df_p.sort_values('B/C Ratio', ascending=False)
            df_p_sorted['Selected'] = df_p_sorted['ID'].isin(selected_ids) 
            st.subheader("Phân tích hiệu quả đầu tư")
            fig_bar = px.bar(df_p_sorted, x='Tên dự án', y='NPV', color='Selected', 
                             title="Xếp hạng dự án theo hiệu suất (B/C Ratio)")
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi: {e}")
 
    with tabs[5]:
        st.header("6. Thảo luận chính sách")
        st.markdown(r"""
- **Vì sao dự án Open Data (P15) bị loại ở một số kịch bản?** Mặc dù P15 có tỷ suất lợi ích/chi phí (B/C Ratio = 2.53) rất cao, nhưng do ngân sách bị giới hạn và các ràng buộc bắt buộc (như SOC P14, và ràng buộc về đào tạo nguồn nhân lực P12 để kéo theo các dự án lớn như AI P8 hay bán dẫn P13), mô hình buộc phải lựa chọn các dự án mang tính nền tảng khác để thỏa mãn hệ ràng buộc phức tạp của quốc gia.
- **Ràng buộc an ninh mạng SOC (P14)** là bắt buộc và giảm nhẹ NPV tối đa, nhưng đây là "chi phí bảo hiểm" thiết yếu cho sự an toàn của toàn bộ nền kinh tế số.
- **Tác động của Redundancy (P1 và P2)**: Khi ép buộc chọn cả hai trung tâm dữ liệu quốc gia (để dự phòng thảm họa), ngân sách năm 1-2 bị thắt chặt nghiêm trọng ($8500 + 7500 = 16000$ tỷ VND đã chiếm 40% ngân sách năm 1-2). Điều này có thể khiến bài toán vô nghiệm hoặc buộc phải loại bỏ các dự án AI và bán dẫn lớn khác.
- **Cộng hưởng dự án**: Để mô hình hóa hiệu ứng cộng hưởng (ví dụ: làm cả P8 và P12 thì lợi ích tăng thêm 20%), có thể đưa thêm biến nhị phân tích hợp $z = y_8 \cdot y_{12}$ tuyến tính hóa bằng các ràng buộc $z \le y_8$, $z \le y_{12}$, $z \ge y_8 + y_{12} - 1$.
        """)
         
    with tabs[6]:
        st.header("7. Tham khảo")
        st.markdown(r"""
        - **Williams, H. P. (2013):** Model Building in Mathematical Programming.
        - Quyết định số 749/QĐ-TTg về Chương trình Chuyển đổi số quốc gia đến năm 2025, định hướng đến năm 2030.
        - PuLP Documentation: Linear and Integer Programming in Python.
        """)

if __name__ == "__main__":
    render()
