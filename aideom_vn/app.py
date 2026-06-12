# -*- coding: utf-8 -*-
"""
==============================================================================
AIDEOM-VN  |  AI-Driven Economic Decision Optimization Model for Vietnam
Hệ thống hỗ trợ ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI
------------------------------------------------------------------------------
Bài tập lớn: Các mô hình ra quyết định
Sinh viên : Tạ Tuấn Chinh      |     Mã sinh viên: 23051191
Dữ liệu thực tế Việt Nam 2020-2025 (GSO, World Bank, MoST, MIC, MPI, GII)
------------------------------------------------------------------------------
Dashboard tích hợp 12 bài tối ưu hoá & học tăng cường (Bài 1 -> Bài 12).
Chạy:  streamlit run app.py
Phụ thuộc tối thiểu: streamlit, numpy, pandas, plotly, scipy  (pulp tuỳ chọn)
==============================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linprog, minimize

# ------------------------------------------------------------------ MIP solver
# Ưu tiên PuLP/CBC; nếu không có thì dùng scipy.optimize.milp (đi kèm scipy).
try:
    import pulp
    HAS_PULP = True
except Exception:
    HAS_PULP = False
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
except Exception:
    HAS_MILP = False

# ============================================================================
# CẤU HÌNH TRANG & GIAO DIỆN
# ============================================================================
st.set_page_config(
    page_title="AIDEOM-VN | Tạ Tuấn Chinh",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bảng màu chủ đạo (cảm hứng cờ Việt Nam: đỏ - vàng, nền sáng học thuật)
ACCENT = "#d4001f"     # đỏ
ACCENT2 = "#f4b400"    # vàng
INK = "#11203a"        # xanh mực
PALETTE = ["#d4001f", "#f4b400", "#1565c0", "#2e7d32", "#6a1b9a",
           "#00838f", "#ef6c00", "#5d4037", "#455a64", "#c2185b"]

st.markdown(f"""
<style>
    .stApp {{ background-color: #f6f8fb; }}
    h1, h2, h3, h4 {{ color: {INK}; font-weight: 700; }}
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #11203a 0%, #1c2f50 100%);
    }}
    section[data-testid="stSidebar"] * {{ color: #e8edf5 !important; }}
    section[data-testid="stSidebar"] .stRadio label {{ font-size: 0.92rem; }}
    /* Thẻ số liệu */
    .metric-card {{
        background:#ffffff; border-radius:14px; padding:16px 18px;
        border:1px solid #e6ebf2; box-shadow:0 2px 8px rgba(17,32,58,.05);
    }}
    .metric-card .label {{ color:#64748b; font-size:.80rem; letter-spacing:.04em;
        text-transform:uppercase; margin-bottom:4px; }}
    .metric-card .value {{ color:{INK}; font-size:1.55rem; font-weight:800; }}
    .metric-card .delta {{ font-size:.80rem; }}
    /* Hộp thông tin sinh viên */
    .student-box {{
        background: rgba(255,255,255,.08); border:1px solid rgba(244,180,0,.45);
        border-radius:12px; padding:12px 14px; margin-top:8px;
    }}
    .student-box .nm {{ font-weight:800; color:#f4b400 !important; font-size:1.02rem; }}
    .student-box .row {{ font-size:.85rem; opacity:.95; margin-top:2px; }}
    .badge {{ display:inline-block; background:{ACCENT}; color:#fff !important;
        border-radius:999px; padding:2px 10px; font-size:.72rem; font-weight:700; }}
    .policy {{ background:#fff7ed; border-left:4px solid {ACCENT2};
        border-radius:8px; padding:12px 16px; margin:8px 0; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:4px; }}
    .stTabs [data-baseweb="tab"] {{ background:#eef2f8; border-radius:8px 8px 0 0; }}
    .stTabs [aria-selected="true"] {{ background:{ACCENT}; color:#fff; }}
</style>
""", unsafe_allow_html=True)


def card(label, value, delta=None, color=ACCENT):
    d = f"<div class='delta' style='color:{color}'>{delta}</div>" if delta else ""
    st.markdown(
        f"<div class='metric-card'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div>{d}</div>", unsafe_allow_html=True)


def section(title, sub=""):
    s = f"<span style='color:#64748b;font-weight:400;font-size:.95rem'> — {sub}</span>" if sub else ""
    st.markdown(f"### {title}{s}", unsafe_allow_html=True)


def policy_box(md):
    st.markdown(f"<div class='policy'>{md}</div>", unsafe_allow_html=True)


# ---- Q&A thảo luận chính sách (trích từ notebooks aideom_vn) ----
POLICY_QA = {1: [('TFP giai đoạn 2020–2025 có ý nghĩa gì với mục tiêu phát triển bền vững của Việt Nam?', 'Xu hướng TFP của Việt Nam giai đoạn 2020-2025 cho thấy sự chuyển dịch quan trọng từ tăng trưởng dựa trên thâm dụng vốn và lao động giá rẻ sang tăng trưởng dựa trên năng suất và đổi mới sáng tạo. Mặc dù chịu ảnh hưởng nặng nề bởi đại dịch COVID-19 trong các năm 2020-2021, TFP vẫn duy trì đà phục hồi mạnh mẽ vào các năm 2022-2025, phản ánh hiệu quả từ các hoạt động số hóa doanh nghiệp. Điều này hoàn toàn phù hợp với định hướng phát triển bền vững của quốc gia, giúp cải thiện chất lượng tăng trưởng thay vì quy mô đơn thuần. Duy trì đà tăng trưởng TFP cao là chìa khóa để Việt Nam tránh bẫy thu nhập trung bình và đạt được các mục tiêu thịnh vượng lâu dài.'), ('Chuyển đổi số (D) và doanh nghiệp công nghệ số (AI) cải thiện TFP thế nào theo QĐ 749/QĐ-TTg?', 'Theo Quyết định 749/QĐ-TTg về Chương trình Chuyển đổi số quốc gia, kinh tế số được xác định là động lực chính để thúc đẩy TFP vượt ngưỡng 35-40% đóng góp vào tăng trưởng GDP. Việc tăng tỷ lệ số hóa (D) và số lượng doanh nghiệp AI giúp tối ưu hóa quy trình sản xuất, cắt giảm chi phí vận hành và tăng cường liên kết chuỗi giá trị. Công nghệ số hoạt động như một chất xúc tác làm tăng hiệu suất của cả vốn vật chất và vốn nhân lực, giúp tạo ra các mô hình kinh doanh mới có giá trị gia tăng cực kỳ cao. Do đó, việc chuyển đổi số sâu rộng chính là con đường ngắn nhất để nâng cao năng lực cạnh tranh quốc gia toàn diện.'), ('Giải pháp nâng cao chất lượng nguồn nhân lực (H) đáp ứng yêu cầu nâng cao TFP?', 'Để nâng cao vốn nhân lực (H) tương xứng với tốc độ phát triển công nghệ, Việt Nam cần thực hiện tái cấu trúc hệ thống giáo dục quốc dân, đẩy mạnh đào tạo các kỹ năng STEM và tư duy số từ bậc phổ thông. Đồng thời, cần xây dựng các chính sách khuyến khích hợp tác công tư giữa các trường đại học hàng đầu và doanh nghiệp công nghệ số để thiết kế các chương trình đào tạo ngắn hạn, thực chiến. Cần đặc biệt ưu tiên ngân sách cho các chương trình đào tạo lại (retraining) cho lực lượng lao động hiện hữu bị ảnh hưởng bởi làn sóng tự động hóa. Cuối cùng, việc thu hút nhân tài số toàn cầu và chuyên gia kiều bào về nước làm việc sẽ giúp đẩy nhanh quá trình nâng cao chất lượng nguồn nhân lực vĩ mô.')], 2: [('Giá đối ngẫu của ràng buộc ngân sách phản ánh điều gì về hiệu quả biên của đầu tư công?', 'Giá đối ngẫu (shadow price) của ràng buộc ngân sách thể hiện mức độ hiệu quả biên của việc bổ sung thêm một đồng vốn ngân sách vào hệ thống kinh tế. Nó biểu thị mức tăng tối đa của GDP khi tổng ngân sách đầu tư tăng thêm một nghìn tỷ đồng, phản ánh tiềm năng sinh lời thực tế của dự án. Nếu giá đối ngẫu này cao, chứng tỏ nguồn lực đầu tư công đang cực kỳ khan hiếm và việc nới lỏng ngân sách sẽ mang lại lợi ích kinh tế vượt trội. Phân tích này giúp các nhà hoạch định chính sách lượng hóa chính xác lợi ích cận biên để đưa ra quyết định mở rộng hay thu hẹp ngân sách một cách khoa học.'), ('Vì sao ràng buộc tỷ lệ công nghệ chiến lược (C5) quan trọng và ảnh hưởng cơ cấu vốn tối ưu thế nào?', 'Ràng buộc tỷ lệ công nghệ chiến lược (C5) đóng vai trò then chốt trong việc định hướng dòng vốn vào các lĩnh vực then chốt có tính lan tỏa công nghệ cao như AI và R&D. Ràng buộc này ngăn chặn xu hướng phân bổ ngắn hạn quá mức vào các hạ tầng vật chất truyền thống vốn có tỷ suất sinh lời cận biên thấp hơn trong kỷ nguyên số. Việc áp dụng ràng buộc C5 ép buộc cơ cấu đầu tư phải dành ít nhất 35% cho các giải pháp công nghệ tương lai, từ đó bảo đảm khả năng cạnh tranh quốc gia lâu dài. Nó cân bằng giữa mục tiêu tăng trưởng nhanh trước mắt và việc tích lũy năng lực khoa học công nghệ bền vững cho thế hệ sau.'), ('Khi ngân sách tăng từ 80 đến 200 nghìn tỷ, thứ tự ưu tiên nhận vốn bổ sung thay đổi thế nào? Tại sao?', 'Khi ngân sách ở mức tối thiểu 80 nghìn tỷ VND, hệ thống chỉ phân bổ vốn vừa đủ để đáp ứng các ràng buộc sàn của từng hạng mục nhằm đảm bảo tính khả thi kinh tế. Khi tổng ngân sách bắt đầu tăng dần lên 200 nghìn tỷ VND, dòng vốn bổ sung được ưu tiên dồn hoàn toàn vào Nghiên cứu và Phát triển (x4) và Công nghệ AI (x2). Điều này xảy ra do hai hạng mục này sở hữu hệ số tác động biên cao nhất (1.35 và 1.20) trong hàm mục tiêu và đồng thời giúp thỏa mãn tối ưu ràng buộc công nghệ chiến lược. Hạ tầng truyền thống và nhân lực thông thường chỉ được duy trì ở mức tối thiểu do có hệ số tác động biên thấp hơn đáng kể.')], 3: [('Vì sao CN chế biến chế tạo và Bán buôn bán lẻ có chỉ số ưu tiên cao trong kịch bản mặc định?', 'Công nghiệp chế biến chế tạo và Bán buôn bán lẻ đạt chỉ số ưu tiên cao vượt trội do hai ngành này sở hữu quy mô lao động khổng lồ, đóng góp GDP trực tiếp lớn và hệ số lan tỏa kinh tế (spillover) đặc biệt rộng. Trong kịch bản mặc định, các tiêu chí về tăng trưởng và mức độ sẵn sàng AI được phân bổ trọng số đồng đều, tạo lợi thế lớn cho các ngành công nghiệp cốt lõi có giá trị xuất khẩu cao. Sự phát triển mạnh mẽ của chế biến chế tạo còn kéo theo hoạt động vận tải và logistics phát triển. Do đó, các ngành này nghiễm nhiên trở thành đầu tàu thu hút đầu tư chiến lược số của quốc gia.'), ('Khác biệt xếp hạng giữa bộ trọng số Tăng trưởng và Bao trùm phản ánh xung đột mục tiêu thế nào?', 'Sự dịch chuyển thứ hạng giữa kịch bản Tăng trưởng và Bao trùm phản ánh rõ nét sự đánh đổi cốt lõi giữa hiệu quả kinh tế ngắn hạn và sự bình đẳng xã hội dài hạn. Trong kịch bản Tăng trưởng, các ngành công nghệ cao, tài chính ngân hàng và xuất khẩu được đặt lên hàng đầu nhằm gia tăng thặng dư kinh tế nhanh nhất có thể. Ngược lại, kịch bản Bao trùm ưu tiên các lĩnh vực có thâm dụng lao động lớn và phúc lợi xã hội cao như Nông-lâm-thủy sản, Giáo dục và Y tế để giảm thiểu rủi ro bị bỏ lại phía sau của người lao động. Sự xung đột này yêu cầu nhà quản lý phải phối hợp nhịp nhàng các công cụ tài khóa để dung hòa cả hai mục tiêu.'), ('Ngành CNTT-TT nên được định vị thế nào trong chiến lược chuyển đổi số quốc gia?', 'Ngành CNTT-TT cần phải được định vị là cơ sở hạ tầng nền tảng và là "trái tim" của toàn bộ chiến lược chuyển đổi số quốc gia chứ không chỉ đơn thuần là một ngành kinh tế độc lập. CNTT-TT đóng vai trò cung cấp giải pháp công nghệ, thuật toán AI và hạ tầng dữ liệu để thúc đẩy quá trình thông minh hóa cho cả 9 ngành còn lại. Đây là ngành có hệ số lan tỏa công nghệ cao nhất (0.92) và năng lực xuất khẩu dịch vụ rất lớn của Việt Nam. Do đó, đầu tư ưu tiên tuyệt đối vào CNTT-TT sẽ tạo ra hiệu ứng số nhân, nâng cánh cho toàn bộ nền kinh tế cất cánh.')], 4: [('Cái giá của sự công bằng (Cost of Fairness) biểu hiện thế nào qua chênh lệch Z giữa hai mô hình?', 'Cái giá của sự công bằng (Cost of Fairness) được đo lường trực tiếp bằng sự suy giảm của tổng tăng trưởng GDP vùng miền (Z) từ mô hình tối đa hóa hiệu quả thuần túy sang mô hình có ràng buộc bình đẳng. Cụ thể, khi áp đặt ràng buộc công bằng (lambda = 0.7), nguồn vốn buộc phải dịch chuyển khỏi các cực tăng trưởng siêu hiệu quả như Đông Nam Bộ để phân bổ cho vùng miền có chỉ số số hóa ban đầu thấp như Tây Nguyên hay Miền núi phía Bắc. Sự dịch chuyển phi tối ưu hóa hiệu quả này làm giảm tổng sản lượng quốc gia nhưng lại cải thiện chỉ số công bằng xã hội. Đây là khoản đánh đổi chi phí cơ hội tất yếu mà bất cứ quốc gia nào cũng phải đối mặt để đảm bảo ổn định chính trị và phát triển bao trùm.'), ('Ràng buộc công bằng số hóa (C5) tác động thế nào đến dòng vốn cho Đông Nam Bộ và Tây Nguyên?', 'Dưới tác động của ràng buộc công bằng số hóa (C5), dòng vốn đầu tư cho số hóa đã có sự dịch chuyển ngoạn mục giữa các vùng miền. Ở mô hình hiệu quả thuần túy, Đông Nam Bộ (SE) được ưu ái phân bổ nguồn vốn lớn nhờ có hiệu suất biên vượt trội trong khi Tây Nguyên (CH) chỉ nhận được mức vốn tối thiểu. Tuy nhiên, khi áp dụng ràng buộc công bằng, nguồn vốn số hóa tại Tây Nguyên tăng mạnh để nâng chỉ số số hóa cơ sở lên bằng ít nhất 70% mức tối đa quốc gia. Điều này giúp ngăn chặn khoảng cách số ngày càng giãn rộng giữa hai vùng miền có điều kiện tự nhiên hoàn toàn khác biệt.'), ('Phương án kết hợp chính sách để dung hòa giữa tăng trưởng hiệu quả và bình đẳng vùng miền?', 'Để dung hòa giữa tăng trưởng hiệu quả và bình đẳng vùng miền, Chính phủ nên áp dụng cơ chế chính sách hai tầng linh hoạt. Tầng một, cho phép các cực tăng trưởng lớn tự chủ thu hút FDI và phát huy tối đa lợi thế tăng trưởng công nghệ AI hiệu quả cao để tạo nguồn thu ngân sách dồi dào. Tầng hai, sử dụng cơ chế điều chuyển thuế trung ương để tái đầu tư có mục tiêu vào các hạ tầng số cơ bản cho các vùng nghèo như Tây Nguyên và Miền núi phía Bắc thông qua các quỹ dịch vụ viễn thông công ích. Phương án này vừa giữ vững động lực tăng trưởng mũi nhọn của đất nước vừa đảm bảo mọi người dân đều có quyền tiếp cận dịch vụ số tối thiểu.')], 5: [('Ràng buộc tiên quyết về đào tạo nhân lực (P12) phản ánh tư duy chiến lược nào?', 'Ràng buộc tiên quyết đòi hỏi dự án đào tạo nhân lực số (P12) phải được thực hiện trước khi triển khai dự án AI (P8) và Bán dẫn (P13) thể hiện tư duy chiến lược cực kỳ đúng đắn về sự phát triển đồng bộ và bền vững. Đầu tư vào các siêu công nghệ phần cứng và giải pháp thuật toán tiên tiến sẽ trở nên hoàn toàn lãng phí và vô hiệu nếu không có đội ngũ kỹ sư vận hành có trình độ chuyên môn cao tương ứng. Con người luôn là nhân tố cốt lõi quyết định sự thành bại của bất kỳ tiến trình chuyển đổi số nào. Ràng buộc này đảm bảo tính hiệu quả trong sử dụng vốn đầu tư công, tránh tình trạng hạ tầng đắp chiếu chờ nhân sự.'), ('Vì sao khi tích hợp rủi ro thất bại, danh mục dự án được chọn lại thay đổi?', 'Khi tích hợp yếu tố rủi ro thất bại, mô hình chuyển đổi mục tiêu từ tối đa hóa giá trị danh nghĩa sang tối đa hóa lợi ích kỳ vọng thực tế (đã nhân với xác suất thành công p_i). Điều này dẫn đến sự dịch chuyển dòng vốn khỏi các dự án có NPV danh nghĩa cao nhưng độ rủi ro lớn (ví dụ các công nghệ đột phá nhưng phức tạp như AI hay bán dẫn có p_i thấp) sang các dự án bền vững có xác suất thành công cao hơn. Kết quả cụ thể cho thấy các dự án dịch vụ công trực tuyến và hạ tầng lõi có tính an toàn cao được ưu tiên chọn trước để bảo đảm sự chắc chắn cho dòng tiền ngân sách. Nó định hình lại phong cách hoạch định dự án theo hướng phòng ngừa rủi ro.'), ('Tầm quan trọng của việc bắt buộc chọn dự án An ninh mạng (P14) theo chiến lược ATTT quốc gia?', 'Việc bắt buộc lựa chọn dự án An ninh mạng (P14) hoàn toàn nhất quán với định hướng coi an toàn, an ninh mạng là chiếc khiên bảo vệ và là điều kiện tiên quyết để xây dựng xã hội số thành công. Khi toàn bộ hệ thống dịch vụ công và cơ sở dữ liệu quốc gia được đưa lên đám mây, các nguy cơ tấn công mạng và rò rỉ dữ liệu sẽ đe dọa trực tiếp đến an ninh quốc phòng. Một hệ thống số hóa mạnh mẽ nhưng thiếu bảo mật sẽ trở thành một miếng mồi ngon và dễ dàng bị sụp đổ bất cứ lúc nào. Vì vậy, an ninh mạng phải luôn đi trước một bước và đồng hành trong mọi dự án chuyển đổi số công.')], 6: [('Sự vượt trội của Đông Nam Bộ và Đồng bằng sông Hồng phản ánh thực trạng phân bổ nguồn lực thế nào?', 'Sự thống trị tuyệt đối của Đông Nam Bộ và Đồng bằng sông Hồng trên bảng xếp hạng TOPSIS phản ánh thực trạng chênh lệch địa lý sâu sắc trong phân bổ nguồn lực kinh tế số ở Việt Nam. Hai vùng này là nơi tập trung hầu hết các khu công nghiệp công nghệ cao, doanh nghiệp phần mềm hàng đầu và thu hút tới hơn 70% tổng dòng vốn FDI cả nước. Hệ thống hạ tầng viễn thông phát triển đồng bộ và mật độ dân số trẻ cao tạo ra thị trường tiêu dùng số dồi dào, thúc đẩy quá trình sẵn sàng AI diễn ra nhanh chóng. Thực tế này đòi hỏi Nhà nước phải có các chính sách điều tiết vĩ mô mạnh mẽ để tránh tình trạng phân cực phát triển kinh tế số quá sâu sắc.'), ('Trọng số Entropy đem lại lợi ích gì so với trọng số chuyên gia chủ quan?', 'Phương pháp trọng số Entropy đem lại tính khách quan khoa học cực kỳ cao nhờ việc xác định trọng số hoàn toàn dựa trên sự phân tán dữ liệu thực tế của từng tiêu chí chứ không phụ thuộc vào cảm tính của con người. Nếu một tiêu chí có sự khác biệt rất lớn giữa các vùng (như dòng vốn FDI hay chỉ số AI), Entropy sẽ tự động gán cho nó trọng số cao vì nó chứa nhiều thông tin để phân loại. Ngược lại, những tiêu chí có sự tương đồng lớn giữa các vùng sẽ nhận trọng số thấp hơn. Điều này bổ khuyết xuất sắc cho phương pháp định lượng chuyên gia vốn dễ bị ảnh hưởng bởi định kiến cá nhân.'), ('Phân tích độ nhạy trọng số AI Readiness gợi ý gì cho việc thiết lập ưu tiên phát triển vùng?', 'Kết quả phân tích độ nhạy cho thấy khi trọng số AI tăng lên, điểm số và thứ hạng của các vùng có nền tảng số hóa mạnh mẽ như Đông Nam Bộ và Đồng bằng sông Hồng ngày càng củng cố vị trí dẫn đầu. Trong khi đó, các vùng gặp khó khăn về hạ tầng công nghệ như Tây Nguyên hay Miền núi phía Bắc sẽ bị tụt hậu xa hơn nữa do điểm số giảm sút nghiêm trọng. Điều này gợi ý rằng đối với các vùng đi sau, các nhà hoạch định chính sách trước tiên cần ưu tiên đầu tư vào hạ tầng kết nối cơ bản và nâng cao tỷ lệ phổ cập internet trước khi trực tiếp áp đặt các chỉ tiêu công nghệ AI cao siêu. Phát triển theo lộ trình từng bước sẽ giúp tối ưu hóa hiệu quả sử dụng nguồn ngân sách hạn hẹp.')], 7: [('Khái niệm tập tối ưu Pareto và ý nghĩa của biên Pareto trong hỗ trợ ra quyết định kinh tế?', 'Tập tối ưu Pareto đại diện cho tập hợp tất cả các phương án phân bổ ngân sách mà ở đó chúng ta không thể cải thiện bất kỳ một mục tiêu nào (ví dụ tăng trưởng GDP) nếu không làm suy giảm đi ít nhất một mục tiêu khác (như công bằng xã hội). Biên Pareto vẽ ra ranh giới giới hạn năng lực kinh tế tối đa của hệ thống, giúp loại bỏ hoàn toàn các quyết định phân bổ kém hiệu quả nằm sâu phía trong. Dựa vào biên Pareto, các nhà hoạch định chính sách có được một cái nhìn trực quan toàn diện để cân nhắc sự đánh đổi một cách tường minh. Thay vì tìm kiếm một giải pháp hoàn hảo duy nhất không tồn tại, họ có thể chọn lựa phương án thỏa hiệp tối ưu nhất dựa trên định hướng chính trị của từng thời kỳ.'), ('Điểm thỏa hiệp Nash (Nash Bargaining Solution) giúp giải quyết xung đột lợi ích thế nào?', 'Điểm thỏa hiệp Nash đóng vai trò là một trọng tài toán học khách quan giúp hài hòa hóa quyền lợi giữa các nhóm lợi ích có mục tiêu xung đột nhau trong nền kinh tế. Bằng cách cực đại hóa tích số vượt trội của lợi ích so với điểm tham chiếu tối thiểu (disagreement point), điểm Nash bảo đảm rằng không một nhóm nào bị chèn ép quá mức và mỗi bên đều nhận được một phần chia sẻ lợi ích công bằng nhất. Giải pháp này hạn chế tối đa các tranh chấp và xung đột quyền lực trong quá trình phân bổ ngân sách công. Điều này tạo điều kiện thuận lợi để đạt được sự đồng thuận cao của các bộ ngành trong thực tiễn điều hành vĩ mô.'), ('Khi ưu tiên an sinh xã hội (F2 tăng), phân bổ cho đào tạo nhân lực (x3) và phát triển AI (x2) đổi ra sao?', 'Khi trọng tâm chính sách dịch chuyển mạnh mẽ sang mục tiêu bình đẳng và an sinh xã hội (F2 tăng), dòng vốn đầu tư ghi nhận sự tái cơ cấu sâu sắc. Hạng mục đào tạo nhân lực (x3) nhận được lượng vốn tăng vọt do đây là công cụ trực tiếp giúp nâng cao năng lực tự thân và thu nhập của người lao động. Ngược lại, nguồn ngân sách phân bổ cho phát triển AI (x2) bị kiềm chế hoặc giảm nhẹ để hạn chế tối đa các tác động tiêu cực của làn sóng tự động hóa gây sa thải lao động quy mô lớn. Sự điều chỉnh này tạo ra bộ đệm an toàn giúp nền kinh tế chuyển đổi số một cách êm ái hơn.')], 8: [('Vì sao các năm đầu dòng vốn ưu tiên tích lũy công nghệ số (u_D) hơn vốn truyền thống (u_K)?', 'Trong những năm đầu tiên của chu kỳ hoạch định, dòng vốn đầu tư tối ưu tập trung đột biến vào số hóa (u_D) do chỉ số số hóa ban đầu của nền kinh tế đang ở mức cực kỳ thấp so với quy mô vốn vật chất lũy kế khổng lồ. Theo quy luật lợi ích cận biên giảm dần, đầu tư thêm vào một lĩnh vực thiếu hụt nghiêm trọng như công nghệ số sẽ mang lại hiệu suất sinh lời GDP cận biên lớn hơn nhiều so với việc tiếp tục thâm dụng vốn vật lý truyền thống. Sự chênh lệch tỷ lệ khấu hao (công nghệ số lỗi thời nhanh hơn với 8% so với 5% của vốn vật chất) cũng đòi hỏi phải bổ sung vốn công nghệ liên tục ở giai đoạn đầu để tạo đà bứt phá. Khi hệ thống đạt đến trạng thái cân bằng động, tỷ lệ phân bổ sẽ tự động được điều chỉnh hài hòa hơn.'), ('Ý nghĩa kinh tế của hệ số chiết khấu r đối với lựa chọn lợi ích trước mắt và bền vững dài hạn?', 'Hệ số chiết khấu r thể hiện mức độ ưu tiên của xã hội đối với tiêu dùng ở thời điểm hiện tại so với lợi ích tích lũy trong tương lai. Nếu r rất cao, nền kinh tế sẽ có xu hướng tối đa hóa các lợi ích ngắn hạn trước mắt, dẫn đến việc cắt giảm đầu tư cho R&D và hạ tầng số dài hạn vốn cần nhiều thời gian để đơm hoa kết trái. Ngược lại, một hệ số chiết khấu r thấp hoặc vừa phải (5%) khuyến khích các nhà hoạch định chấp nhận hy sinh một phần tiêu dùng hiện tại để xây dựng năng lực công nghệ nền tảng bền vững cho các thế hệ tương lai. Điều này tạo điều kiện tối ưu để tích lũy các tài sản số quốc gia có tuổi thọ kinh tế dài hạn.'), ('Ảnh hưởng của khác biệt tỷ lệ khấu hao giữa vốn vật lý và vốn số hóa đối với quỹ đạo tích lũy?', 'Sự lỗi thời nhanh chóng của công nghệ số (khấu hao 8%) so với hạ tầng truyền thống (khấu hao 5%) đặt ra những thách thức cực kỳ lớn đối với quỹ đạo đầu tư tích lũy dài hạn. Nó yêu cầu dòng vốn đầu tư cho công nghệ số phải được bổ sung liên tục với cường độ cao chỉ để bù đắp phần hao hụt tự nhiên do tốc độ thay đổi nhanh chóng của các hệ điều hành và phần cứng AI toàn cầu. Nếu ngừng đầu tư dù chỉ một thời gian ngắn, năng lực công nghệ số của quốc gia sẽ suy giảm nghiêm trọng và nhanh chóng trở nên lạc hậu so với thế giới. Do đó, ngân sách công phải có tính ổn định lâu dài và tránh các cú sốc cắt giảm đột ngột trong đầu tư công nghệ số.')], 9: [('Đánh giá nguy cơ thất nghiệp công nghệ trong ngành Công nghiệp chế biến chế tạo tại Việt Nam?', 'Ngành Công nghiệp chế biến chế tạo tại Việt Nam đang đứng trước rủi ro cực kỳ lớn về thất nghiệp công nghệ do phần lớn lao động hiện nay là lao động phổ thông, thực hiện các công việc có tính lặp đi lặp lại cao và dễ dàng bị robot hóa thay thế. Khi chi phí đầu tư cho các giải pháp tự động hóa và cánh tay robot công nghiệp ngày càng rẻ đi, các doanh nghiệp FDI sẽ nhanh chóng chuyển đổi để tối ưu hóa năng suất và chất lượng sản phẩm. Nếu không có các giải pháp can thiệp kịp thời từ Chính phủ, làn sóng sa thải quy mô lớn sẽ diễn ra, đe dọa trực tiếp đến sinh kế của hàng triệu công nhân và gây sức ép khổng lồ lên hệ thống an sinh xã hội. Đây là bài toán cấp bách đòi hỏi các chính sách chuyển đổi nghề nghiệp chủ động trước khi rủi ro chuyển hóa thành khủng hoảng.'), ('Tầm quan trọng của "ngưỡng đào tạo lại tối thiểu" với lập kế hoạch ngân sách của Bộ LĐ-TB&XH?', 'Việc xác định chính xác "ngưỡng đào tạo lại tối thiểu" là công cụ quản trị định lượng vô cùng quan trọng giúp Bộ LĐ-TB&XH thoát khỏi phương pháp phân bổ ngân sách theo cảm tính hay định mức truyền thống. Chỉ số này chỉ ra ranh giới tài khóa an toàn tuyệt đối mà Nhà nước bắt buộc phải đầu tư cho đào tạo lại để tránh kịch bản tạo việc làm ròng bị âm trong từng ngành kinh tế cụ thể. Nó giúp tối ưu hóa việc phân bổ nguồn lực công, hướng dòng vốn trực tiếp vào các chương trình nâng cao kỹ năng thực chất của những ngành có nguy cơ tổn thương cao nhất. Nhờ đó, ngân sách an sinh xã hội được sử dụng với hiệu quả kinh tế - xã hội cao nhất.'), ('QĐ 127/QĐ-TTg về Chiến lược AI quốc gia đặt ra yêu cầu gì cho lực lượng lao động hiện tại?', 'Quyết định 127/QĐ-TTg đặt ra yêu cầu vô cùng cấp thiết đối với lực lượng lao động hiện tại là phải nhanh chóng tái trang bị kỹ năng và nâng cao năng lực thích ứng trong môi trường làm việc có sự tương tác cao với AI. Người lao động không còn có thể dựa vào các kỹ năng thủ công đơn thuần mà bắt buộc phải sở hữu tư duy số cơ bản, khả năng cộng tác và khai thác hiệu quả các công cụ trí tuệ nhân tạo để gia tăng năng suất cá nhân. Chiến lược này đòi hỏi việc chuyển dịch cơ cấu nhân lực mạnh mẽ từ các thao tác cơ học sang các khâu sáng tạo, thiết kế và quản trị hệ thống. Đây là một cuộc cách mạng toàn diện về tư duy học tập suốt đời của người lao động Việt Nam.')], 10: [('Ý nghĩa của VSS (Value of Stochastic Solution) — vì sao cân nhắc bất định đem lại lợi ích kinh tế?', 'Giá trị của giải pháp ngẫu nhiên (VSS) đo lường trực tiếp mức độ thiệt hại tài chính nếu nhà hoạch định nhắm mắt bỏ qua các rủi ro không chắc chắn và ngây thơ sử dụng mô hình tất định giá trị trung bình (EV). Việc cân nhắc sự không chắc chắn đem lại lợi ích kinh tế to lớn bởi nó giúp xây dựng các quyết định đầu tư ban đầu có tính linh hoạt cao, tạo ra các dư địa an toàn để điều chỉnh quy mô trong tương lai. Nó ngăn chặn các tình trạng đầu tư quá mức dẫn đến lãng phí nguồn lực khi thị trường xấu hoặc đầu tư dưới mức làm bỏ lỡ các cơ hội bứt phá khi thị trường thuận lợi. Nói cách khác, VSS chính là phí bảo hiểm thông minh mà xã hội nhận được khi thực hiện quản trị rủi ro một cách khoa học.'), ('Ý nghĩa kinh tế của EVPI đối với đầu tư cho hoạt động dự báo và nghiên cứu thị trường?', 'Chỉ số EVPI (Expected Value of Perfect Information) thiết lập một cái trần tài chính tuyệt đối về số ngân sách tối đa mà Chính phủ nên chi trả cho các hoạt động nghiên cứu thị trường, thuê chuyên gia tư vấn hoặc nâng cấp năng lực dự báo vĩ mô. Nếu chi phí thu thập thông tin và dự báo vượt quá giá trị EVPI, việc mua thêm thông tin sẽ trở nên bất hợp lý về mặt kinh tế vì lợi ích cận biên thu về không bù đắp được chi phí bỏ ra. EVPI giúp nhà quản lý lượng hóa giá trị thực tế của tri thức và sự chắc chắn trong nền kinh tế. Đây là công cụ đắc lực để tối ưu hóa hiệu quả hoạt động của các cơ quan thống kê quốc gia.'), ('Khi nào nên chọn Minimax Regret thay vì tối đa hóa lợi nhuận kỳ vọng RP?', 'Nhà hoạch định chính sách nên chuyển từ mô hình tối đa hóa kỳ vọng (RP) sang mô hình Minimax Regret vững chắc trong các bối cảnh bất định cực đoan (Knightian uncertainty), nơi chúng ta không thể xác định được phân phối xác suất khách quan của các kịch bản tương lai. Minimax Regret đặc biệt phù hợp cho các quyết định an ninh quốc gia, hạ tầng huyết mạch hoặc chính sách y tế khẩn cấp, nơi các sai lầm có thể dẫn đến thảm họa hoặc tổn thất không thể đảo ngược. Mô hình này bảo đảm an toàn tối đa cho hệ thống bằng cách hạn chế tối thiểu sự hối tiếc lớn nhất trong tình huống xấu nhất xảy ra. Nó phản ánh triết lý quản trị phòng ngừa rủi ro chủ động ở cấp độ cao nhất.')], 11: [('Khác biệt bản chất giữa tối ưu hóa tĩnh (LP/MIP) và học máy tăng cường (RL) trong hoạch định dài hạn?', 'Sự khác biệt bản chất nằm ở khả năng đối phó với tính động và tính bất định dài hạn của hệ thống kinh tế. Các mô hình tối ưu hóa tĩnh (LP/MIP) giải quyết bài toán phân bổ nguồn lực tại một thời điểm hoặc trong các chu kỳ độc lập với giả định mọi thông số đều cố định và biết trước. Ngược lại, Học máy tăng cường (RL) huấn luyện Agent tương tác trực tiếp với một môi trường mô phỏng đầy biến động, tự học hỏi thông qua thử và sai để tối đa hóa phúc lợi xã hội dài hạn. RL có khả năng tự động điều chỉnh hành vi khi trạng thái hệ thống thay đổi (ví dụ khi tỷ lệ thất nghiệp tăng cao đột ngột, Agent sẽ tự chuyển sang hành động an sinh xã hội). Điều này giúp xây dựng các chiến lược điều hành linh hoạt và có khả năng chống chịu cao với các cú sốc vĩ mô.'), ('Q-learning Agent học cách dung hòa giữa phát triển AI làm tăng GDP và rủi ro thất nghiệp công nghệ thế nào?', 'Q-learning Agent học cách dung hòa hai mục tiêu xung đột này thông qua cấu trúc tinh tế của hàm thưởng phúc lợi xã hội (Reward Function). Ban đầu, do bị hấp dẫn bởi điểm cộng rất lớn của tăng trưởng GDP nhanh khi đầu tư mạnh vào AI (Action 1), Agent sẽ liên tục chọn hành động này. Tuy nhiên, hành động này nhanh chóng làm đẩy rủi ro thất nghiệp lên mức tối đa (Risk=2), khiến hệ thống phải chịu hình phạt cực kỳ nặng nề (điểm trừ -30.0) và lâm vào khủng hoảng kép. Qua hàng ngàn tập huấn luyện, Agent nhận thức được quy luật này và tự động phát triển chiến lược xen kẽ thông minh: đầu tư mạnh mẽ vào AI đi kèm ngay với các chương trình đào tạo lại nhân lực số và an sinh xã hội để kéo giảm rủi ro thất nghiệp về mức an toàn. Sự tự điều tiết này chính là biểu hiện rõ nét của phát triển bền vững.'), ('Quỹ đạo hành động chính sách của Agent gợi ý lộ trình thế nào cho Việt Nam hướng tới 2045?', 'Quỹ đạo hành động của Agent gợi ý một lộ trình phát triển ba giai đoạn cực kỳ logic cho Việt Nam hướng tới tầm nhìn 2045 thịnh vượng. Giai đoạn một (Năm 1-5), ưu tiên hàng đầu là đầu tư hạ tầng số cơ bản và đào tạo nâng cao chất lượng nhân lực số nền tảng để tạo bộ đệm hấp thụ công nghệ. Giai đoạn hai (Năm 6-15), khi nguồn lực nhân sự đã sẵn sàng, tập trung đẩy mạnh ứng dụng AI đột phá trong các ngành công nghiệp mũi nhọn để tối đa hóa GDP và nâng cao năng lực cạnh tranh quốc gia. Giai đoạn ba (Năm 16-20), chủ động chuyển trọng tâm sang an sinh xã hội, đào tạo lại thường xuyên và duy trì phát triển hài hòa để bảo đảm thặng dư kinh tế được chia sẻ công bằng toàn xã hội, hướng tới sự phát triển phồn vinh lâu dài.')]}

def policy_discussion(n):
    """Render mục Thảo luận chính sách đầy đủ cho Bài n (3 câu hỏi + trả lời)."""
    items = POLICY_QA.get(n, [])
    if not items:
        return
    section("Thảo luận chính sách")
    labels = ["a", "b", "c", "d", "e"]
    for i, (q, a) in enumerate(items):
        with st.expander(f"Câu {labels[i]}. {q}", expanded=(i == 0)):
            st.markdown(a)


# ============================================================================
# DỮ LIỆU GỐC (nhúng trực tiếp — đảm bảo chạy độc lập trên Streamlit Cloud)
# Tự động đọc đè từ thư mục data/ nếu tồn tại.
# ============================================================================
@st.cache_data
def load_macro():
    df = pd.DataFrame({
        "year": [2020, 2021, 2022, 2023, 2024, 2025],
        "GDP_trillion_VND": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
        "K_trillion_VND": [16500, 17800, 19600, 21300, 23500, 25900],
        "L_million": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
        "D_digital_pct": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
        "AI_tech_firms_thousand": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
        "H_trained_pct": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
    })
    return df


@st.cache_data
def load_sectors():
    df = pd.DataFrame({
        "sector_id": list(range(1, 11)),
        "sector_name_vi": ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng",
                            "Khai khoáng", "Bán buôn-bán lẻ", "Tài chính-Ngân hàng",
                            "Logistics-Vận tải", "CNTT-Truyền thông", "Giáo dục-Đào tạo", "Y tế"],
        "growth_rate_2024_pct": [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
        "productivity_million_VND_per_worker": [103.4, 241.2, 168.8, 1290.5, 145.3,
                                                1072.4, 321.4, 713.8, 205.7, 437.1],
        "spillover_coef_0_1": [0.35, 0.78, 0.42, 0.30, 0.55, 0.85, 0.72, 0.92, 0.65, 0.60],
        "export_billion_USD": [40.5, 290.9, 2.5, 8.2, 5.5, 1.2, 3.1, 178.0, 0.0, 0.0],
        "labor_million": [13.20, 11.50, 4.80, 0.30, 7.80, 0.55, 1.95, 0.62, 2.15, 0.75],
        "ai_readiness_0_100": [15, 55, 20, 30, 48, 72, 42, 88, 38, 45],
        "automation_risk_pct": [18, 42, 25, 55, 38, 52, 35, 28, 22, 18],
    })
    return df


@st.cache_data
def load_regions():
    df = pd.DataFrame({
        "region_id": list(range(1, 7)),
        "region_name_vi": ["Trung du miền núi phía Bắc", "Đồng bằng sông Hồng",
                            "Bắc Trung Bộ và Duyên hải miền Trung", "Tây Nguyên",
                            "Đông Nam Bộ", "Đồng bằng sông Cửu Long"],
        "grdp_per_capita_million_VND": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
        "fdi_registered_billion_USD": [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
        "digital_index_0_100": [38, 78, 55, 32, 82, 48],
        "ai_readiness_0_100": [22, 68, 40, 18, 75, 30],
        "trained_labor_pct": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
        "rd_intensity_pct": [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
        "internet_penetration_pct": [72, 92, 84, 68, 94, 78],
        "gini_coef": [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
    })
    return df


MACRO = load_macro()
SECTORS = load_sectors()
REGIONS = load_regions()
REGION_SHORT = ["TDMNPB", "ĐBSH", "BTB&DHMT", "TN", "ĐNB", "ĐBSCL"]

# ============================================================================
# SIDEBAR — MỤC LỤC + THÔNG TIN SINH VIÊN
# ============================================================================
st.sidebar.markdown(
    "<div style='font-size:1.35rem;font-weight:900;color:#f4b400 !important'>🇻🇳 AIDEOM-VN</div>"
    "<div style='font-size:.78rem;opacity:.85;margin-bottom:6px'>Vietnam Economic Decision Support System</div>",
    unsafe_allow_html=True)

PAGES = [
    "🏠 Trang chủ",
    "🌱 Bài 1 · Cobb-Douglas mở rộng",
    "💰 Bài 2 · LP ngân sách số",
    "📊 Bài 3 · Chỉ số ưu tiên 10 ngành",
    "🗺️ Bài 4 · LP phân bổ ngành-vùng",
    "🎯 Bài 5 · MIP chọn 15 dự án",
    "🏆 Bài 6 · TOPSIS xếp hạng 6 vùng",
    "🌐 Bài 7 · NSGA-II đa mục tiêu",
    "⏳ Bài 8 · Tối ưu động 2026-2035",
    "👷 Bài 9 · Lao động & AI",
    "🎲 Bài 10 · Quy hoạch ngẫu nhiên 2 GĐ",
    "🤖 Bài 11 · Q-learning RL",
    "🧠 Bài 12 · AIDEOM-VN tích hợp",
]
choice = st.sidebar.radio("MỤC LỤC", PAGES, label_visibility="visible")

st.sidebar.markdown(
    "<div class='student-box'>"
    "<span class='badge'>BÀI TẬP LỚN</span>"
    "<div class='row'>Các mô hình ra quyết định</div>"
    "<div class='nm'>Tạ Tuấn Chinh</div>"
    "<div class='row'>Mã sinh viên: <b>23051191</b></div>"
    "</div>", unsafe_allow_html=True)
st.sidebar.caption("Dữ liệu Việt Nam 2020-2025 · GSO · WB · MoST · MIC · MPI · GII")


# ============================================================================
# TRANG CHỦ — tổng hợp dữ liệu gốc & nội dung bài tập
# ============================================================================
def render_home():
    st.markdown(
        f"<h1 style='margin-bottom:0'>AIDEOM-VN <span style='color:{ACCENT}'>·</span> "
        "Hệ thống hỗ trợ ra quyết định kinh tế số</h1>"
        "<p style='color:#475569;font-size:1.05rem;margin-top:4px'>"
        "Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI — "
        "tích hợp 12 bài tối ưu hoá, MCDM, đa mục tiêu, ngẫu nhiên và học tăng cường.</p>",
        unsafe_allow_html=True)

    g25 = MACRO.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("GDP 2025 (ngh.tỷ VND)", "12.847,6", "▲ 8,02% so với 2024", "#2e7d32")
    with c2: card("GDP/người 2025 (USD)", "5.026", "▲ từ 4.700 (2024)", "#2e7d32")
    with c3: card("Kinh tế số / GDP", "≈19,5%", "Mục tiêu 2030: 30%", ACCENT)
    with c4: card("Đóng góp KH-CN/GDP", "2,49%", "1,68% trực tiếp + 0,81% lan toả", INK)

    st.divider()
    section("📚 Bản đồ 12 bài tập theo 4 cấp độ")
    tiers = pd.DataFrame({
        "Cấp độ": ["DỄ", "DỄ", "DỄ", "TRUNG BÌNH", "TRUNG BÌNH", "TRUNG BÌNH",
                   "KHÁ KHÓ", "KHÁ KHÓ", "KHÁ KHÓ", "KHÓ", "KHÓ", "KHÓ"],
        "Bài": [f"Bài {i}" for i in range(1, 13)],
        "Nội dung": [
            "Hàm sản xuất Cobb-Douglas mở rộng + AI/số hoá, phân rã tăng trưởng",
            "LP phân bổ ngân sách số 4 hạng mục + giá đối ngẫu",
            "Chỉ số ưu tiên Priorityᵢ cho 10 ngành (chuẩn hoá min-max)",
            "LP phân bổ ngân sách ngành-vùng + ràng buộc công bằng vùng",
            "MIP 0-1 lựa chọn 15 dự án chuyển đổi số (knapsack tổng quát)",
            "TOPSIS + Entropy xếp hạng 6 vùng theo sẵn sàng AI",
            "NSGA-II tối ưu Pareto 4 mục tiêu (tăng trưởng/bao trùm/môi trường/an ninh)",
            "Tối ưu động liên thời gian 2026-2035 (Cobb-Douglas + tích luỹ vốn)",
            "Mô phỏng tác động AI tới thị trường lao động (NetJob ròng)",
            "Quy hoạch ngẫu nhiên 2 giai đoạn (VSS & EVPI)",
            "Q-learning / DQN cho chính sách kinh tế thích nghi (MDP)",
            "Tích hợp 6 module + dashboard 5 kịch bản chính sách",
        ],
        "Công cụ": ["numpy", "scipy/pulp", "numpy", "scipy/pulp", "pulp/milp",
                    "numpy", "evolutionary", "scipy", "scipy/pulp",
                    "scipy (SP)", "Q-learning", "tích hợp"],
    })
    st.dataframe(tiers, use_container_width=True, hide_index=True, height=460)

    st.divider()
    section("🗃️ Dữ liệu gốc Việt Nam 2020-2025", "3 bộ dữ liệu thực tế")
    t1, t2, t3 = st.tabs(["📈 Vĩ mô 2020-2025", "🏭 10 ngành 2024", "🗺️ 6 vùng 2024"])
    with t1:
        st.dataframe(MACRO, use_container_width=True, hide_index=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=MACRO.year, y=MACRO.GDP_trillion_VND, name="GDP (ngh.tỷ VND)",
                    marker_color=ACCENT, opacity=.85)
        fig.add_trace(go.Scatter(x=MACRO.year, y=MACRO.D_digital_pct, name="Kinh tế số/GDP (%)",
                      mode="lines+markers", line=dict(color=ACCENT2, width=3)), secondary_y=True)
        fig.update_layout(height=360, template="plotly_white",
                          title="GDP và tỷ trọng kinh tế số")
        fig.update_yaxes(title_text="GDP (ngh.tỷ VND)", secondary_y=False)
        fig.update_yaxes(title_text="Kinh tế số/GDP (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        st.dataframe(SECTORS, use_container_width=True, hide_index=True)
        fig = px.scatter(SECTORS, x="ai_readiness_0_100", y="growth_rate_2024_pct",
                         size="labor_million", color="spillover_coef_0_1",
                         hover_name="sector_name_vi", color_continuous_scale="RdYlGn",
                         labels={"ai_readiness_0_100": "AI Readiness",
                                 "growth_rate_2024_pct": "Tăng trưởng 2024 (%)"})
        fig.update_layout(height=380, template="plotly_white",
                          title="10 ngành: AI readiness × Tăng trưởng (kích thước = lao động)")
        st.plotly_chart(fig, use_container_width=True)
    with t3:
        st.dataframe(REGIONS, use_container_width=True, hide_index=True)
        fig = px.bar(REGIONS, x="region_name_vi", y="ai_readiness_0_100",
                     color="digital_index_0_100", color_continuous_scale="Reds",
                     labels={"region_name_vi": "", "ai_readiness_0_100": "AI Readiness"})
        fig.update_layout(height=360, template="plotly_white",
                          title="Mức sẵn sàng AI của 6 vùng kinh tế-xã hội")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    section("🏛️ Khung chính sách tham chiếu")
    policy_box(
        "<b>Nghị quyết 57-NQ/TW</b> (2024) — đột phá phát triển KHCN, ĐMST & chuyển đổi số · "
        "<b>QĐ 749/QĐ-TTg</b> — Chương trình Chuyển đổi số quốc gia · "
        "<b>QĐ 127/QĐ-TTg</b> — Chiến lược quốc gia về AI đến 2030 · "
        "<b>QĐ 411/QĐ-TTg</b> — Chiến lược kinh tế số & xã hội số · "
        "<b>Cam kết COP26</b> — phát thải ròng bằng 0 vào 2050.")
    st.caption("👉 Chọn từng bài ở **MỤC LỤC** bên trái. Mỗi bài gồm: Bối cảnh · Mô hình "
               "toán học · Kết quả (tương tác) · Thảo luận chính sách.")


# ============================================================================
# BÀI 1 — HÀM SẢN XUẤT COBB-DOUGLAS MỞ RỘNG
# ============================================================================
@st.cache_data
def cobb_compute(a, b, g, d, th):
    K = MACRO.K_trillion_VND.values.astype(float)
    L = MACRO.L_million.values.astype(float)
    D = MACRO.D_digital_pct.values.astype(float)
    AI = MACRO.AI_tech_firms_thousand.values.astype(float)
    H = MACRO.H_trained_pct.values.astype(float)
    Y = MACRO.GDP_trillion_VND.values.astype(float)
    core = K**a * L**b * D**g * AI**d * H**th
    A = Y / core
    Ybar = A.mean()
    Yhat = Ybar * core
    mape = float(np.mean(np.abs((Y - Yhat) / Y)) * 100)
    return K, L, D, AI, H, Y, A, Yhat, mape


def render_bai1():
    st.title("🌱 Bài 1 · Hàm sản xuất Cobb-Douglas mở rộng với AI và số hoá")
    policy_box("<b>Bối cảnh:</b> GDP 2024 đạt 11.511,9 ngh.tỷ VND (+7,09%); năng suất lao động "
               "245 tr.VND/người (2025); KH-CN đóng góp 2,49% GDP. Câu hỏi: mô hình hoá nền kinh tế "
               "bằng Cobb-Douglas mở rộng có thêm số hoá D, năng lực AI và vốn nhân lực H — yếu tố "
               "nào đóng góp lớn nhất cho tăng trưởng?")
    st.latex(r"Y_t = A_t\,K_t^{\alpha} L_t^{\beta} D_t^{\gamma} AI_t^{\delta} H_t^{\theta},"
             r"\quad \alpha+\beta+\gamma+\delta+\theta = 1")

    st.sidebar.markdown("**⚙️ Hệ số co giãn (Bài 1)**")
    a = st.sidebar.slider("α — vốn vật chất K", 0.0, 0.6, 0.33, 0.01)
    b = st.sidebar.slider("β — lao động L", 0.0, 0.6, 0.42, 0.01)
    g = st.sidebar.slider("γ — số hoá D", 0.0, 0.3, 0.10, 0.01)
    d = st.sidebar.slider("δ — năng lực AI", 0.0, 0.3, 0.08, 0.01)
    th = st.sidebar.slider("θ — nhân lực số H", 0.0, 0.3, 0.07, 0.01)
    tot = a + b + g + d + th
    if abs(tot - 1) > 1e-9:
        st.sidebar.warning(f"Σ hệ số = {tot:.2f} ≠ 1 (CRS bị vi phạm)")

    K, L, D, AI, H, Y, A, Yhat, mape = cobb_compute(a, b, g, d, th)
    yrs = MACRO.year.values

    c1, c2, c3 = st.columns(3)
    with c1: card("TFP 2025 (A₂₀₂₅)", f"{A[-1]:.2f}", f"▲ từ {A[0]:.2f} (2020)", "#2e7d32")
    with c2: card("MAPE dự báo (Ā)", f"{mape:.2f}%", "Sai số tuyệt đối TB", ACCENT)
    with c3: card("Tăng GDP TB/năm", f"{(np.log(Y[-1]/Y[0])/5)*100:.2f}%", "CAGR 2020-2025", INK)

    section("1.4.1 & 1.4.2 · TFP và dự báo Ŷ vs thực tế")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(x=yrs, y=A, markers=True, labels={"x": "Năm", "y": "TFP Aₜ"})
        fig.update_traces(line_color=ACCENT, line_width=3)
        fig.update_layout(height=320, template="plotly_white", title="Năng suất nhân tố tổng hợp Aₜ")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_bar(x=yrs, y=Y, name="Y thực tế", marker_color=INK, opacity=.8)
        fig.add_trace(go.Scatter(x=yrs, y=Yhat, name="Ŷ dự báo (Ā)",
                      mode="lines+markers", line=dict(color=ACCENT2, width=3)))
        fig.update_layout(height=320, template="plotly_white", title="Ŷ dự báo vs Y thực tế")
        st.plotly_chart(fig, use_container_width=True)

    section("1.4.3 · Phân rã tăng trưởng (Δln) 2020-2025")
    dln = lambda x: np.log(x[-1] / x[0]) / (len(x) - 1)
    contrib = {"TFP (Aₜ)": dln(A), "Vốn K": a * dln(K), "Lao động L": b * dln(L),
               "Số hoá D": g * dln(D), "Năng lực AI": d * dln(AI), "Nhân lực H": th * dln(H)}
    gtot = sum(contrib.values())
    dfc = pd.DataFrame({"Yếu tố": list(contrib),
                        "Đóng góp (đ.vị/năm)": [round(v, 4) for v in contrib.values()],
                        "Tỷ trọng (%)": [round(v / gtot * 100, 1) for v in contrib.values()]})
    cc1, cc2 = st.columns([1.1, 1])
    with cc1:
        st.dataframe(dfc, use_container_width=True, hide_index=True)
    with cc2:
        fig = px.bar(dfc, x="Yếu tố", y="Tỷ trọng (%)", color="Yếu tố",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(height=320, template="plotly_white", showlegend=False,
                          title="Tỷ trọng đóng góp tăng trưởng")
        st.plotly_chart(fig, use_container_width=True)

    section("1.4.4 · Kịch bản dự báo GDP 2030")
    s1, s2, s3 = st.columns(3)
    D30 = s1.slider("D 2030 (% KTS/GDP)", 19.5, 40.0, 30.0, 0.5)
    AI30 = s2.slider("AI 2030 (ngh. DN số)", 80.0, 150.0, 100.0, 1.0)
    H30 = s3.slider("H 2030 (% LĐ đào tạo)", 29.2, 45.0, 35.0, 0.5)
    gKL = st.slider("Tăng trưởng K, L (%/năm)", 0.0, 10.0, 6.0, 0.5)
    gtfp = st.slider("Tăng TFP (%/năm)", 0.0, 3.0, 1.2, 0.1)
    K30 = K[-1] * (1 + gKL / 100) ** 5
    L30 = L[-1] * (1 + gKL / 100) ** 5
    A30 = A[-1] * (1 + gtfp / 100) ** 5
    Y30 = A30 * K30**a * L30**b * D30**g * AI30**d * H30**th
    cagr30 = (np.log(Y30 / Y[-1]) / 5) * 100
    r1, r2 = st.columns(2)
    with r1: card("GDP dự báo 2030 (ngh.tỷ VND)", f"{Y30:,.0f}", f"CAGR {cagr30:.2f}%/năm", "#2e7d32")
    with r2: card("Quy đổi (tỷ USD, ~25.500 VND/USD)", f"{Y30*1000/25.5:,.0f}", "tham khảo", INK)

    policy_discussion(1)


# ============================================================================
# BÀI 2 — LP PHÂN BỔ NGÂN SÁCH 4 HẠNG MỤC ĐẦU TƯ SỐ
# ============================================================================
def solve_bai2(budget=100.0, x3_min=20.0, tech_share=0.35,
               coef=(0.85, 1.20, 0.95, 1.35), mins=(25, 15, 20, 10)):
    c = [-v for v in coef]
    A_ub = [[1, 1, 1, 1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1],
            [tech_share, -(1 - tech_share), tech_share, -(1 - tech_share)]]
    b_ub = [budget, -mins[0], -mins[1], -x3_min, -mins[3], 0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")
    return res


def render_bai2():
    st.title("💰 Bài 2 · LP phân bổ ngân sách số (4 hạng mục)")
    policy_box("<b>Bối cảnh (QĐ 749/QĐ-TTg):</b> phân bổ 100.000 tỷ VND năm 2026 cho hạ tầng số (x₁), "
               "AI & dữ liệu (x₂), nhân lực số (x₃), R&D công nghệ (x₄) nhằm tối đa hoá tăng GDP kỳ vọng.")
    st.latex(r"\max Z = 0{,}85x_1 + 1{,}20x_2 + 0{,}95x_3 + 1{,}35x_4")
    st.caption("s.t. Σx ≤ B · x₁≥25 · x₂≥15 · x₃≥20 · x₄≥10 · x₂+x₄ ≥ 35%·Σx")

    st.sidebar.markdown("**⚙️ Tham số (Bài 2)**")
    budget = st.sidebar.slider("Ngân sách tổng B (ngh.tỷ)", 100, 160, 100, 5)
    x3min = st.sidebar.slider("Sàn nhân lực số x₃ ≥", 20, 40, 20, 1)
    tech = st.sidebar.slider("Tỷ trọng công nghệ chiến lược", 0.25, 0.55, 0.35, 0.05)

    res = solve_bai2(budget, x3min, tech)
    labels = ["x₁ Hạ tầng số", "x₂ AI & dữ liệu", "x₃ Nhân lực số", "x₄ R&D"]
    if not res.success:
        st.error("❌ Bài toán KHÔNG khả thi với cấu hình hiện tại (vd: x₃ quá cao so với ngân sách).")
        return
    x = res.x
    c1, c2, c3 = st.columns(3)
    with c1: card("Z* (tăng GDP kỳ vọng)", f"{-res.fun:.2f}", "ngh.tỷ VND", "#2e7d32")
    with c2: card("Hiệu suất Z*/B", f"{-res.fun/budget:.3f}", "đồng GDP/đồng vốn", INK)
    with c3: card("Phần dành R&D x₄", f"{x[3]:.1f}", f"{x[3]/budget*100:.0f}% ngân sách", ACCENT)

    col1, col2 = st.columns([1, 1])
    with col1:
        dfx = pd.DataFrame({"Hạng mục": labels, "Phân bổ (ngh.tỷ)": np.round(x, 2)})
        st.dataframe(dfx, use_container_width=True, hide_index=True)
        fig = px.pie(dfx, names="Hạng mục", values="Phân bổ (ngh.tỷ)", hole=.45,
                     color_discrete_sequence=PALETTE)
        fig.update_layout(height=300, template="plotly_white", title="Cơ cấu phân bổ tối ưu")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        section("2.4.2 · Giá đối ngẫu (shadow price)")
        try:
            shadow = res.ineqlin.marginals
            sp_budget = -shadow[0]
        except Exception:
            sp_budget = None
        if sp_budget is not None:
            st.metric("Shadow price ràng buộc ngân sách", f"{sp_budget:.3f}",
                      help="Tăng 1 ngh.tỷ ngân sách → Z* tăng thêm bấy nhiêu")
            st.info(f"💡 Mỗi **1 tỷ VND** ngân sách tăng thêm → GDP kỳ vọng tăng ~**{sp_budget:.2f} tỷ VND**. "
                    "Đây là cận trên hợp lý của chi phí cơ hội của vốn công.")
        section("2.4.3 · Độ nhạy Z*(B)")
        Bs = np.arange(100, 161, 5)
        Zs = [(-solve_bai2(B, x3min, tech).fun) if solve_bai2(B, x3min, tech).success else np.nan
              for B in Bs]
        fig = px.line(x=Bs, y=Zs, markers=True, labels={"x": "Ngân sách B", "y": "Z*"})
        fig.update_traces(line_color=ACCENT, line_width=3)
        fig.update_layout(height=240, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    section("2.4.4 · Ưu tiên nhân lực số: x₃ ≥ 30")
    r30 = solve_bai2(budget, 30, tech)
    if r30.success:
        st.success(f"✅ Vẫn khả thi. Z* = **{-r30.fun:.2f}** "
                   f"(thay đổi {(-r30.fun)-(-solve_bai2(budget,20,tech).fun):+.2f} so với x₃≥20). "
                   "Ưu tiên nhân lực số làm giảm nhẹ Z* do đẩy vốn khỏi R&D hệ số cao.")
    else:
        st.error("❌ Không khả thi khi x₃ ≥ 30 với ngân sách hiện tại.")

    policy_discussion(2)


# ============================================================================
# BÀI 3 — CHỈ SỐ ƯU TIÊN NGÀNH Priorityᵢ
# ============================================================================
def priority_scores(weights, w_risk):
    good = SECTORS[["growth_rate_2024_pct", "productivity_million_VND_per_worker",
                    "spillover_coef_0_1", "export_billion_USD", "labor_million",
                    "ai_readiness_0_100"]].values.astype(float)
    bad = SECTORS["automation_risk_pct"].values.astype(float)
    Xg = (good - good.min(0)) / (good.max(0) - good.min(0))
    Xb = (bad.max() - bad) / (bad.max() - bad.min())
    return Xg @ np.array(weights) + w_risk * Xb, Xg, Xb


def render_bai3():
    st.title("📊 Bài 3 · Chỉ số ưu tiên ngành (10 ngành)")
    policy_box("<b>Bối cảnh:</b> ngành nào nên đẩy mạnh chuyển đổi số & AI trước để tạo hiệu ứng lan toả "
               "tối đa? Xây dựng chỉ số ưu tiên định lượng từ 7 tiêu chí (chuẩn hoá min-max, đảo dấu Rủi ro).")
    st.latex(r"Priority_i = a_1 G + a_2 P + a_3 Spill + a_4 Exp + a_5 Emp + a_6 AI - a_7 Risk")

    st.sidebar.markdown("**⚙️ Trọng số (Bài 3)**")
    a1 = st.sidebar.slider("a₁ Tăng trưởng", 0.0, 0.4, 0.15, 0.01)
    a2 = st.sidebar.slider("a₂ Năng suất", 0.0, 0.4, 0.15, 0.01)
    a3 = st.sidebar.slider("a₃ Lan toả", 0.0, 0.4, 0.20, 0.01)
    a4 = st.sidebar.slider("a₄ Xuất khẩu", 0.0, 0.4, 0.15, 0.01)
    a5 = st.sidebar.slider("a₅ Việc làm", 0.0, 0.4, 0.10, 0.01)
    a6 = st.sidebar.slider("a₆ AI Readiness", 0.0, 0.4, 0.20, 0.01)
    a7 = st.sidebar.slider("a₇ Rủi ro (đảo dấu)", 0.0, 0.4, 0.15, 0.01)

    scores, Xg, Xb = priority_scores([a1, a2, a3, a4, a5, a6], a7)
    df = SECTORS[["sector_name_vi"]].copy()
    df["Priority"] = scores
    df = df.sort_values("Priority", ascending=False).reset_index(drop=True)
    df.index += 1

    section("3.4.2 · Xếp hạng 10 ngành theo Priority")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.dataframe(df.rename_axis("Hạng").reset_index(), use_container_width=True, hide_index=True)
    with c2:
        fig = px.bar(df.sort_values("Priority"), x="Priority", y="sector_name_vi",
                     orientation="h", color="Priority", color_continuous_scale="RdYlGn")
        fig.update_layout(height=400, template="plotly_white", yaxis_title="",
                          title="Chỉ số ưu tiên ngành")
        st.plotly_chart(fig, use_container_width=True)

    top3 = df.head(3)["sector_name_vi"].tolist()
    st.success(f"🏆 **Top-3 ưu tiên:** {', '.join(top3)}")

    section("3.4.3 · Độ nhạy theo trọng số AI Readiness (a₆)")
    a6_range = np.arange(0.05, 0.41, 0.05)
    heat = []
    for a6v in a6_range:
        w = np.array([a1, a2, a3, a4, a5, a6v]); w = w / w.sum() * (1 - a7)
        sc, _, _ = priority_scores(w, a7)
        heat.append(sc)
    heat = np.array(heat)
    fig = px.imshow(heat.T, x=[f"{v:.2f}" for v in a6_range],
                    y=SECTORS.sector_name_vi, color_continuous_scale="Reds",
                    labels={"x": "Trọng số a₆ (AI)", "y": "", "color": "Priority"}, aspect="auto")
    fig.update_layout(height=380, template="plotly_white", title="Heatmap độ nhạy Priority theo a₆")
    st.plotly_chart(fig, use_container_width=True)

    section("3.4.4 · So sánh 'Định hướng tăng trưởng' vs 'Bao trùm'")
    sc_g, _, _ = priority_scores([0.25, 0.20, 0.15, 0.20, 0.05, 0.10], 0.05)
    sc_i, _, _ = priority_scores([0.10, 0.10, 0.25, 0.05, 0.25, 0.10], 0.15)
    cmp = pd.DataFrame({"Ngành": SECTORS.sector_name_vi,
                        "Tăng trưởng": sc_g, "Bao trùm": sc_i})
    cmp["Top tăng trưởng"] = cmp["Tăng trưởng"].rank(ascending=False).astype(int)
    cmp["Top bao trùm"] = cmp["Bao trùm"].rank(ascending=False).astype(int)
    g3 = cmp.nsmallest(3, "Top tăng trưởng")["Ngành"].tolist()
    i3 = cmp.nsmallest(3, "Top bao trùm")["Ngành"].tolist()
    cA, cB = st.columns(2)
    cA.info("**Top-3 Tăng trưởng:** " + ", ".join(g3))
    cB.info("**Top-3 Bao trùm:** " + ", ".join(i3))

    policy_discussion(3)


# ============================================================================
# BÀI 4 — LP PHÂN BỔ NGÂN SÁCH NGÀNH-VÙNG (công bằng vùng)
# ============================================================================
BETA4 = np.array([  # 6 vùng x 4 hạng mục [I, D, AI, H]
    [1.15, 0.85, 0.55, 1.30], [0.95, 1.25, 1.40, 1.05], [1.05, 0.95, 0.85, 1.15],
    [1.20, 0.75, 0.45, 1.35], [0.90, 1.30, 1.55, 1.00], [1.10, 0.85, 0.65, 1.25]])
D0 = np.array([38, 78, 55, 32, 82, 48], float)


def solve_bai4(equity=True, lam=0.70, total=50000, floor=5000, cap=12000,
               h_floor=12000, gamma=0.002):
    # 24 biến x[r,j] flatten + 1 biến phụ M (Dmax). thứ tự: 24 x + M
    nv = 25
    c = np.zeros(nv); c[:24] = -BETA4.flatten()
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    # C1 tổng ngân sách
    row = np.zeros(nv); row[:24] = 1; A_ub.append(row); b_ub.append(total)
    for r in range(6):
        # C2 sàn vùng (>= floor -> -sum <= -floor)
        row = np.zeros(nv); row[r*4:r*4+4] = -1; A_ub.append(row); b_ub.append(-floor)
        # C3 trần vùng
        row = np.zeros(nv); row[r*4:r*4+4] = 1; A_ub.append(row); b_ub.append(cap)
    # C4 sàn nhân lực H (cột index 3 mỗi vùng)
    row = np.zeros(nv)
    for r in range(6): row[r*4+3] = -1
    A_ub.append(row); b_ub.append(-h_floor)
    if equity:
        for r in range(6):
            # D0r + gamma*x_D,r <= M
            row = np.zeros(nv); row[r*4+1] = gamma; row[24] = -1
            A_ub.append(row); b_ub.append(-D0[r])
            # D0r + gamma*x_D,r >= lam*M  -> -gamma*x_D,r + lam*M <= D0r
            row = np.zeros(nv); row[r*4+1] = -gamma; row[24] = lam
            A_ub.append(row); b_ub.append(D0[r])
    bounds = [(0, None)] * 24 + [(0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res


def render_bai4():
    st.title("🗺️ Bài 4 · LP phân bổ ngân sách số ngành-vùng")
    policy_box("<b>Bối cảnh (QĐ 411/QĐ-TTg):</b> phân bổ 50.000 tỷ VND cho 6 vùng × 4 hạng mục "
               "(I-hạ tầng, D-CĐS DN, AI, H-nhân lực) tối đa hoá GDP gain nhưng bảo đảm công bằng vùng.")
    st.latex(r"\max Z=\sum_r\sum_j \beta_{j,r}x_{j,r}\;;\;\; "
             r"D_r+\gamma x_{D,r}\ge \lambda\max_r(D_r+\gamma x_{D,r})")

    st.sidebar.markdown("**⚙️ Tham số (Bài 4)**")
    lam = st.sidebar.slider("λ — ngưỡng công bằng C5", 0.0, 0.9, 0.70, 0.05)
    cap = st.sidebar.slider("Trần ngân sách/vùng (ngh.tỷ)", 9000, 15000, 12000, 500)

    res = solve_bai4(True, lam, cap=cap)
    res_no = solve_bai4(False, cap=cap)
    if not res.success:
        st.error("❌ Không khả thi — thử giảm λ hoặc tăng trần vùng. "
                 f"(Lưu ý: với λ≈0,70 và D thấp ở Tây Nguyên/Bắc, ràng buộc công bằng có thể vô nghiệm; "
                 "đây là 'bẫy khả thi' có chủ đích trong đề bài.)")
        return
    X = res.x[:24].reshape(6, 4)
    items = ["I", "D", "AI", "H"]
    c1, c2, c3 = st.columns(3)
    with c1: card("Z* (có công bằng)", f"{-res.fun:,.0f}", "tỷ VND GDP gain", "#2e7d32")
    with c2: card("Z* (bỏ công bằng)", f"{-res_no.fun:,.0f}", "tỷ VND", INK)
    with c3:
        cost = (-res_no.fun) - (-res.fun)
        card("Chi phí công bằng vùng", f"{cost:,.0f}", f"{cost/(-res_no.fun)*100:.2f}% Z*", ACCENT)

    section("4.4.1 & 4.4.3 · Ma trận phân bổ tối ưu (heatmap)")
    dfX = pd.DataFrame(X, columns=items, index=REGION_SHORT)
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.dataframe(dfX.style.format("{:,.0f}"), use_container_width=True)
    with c2:
        fig = px.imshow(X, x=items, y=REGION_SHORT, color_continuous_scale="Reds",
                        text_auto=".0f", aspect="auto",
                        labels={"color": "ngh.tỷ VND"})
        fig.update_layout(height=360, template="plotly_white", title="Phân bổ x[vùng, hạng mục]")
        st.plotly_chart(fig, use_container_width=True)

    region_total = X.sum(1)
    top_r = REGIONS.region_name_vi.values[np.argmax(region_total)]
    st.success(f"🏆 Vùng nhận nhiều ngân sách nhất: **{top_r}** ({region_total.max():,.0f} ngh.tỷ). "
               f"Hạng mục nổi bật toàn quốc: **{items[int(np.argmax(X.sum(0)))]}**.")

    section("4.4.4 · So sánh có / không ràng buộc công bằng")
    Xno = res_no.x[:24].reshape(6, 4)
    cmp = pd.DataFrame({"Vùng": REGION_SHORT,
                        "Có công bằng": X.sum(1), "Bỏ công bằng": Xno.sum(1)})
    fig = px.bar(cmp, x="Vùng", y=["Có công bằng", "Bỏ công bằng"], barmode="group",
                 color_discrete_sequence=[ACCENT, "#94a3b8"])
    fig.update_layout(height=320, template="plotly_white", title="Tổng ngân sách mỗi vùng")
    st.plotly_chart(fig, use_container_width=True)

    policy_discussion(4)


# ============================================================================
# BÀI 5 — MIP LỰA CHỌN 15 DỰ ÁN CHUYỂN ĐỔI SỐ
# ============================================================================
PROJ = pd.DataFrame({
    "Mã": [f"P{i}" for i in range(1, 16)],
    "Tên": ["TT dữ liệu QG Hoà Lạc", "TT dữ liệu QG phía Nam", "5G toàn quốc",
            "VNeID 2.0", "Cổng DVC QG v3", "Y tế số QG", "Giáo dục số K-12",
            "TT AI QG + supercomputing", "Sandbox fintech", "Logistics thông minh",
            "Nông nghiệp số ĐBSCL", "Đào tạo 50.000 kỹ sư AI", "KCN bán dẫn BN-BG",
            "An ninh mạng QG (SOC)", "Open Data QG"],
    "C": [12000, 11500, 18000, 4500, 3200, 5800, 6500, 15000, 2500, 7200, 4800, 8500, 20000, 3800, 1500],
    "C1": [8500, 7500, 12000, 3500, 2500, 4000, 4500, 9000, 1800, 5000, 3500, 5500, 13000, 2800, 1200],
    "B": [21500, 20800, 32500, 9200, 6800, 11400, 12200, 28500, 5800, 13800, 8500, 16200, 35000, 7500, 3800],
})


def solve_bai5(total=80000, y1y2=80000, force_p1p2=False):
    C = PROJ.C.values.astype(float); C1 = PROJ.C1.values.astype(float); B = PROJ.B.values.astype(float)
    n = 15
    if HAS_PULP:
        m = pulp.LpProblem("sel", pulp.LpMaximize)
        y = pulp.LpVariable.dicts("y", range(n), cat="Binary")
        m += pulp.lpSum(B[i] * y[i] for i in range(n))
        m += pulp.lpSum(C[i] * y[i] for i in range(n)) <= total
        m += pulp.lpSum(C1[i] * y[i] for i in range(n)) <= 40000
        m += y[0] + y[1] <= 1
        m += y[7] <= y[11]; m += y[12] <= y[11]
        m += y[3] + y[4] >= 1; m += y[13] >= 1
        m += pulp.lpSum(y[i] for i in range(n)) >= 7
        m += pulp.lpSum(y[i] for i in range(n)) <= 11
        if force_p1p2: m += y[0] == 1; m += y[1] == 1
        st_ = m.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[m.status] != "Optimal":
            return None, None
        xv = np.array([y[i].value() for i in range(n)])
        return xv, pulp.value(m.objective)
    # fallback scipy.milp
    cons = [LinearConstraint(C, -np.inf, total), LinearConstraint(C1, -np.inf, 40000)]
    a = np.zeros(n); a[0] = a[1] = 1; cons.append(LinearConstraint(a, -np.inf, 1))
    a = np.zeros(n); a[7] = 1; a[11] = -1; cons.append(LinearConstraint(a, -np.inf, 0))
    a = np.zeros(n); a[12] = 1; a[11] = -1; cons.append(LinearConstraint(a, -np.inf, 0))
    a = np.zeros(n); a[3] = a[4] = 1; cons.append(LinearConstraint(a, 1, np.inf))
    a = np.zeros(n); a[13] = 1; cons.append(LinearConstraint(a, 1, np.inf))
    cons.append(LinearConstraint(np.ones(n), 7, 11))
    lb = np.zeros(n); ub = np.ones(n)
    if force_p1p2: lb[0] = lb[1] = 1
    res = milp(c=-B, constraints=cons, integrality=np.ones(n), bounds=Bounds(lb, ub))
    if not res.success: return None, None
    return np.round(res.x), -res.fun


def render_bai5():
    st.title("🎯 Bài 5 · MIP lựa chọn dự án chuyển đổi số (15 dự án)")
    policy_box("<b>Bối cảnh:</b> 15 dự án ứng cử, ngân sách 80.000 tỷ VND (2026-2030). Chọn tập dự án tối "
               "đa hoá lợi ích NPV với ràng buộc loại trừ, tiên quyết, cân đối lĩnh vực và số lượng 7≤Σy≤11.")
    solver_name = "PuLP/CBC" if HAS_PULP else "scipy.optimize.milp"
    st.caption(f"Giải bằng **{solver_name}** · biến nhị phân yᵢ ∈ {{0,1}}")

    st.sidebar.markdown("**⚙️ Tham số (Bài 5)**")
    total = st.sidebar.slider("Ngân sách tổng 5 năm (ngh.tỷ)", 60000, 120000, 80000, 5000)
    force = st.sidebar.checkbox("Bắt buộc cả P1 & P2 (redundancy)", False)

    xv, Z = solve_bai5(total, force_p1p2=force)
    if xv is None:
        st.error("❌ Bài toán không khả thi với cấu hình hiện tại.")
        return
    sel = PROJ[xv > 0.5].copy()
    cost = sel.C.sum()
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Tổng lợi ích Z*", f"{Z:,.0f}", "tỷ VND NPV", "#2e7d32")
    with c2: card("Tổng chi phí", f"{cost:,.0f}", f"{cost/total*100:.0f}% ngân sách", INK)
    with c3: card("Số dự án chọn", f"{int(xv.sum())}", "trong khoảng 7-11", ACCENT)
    with c4: card("NPV biên Z*/chi phí", f"{Z/cost:.2f}", "lần", "#2e7d32")

    section("5.4.1 · Danh mục dự án được chọn")
    show = PROJ.copy()
    show["Được chọn"] = np.where(xv > 0.5, "✅", "—")
    show["B/C"] = (show.B / show.C).round(2)
    st.dataframe(show[["Mã", "Tên", "C", "B", "B/C", "Được chọn"]],
                 use_container_width=True, hide_index=True, height=420)

    fig = px.scatter(PROJ, x="C", y="B", text="Mã", color=(xv > 0.5),
                     color_discrete_map={True: ACCENT, False: "#94a3b8"},
                     labels={"C": "Chi phí (tỷ)", "B": "Lợi ích NPV (tỷ)", "color": "Chọn"})
    fig.update_traces(textposition="top center", marker_size=12)
    fig.update_layout(height=380, template="plotly_white", title="Chi phí × Lợi ích NPV (15 dự án)")
    st.plotly_chart(fig, use_container_width=True)

    section("5.4.3 · Bắt buộc cả P1 & P2 (redundancy)")
    xv2, Z2 = solve_bai5(total, force_p1p2=True)
    base = solve_bai5(total, force_p1p2=False)[1]
    if xv2 is not None:
        st.warning(f"Khi ép chọn cả P1 & P2: Z* = **{Z2:,.0f}** "
                   f"(giảm **{base-Z2:,.0f}** tỷ so với {base:,.0f}). Redundancy 2 trung tâm dữ liệu "
                   "có chi phí cơ hội rõ rệt vì cả hai cùng chiếm ngân sách lớn.")
    else:
        st.error("❌ Không khả thi khi ép cả P1 & P2.")

    policy_discussion(5)


# ============================================================================
# BÀI 6 — TOPSIS XẾP HẠNG 6 VÙNG (sẵn sàng AI)
# ============================================================================
CRIT6 = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD", "digital_index_0_100",
         "ai_readiness_0_100", "trained_labor_pct", "rd_intensity_pct",
         "internet_penetration_pct", "gini_coef"]
CRIT6_LBL = ["GRDP/người", "FDI", "Digital Index", "AI Readiness",
             "LĐ đào tạo", "R&D/GRDP", "Internet", "Gini"]
IS_BENEFIT = np.array([1, 1, 1, 1, 1, 1, 1, 0])


def topsis(w):
    X = REGIONS[CRIT6].values.astype(float)
    R = X / np.sqrt((X**2).sum(0))
    V = R * w
    Astar = np.where(IS_BENEFIT == 1, V.max(0), V.min(0))
    Aneg = np.where(IS_BENEFIT == 1, V.min(0), V.max(0))
    Sstar = np.sqrt(((V - Astar)**2).sum(1))
    Sneg = np.sqrt(((V - Aneg)**2).sum(1))
    return Sneg / (Sstar + Sneg)


def entropy_weights():
    X = REGIONS[CRIT6].values.astype(float).copy()
    X[:, IS_BENEFIT == 0] = X[:, IS_BENEFIT == 0].max(0) - X[:, IS_BENEFIT == 0]
    P = X / X.sum(0)
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    return d / d.sum()


def render_bai6():
    st.title("🏆 Bài 6 · TOPSIS xếp hạng 6 vùng theo mức sẵn sàng AI")
    policy_box("<b>Bối cảnh (QĐ 127/QĐ-TTg):</b> chọn vùng triển khai trung tâm AI & sandbox dữ liệu trước. "
               "Áp dụng TOPSIS (chuẩn hoá vector, khoảng cách tới phương án lý tưởng tốt/xấu).")

    st.sidebar.markdown("**⚙️ Trọng số TOPSIS (Bài 6)**")
    use_entropy = st.sidebar.checkbox("Dùng trọng số Entropy (khách quan)", False)
    w_default = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
    w_ai = st.sidebar.slider("Trọng số AI Readiness", 0.05, 0.40, 0.20, 0.05)
    w = w_default.copy(); w[3] = w_ai; w = w / w.sum()
    if use_entropy:
        w = entropy_weights()

    C = topsis(w)
    df = REGIONS[["region_name_vi"]].copy()
    df["C*"] = C
    df = df.sort_values("C*", ascending=False).reset_index(drop=True)
    df.index += 1

    c1, c2 = st.columns([1, 1.2])
    with c1:
        section("6.4.1 · Xếp hạng C*")
        st.dataframe(df.rename_axis("Hạng").reset_index(), use_container_width=True, hide_index=True)
        st.markdown("**Bộ trọng số đang dùng:**")
        wdf = pd.DataFrame({"Tiêu chí": CRIT6_LBL, "w": np.round(w, 3)})
        st.dataframe(wdf, use_container_width=True, hide_index=True, height=320)
    with c2:
        fig = px.bar(df.sort_values("C*"), x="C*", y="region_name_vi", orientation="h",
                     color="C*", color_continuous_scale="Reds")
        fig.update_layout(height=320, template="plotly_white", yaxis_title="",
                          title="Hệ số gần gũi C* (càng cao càng ưu tiên)")
        st.plotly_chart(fig, use_container_width=True)
        # so sánh chuyên gia vs entropy
        Ce, Cd = topsis(entropy_weights()), topsis(w_default / w_default.sum())
        comp = pd.DataFrame({"Vùng": REGION_SHORT, "Chuyên gia": Cd, "Entropy": Ce})
        fig2 = px.line(comp, x="Vùng", y=["Chuyên gia", "Entropy"], markers=True,
                       color_discrete_sequence=[ACCENT, INK])
        fig2.update_layout(height=260, template="plotly_white", title="6.4.2 · So sánh trọng số")
        st.plotly_chart(fig2, use_container_width=True)

    top3 = df.head(3)["region_name_vi"].tolist()
    st.success(f"🏆 **3 vùng đặt trung tâm AI (QĐ 127):** {', '.join(top3)}")

    section("6.4.3 · Độ nhạy theo trọng số AI Readiness")
    rng = np.arange(0.10, 0.41, 0.05)
    ranks = []
    for v in rng:
        wv = w_default.copy(); wv[3] = v; wv = wv / wv.sum()
        ranks.append(topsis(wv))
    ranks = np.array(ranks)
    fig = go.Figure()
    for i in range(6):
        fig.add_trace(go.Scatter(x=rng, y=ranks[:, i], mode="lines+markers",
                                 name=REGION_SHORT[i], line=dict(color=PALETTE[i])))
    fig.update_layout(height=320, template="plotly_white", xaxis_title="Trọng số AI Readiness",
                      yaxis_title="C*", title="Ổn định Top-3 khi tăng w_AI")
    st.plotly_chart(fig, use_container_width=True)

    policy_discussion(6)


# ============================================================================
# BÀI 7 — TỐI ƯU ĐA MỤC TIÊU PARETO (NSGA-II nhẹ bằng numpy)
# ============================================================================
E_R = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])      # phát thải
RHO_R = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])    # rủi ro/AI
SIG_R = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])    # giảm rủi ro/H


def evaluate7(X):  # X: (N,6,4)
    f1 = (BETA4 * X).sum((1, 2))                      # max GDP gain
    sums = X.sum(2)
    f2 = np.abs(sums - sums.mean(1, keepdims=True)).mean(1)   # min Gini~MAD
    f3 = (E_R * (X[:, :, 0] + X[:, :, 2])).sum(1)     # min phát thải
    f4 = (RHO_R * X[:, :, 2]).sum(1) - (SIG_R * X[:, :, 3]).sum(1)  # min rủi ro
    return np.column_stack([-f1, f2, f3, f4])         # tất cả về dạng min


def feasible_sample(n=4000, total=50000, floor=5000, cap=12000, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        reg = rng.uniform(floor, cap, size=(n, 6))
        reg = reg / reg.sum(1, keepdims=True) * rng.uniform(0.85, 1.0, (n, 1)) * total
        ok = (reg.sum(1) <= total + 1) & (reg.min(1) >= floor - 1) & (reg.max(1) <= cap + 1)
        for r in reg[ok]:
            w = rng.dirichlet(np.ones(4), size=6)      # tỷ lệ I,D,AI,H mỗi vùng
            X = (r[:, None] * w)
            out.append(X)
            if len(out) >= n:
                break
    return np.array(out[:n])


def pareto_front(F):
    n = len(F); dom = np.ones(n, bool)
    for i in range(n):
        if not dom[i]:
            continue
        d = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
        d[i] = False
        dom[d] = False
    return dom


@st.cache_data
def run_nsga2(n=4000):
    X = feasible_sample(n)
    F = evaluate7(X)
    mask = pareto_front(F)
    return X[mask], F[mask]


def render_bai7():
    st.title("🌐 Bài 7 · Tối ưu đa mục tiêu Pareto (NSGA-II)")
    policy_box("<b>Bối cảnh:</b> phát triển kinh tế số đồng thời 4 mục tiêu xung đột — (1) tăng trưởng GDP, "
               "(2) bao trùm (giảm Gini), (3) môi trường (COP26), (4) an ninh dữ liệu. Nghiệm là <b>tập "
               "Pareto</b>, không phải một điểm tối ưu duy nhất.")
    st.caption("Triển khai bằng lấy mẫu tiến hoá + lọc không-bị-trội (numpy) để chạy nhẹ trên web; "
               "tương đương ý tưởng NSGA-II của pymoo.")

    Xp, Fp = run_nsga2()
    f1 = -Fp[:, 0]; f2 = Fp[:, 1]; f3 = Fp[:, 2]; f4 = Fp[:, 3]
    c1, c2, c3 = st.columns(3)
    with c1: card("Số nghiệm Pareto", f"{len(Fp)}", "không bị trội", INK)
    with c2: card("GDP gain cao nhất", f"{f1.max():,.0f}", "tỷ VND (f₁)", "#2e7d32")
    with c3: card("Bao trùm tốt nhất", f"{f2.min():,.0f}", "MAD nhỏ nhất (f₂)", ACCENT)

    section("7.4.2 · Biên Pareto 3D & toạ độ song song")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter_3d(x=f1, y=f2, z=f3, color=f4,
                            labels={"x": "f₁ GDP", "y": "f₂ Gini", "z": "f₃ Phát thải", "color": "f₄"},
                            color_continuous_scale="RdYlGn_r")
        fig.update_traces(marker_size=4)
        fig.update_layout(height=420, template="plotly_white", title="Biên Pareto (f₁,f₂,f₃)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        dpc = pd.DataFrame({"f₁ Tăng trưởng": f1, "f₂ Bao trùm": f2,
                            "f₃ Môi trường": f3, "f₄ An ninh": f4})
        fig = px.parallel_coordinates(dpc, color="f₁ Tăng trưởng",
                                      color_continuous_scale="Reds")
        fig.update_layout(height=420, template="plotly_white", title="Parallel coordinates 4 mục tiêu")
        st.plotly_chart(fig, use_container_width=True)

    section("7.4.3 · Nghiệm thoả hiệp (TOPSIS trên tập Pareto)")
    w74 = st.columns(4)
    wg = w74[0].slider("Tăng trưởng", 0.0, 1.0, 0.40, 0.05)
    wi = w74[1].slider("Bao trùm", 0.0, 1.0, 0.25, 0.05)
    we = w74[2].slider("Môi trường", 0.0, 1.0, 0.20, 0.05)
    ws = w74[3].slider("An ninh", 0.0, 1.0, 0.15, 0.05)
    W = np.array([wg, wi, we, ws]); W = W / W.sum()
    # TOPSIS trên F (tất cả là min) -> chuẩn hoá
    Fn = Fp / np.sqrt((Fp**2).sum(0))
    Vt = Fn * W
    best = Vt.min(0); worst = Vt.max(0)
    Sp = np.sqrt(((Vt - best)**2).sum(1)); Sn = np.sqrt(((Vt - worst)**2).sum(1))
    Cc = Sn / (Sp + Sn)
    idx = int(np.argmax(Cc))
    fmax = int(np.argmax(f1))   # nghiệm tăng trưởng cao nhất
    rcomp = pd.DataFrame({
        "Nghiệm": ["Thoả hiệp (TOPSIS)", "Tăng trưởng cao nhất"],
        "f₁ GDP": [f1[idx], f1[fmax]], "f₂ Bao trùm": [f2[idx], f2[fmax]],
        "f₃ Môi trường": [f3[idx], f3[fmax]], "f₄ An ninh": [f4[idx], f4[fmax]]})
    st.dataframe(rcomp.style.format({"f₁ GDP": "{:,.0f}", "f₂ Bao trùm": "{:,.0f}",
                 "f₃ Môi trường": "{:,.0f}", "f₄ An ninh": "{:,.0f}"}),
                 use_container_width=True, hide_index=True)
    dg = (f1[fmax] - f1[idx]) / f1[idx] * 100
    di = (f2[fmax] - f2[idx]) / abs(f2[idx]) * 100
    st.info(f"💡 **Chi phí cơ hội:** nghiệm tăng trưởng cao nhất hơn nghiệm thoả hiệp {dg:+.1f}% GDP "
            f"nhưng xấu đi {di:+.1f}% về bao trùm (Gini). Đánh đổi tăng trưởng ↔ công bằng là rõ rệt.")

    policy_discussion(7)


# ============================================================================
# BÀI 8 — TỐI ƯU ĐỘNG LIÊN THỜI GIAN 2026-2035
# ============================================================================
@st.cache_data
def solve_bai8(rho=0.97, shock=False, strategy="optimize"):
    T = 10
    dK, dD, dAI = 0.05, 0.12, 0.15
    thH, mu = 0.8, 0.02
    p1, p2, p3 = 0.003, 0.002, 0.004
    K0, L, D0_, AI0, H0, A0 = 27500., 54., 20.3, 86., 30., 34.9
    aK, bL, gD, dAI_, thH_ = 0.33, 0.42, 0.10, 0.08, 0.07

    def simulate(shares):
        K, D, AI, H, A = K0, D0_, AI0, H0, A0
        Cs, Ks, Ds, AIs, Hs, Ys = [], [], [], [], [], []
        for t in range(T):
            Y = A * K**aK * L**bL * D**gD * AI**dAI_ * H**thH_
            if shock and t == 2:
                Y *= 0.92
            sK, sD, sAI, sH = shares
            invest = (sK + sD + sAI + sH)
            inv_total = min(invest, 0.6) * Y       # tối đa 60% Y dành đầu tư
            IK, ID, IAI, IH = inv_total * np.array([sK, sD, sAI, sH]) / max(invest, 1e-6)
            C = Y - inv_total
            Cs.append(max(C, 1e-6)); Ks.append(K); Ds.append(D); AIs.append(AI); Hs.append(H); Ys.append(Y)
            K = (1 - dK) * K + IK
            D = (1 - dD) * D + ID / 100
            AI = (1 - dAI) * AI + IAI / 20
            H = H + thH * IH / 200 - mu * H
            A = A * (1 + p1 * D + p2 * AI + p3 * H) ** 0.001
        return np.array(Cs), np.array(Ks), np.array(Ds), np.array(AIs), np.array(Hs), np.array(Ys)

    def neg_welfare(shares):
        Cs = simulate(shares)[0]
        return -sum(rho**t * np.log(Cs[t]) for t in range(T))

    if strategy == "even":
        sh = np.array([0.15, 0.05, 0.05, 0.05])
    elif strategy == "frontload":
        sh = np.array([0.25, 0.08, 0.08, 0.08])
    else:
        res = minimize(neg_welfare, x0=[0.15, 0.05, 0.05, 0.05], method="SLSQP",
                       bounds=[(0.01, 0.4)] * 4)
        sh = res.x
    Cs, Ks, Ds, AIs, Hs, Ys = simulate(sh)
    W = float(sum(rho**t * np.log(Cs[t]) for t in range(T)))
    return sh, Cs, Ks, Ds, AIs, Hs, Ys, W


def render_bai8():
    st.title("⏳ Bài 8 · Tối ưu động phân bổ liên thời gian 2026-2035")
    policy_box("<b>Bối cảnh (Đại hội XIII):</b> thiết kế quỹ đạo phân bổ vốn 10 năm tối đa hoá phúc lợi "
               "xã hội chiết khấu, có động học tích luỹ K/D/AI/H và cập nhật TFP nội sinh.")
    st.latex(r"\max \sum_{t} \rho^{t}\,\ln C_t \quad s.t.\quad C_t+\sum I_{j,t}\le Y_t")

    st.sidebar.markdown("**⚙️ Tham số (Bài 8)**")
    rho = st.sidebar.slider("ρ — hệ số chiết khấu", 0.85, 0.99, 0.97, 0.01)
    shock = st.sidebar.checkbox("Cú sốc 2028 (−8% Y, kiểu bão Yagi)", False)

    sh, Cs, Ks, Ds, AIs, Hs, Ys, W = solve_bai8(rho, shock, "optimize")
    yrs = list(range(2026, 2036))
    c1, c2, c3 = st.columns(3)
    with c1: card("Phúc lợi tổng W*", f"{W:.3f}", "Σ ρᵗ ln Cₜ", "#2e7d32")
    with c2: card("Y 2035 (ngh.tỷ)", f"{Ys[-1]:,.0f}", f"từ {Ys[0]:,.0f} (2026)", INK)
    with c3: card("Tỷ lệ đầu tư K:H", f"{sh[0]/max(sh[3],1e-6):.1f}:1", "trung bình", ACCENT)

    section("8.3.2 · Quỹ đạo tối ưu K, D, AI, H, Y, C")
    fig = make_subplots(rows=2, cols=3, subplot_titles=("Vốn K", "Số hoá D", "Năng lực AI",
                        "Nhân lực H", "Sản lượng Y", "Tiêu dùng C"))
    series = [Ks, Ds, AIs, Hs, Ys, Cs]
    pos = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    for s, (r, cl), col in zip(series, pos, PALETTE):
        fig.add_trace(go.Scatter(x=yrs, y=s, mode="lines+markers",
                      line=dict(color=col, width=2.5), showlegend=False), row=r, col=cl)
    fig.update_layout(height=460, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    section("8.3.4 · So sánh chiến lược đầu tư")
    _, _, _, _, _, _, _, W_even = solve_bai8(rho, shock, "even")
    _, _, _, _, _, _, _, W_front = solve_bai8(rho, shock, "frontload")
    cmp = pd.DataFrame({"Chiến lược": ["Tối ưu (SLSQP)", "Trải đều", "Front-load"],
                        "Phúc lợi W": [W, W_even, W_front]})
    fig = px.bar(cmp, x="Chiến lược", y="Phúc lợi W", color="Chiến lược",
                 color_discrete_sequence=PALETTE)
    fig.update_layout(height=300, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    policy_discussion(8)


# ============================================================================
# BÀI 9 — TÁC ĐỘNG AI TỚI THỊ TRƯỜNG LAO ĐỘNG
# ============================================================================
SEC9 = ["Nông-Lâm-TS", "CN chế biến", "Xây dựng", "Bán buôn-lẻ",
        "Tài chính-NH", "Logistics", "CNTT-TT", "Giáo dục"]
L9 = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
RISK9 = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
A1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
B1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
C1c = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
D1 = np.array([50, 32, 42, 38, 26, 36, 24, 62])


def solve_bai9(budget=30000, cap5pct=False):
    # biến: x_AI[8], x_H[8] => 16 biến
    n = 16
    # NetJob_i = a1*xAI + b1*xH - (c1*risk)*xAI ; maximize sum
    coef_AI = A1 - C1c * RISK9
    coef_H = B1
    c = -np.concatenate([coef_AI, coef_H])
    A_ub, b_ub = [], []
    # ngân sách
    row = np.ones(n); A_ub.append(row); b_ub.append(budget)
    for i in range(8):
        # NetJob_i >= 0 -> -(coefAI*xAI + b1*xH) <= 0
        row = np.zeros(n); row[i] = -coef_AI[i]; row[8 + i] = -coef_H[i]
        A_ub.append(row); b_ub.append(0)
        # Displaced <= RetrainCap : c1*risk*xAI - d1*xH <= 0
        row = np.zeros(n); row[i] = C1c[i] * RISK9[i]; row[8 + i] = -D1[i]
        A_ub.append(row); b_ub.append(0)
        if cap5pct:
            # Displaced <= 0.05*L (đơn vị: việc/tỷ * tỷ -> nghìn việc; L triệu)
            row = np.zeros(n); row[i] = C1c[i] * RISK9[i]
            A_ub.append(row); b_ub.append(0.05 * L9[i] * 1000)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n, method="highs")
    return res, coef_AI, coef_H


def render_bai9():
    st.title("👷 Bài 9 · Tác động AI tới thị trường lao động")
    policy_box("<b>Bối cảnh (ILO/OECD 2024):</b> 30-50% việc làm Việt Nam có nguy cơ tự động hoá nhưng AI "
               "cũng tạo việc mới. Phân bổ 30.000 tỷ cho 8 ngành (x_AI, x_H) tối đa hoá NetJob ròng, "
               "ràng buộc NetJobᵢ≥0 và Displacedᵢ ≤ RetrainCapᵢ.")
    st.latex(r"NetJob_i = a_{1i}x^{AI}_i + b_{1i}x^{H}_i - c_{1i}\,risk_i\,x^{AI}_i \ge 0")

    st.sidebar.markdown("**⚙️ Tham số (Bài 9)**")
    budget = st.sidebar.slider("Ngân sách (ngh.tỷ)", 20000, 50000, 30000, 5000)
    cap5 = st.sidebar.checkbox("Ràng buộc: mất ≤ 5% LĐ/ngành (9.4.4)", False)

    res, cAI, cH = solve_bai9(budget, cap5)
    if not res.success:
        st.error("❌ Không khả thi với ràng buộc hiện tại (vd: ràng buộc 5% quá chặt cho ngành rủi ro cao).")
        return
    xAI = res.x[:8]; xH = res.x[8:]
    NewJob = A1 * xAI; Upgrade = B1 * xH
    Displaced = C1c * RISK9 * xAI; NetJob = NewJob + Upgrade - Displaced
    c1, c2, c3 = st.columns(3)
    with c1: card("Tổng NetJob ròng", f"{NetJob.sum():,.0f}", "nghìn việc làm", "#2e7d32")
    with c2: card("Tổng việc dịch chuyển", f"{Displaced.sum():,.0f}", "nghìn việc (Displaced)", ACCENT)
    with c3: card("Phần ngân sách → đào tạo H", f"{xH.sum()/budget*100:.0f}%", "ưu tiên kỹ năng", INK)

    section("9.4.1 · Phân bổ tối ưu & NetJob theo ngành")
    df9 = pd.DataFrame({"Ngành": SEC9, "x_AI": np.round(xAI, 0), "x_H": np.round(xH, 0),
                        "NewJob": np.round(NewJob, 0), "Displaced": np.round(Displaced, 0),
                        "NetJob": np.round(NetJob, 0)})
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(df9, use_container_width=True, hide_index=True, height=340)
    with c2:
        fig = px.bar(df9, x="Ngành", y=["x_AI", "x_H"], barmode="stack",
                     color_discrete_sequence=[ACCENT, ACCENT2])
        fig.update_layout(height=340, template="plotly_white", title="Phân bổ x_AI vs x_H")
        st.plotly_chart(fig, use_container_width=True)

    section("9.4.3 · Luồng dịch chuyển lao động (Sankey)")
    src, tgt, val = [], [], []
    labels = SEC9 + ["Việc mới (AI)", "Nâng cấp (H)", "Dịch chuyển (mất)"]
    for i in range(8):
        src += [i, i, i]; tgt += [8, 9, 10]
        val += [max(NewJob[i], 0.1), max(Upgrade[i], 0.1), max(Displaced[i], 0.1)]
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=14, thickness=14,
                  color=PALETTE[:8] + ["#2e7d32", "#1565c0", "#d4001f"]),
        link=dict(source=src, target=tgt, value=val)))
    fig.update_layout(height=420, template="plotly_white", title="Luồng lao động 8 ngành")
    st.plotly_chart(fig, use_container_width=True)

    section("9.4.2 · Ngưỡng đào tạo tối thiểu ngành CN chế biến chế tạo")
    # x_H để NetJob>=0 khi xAI tối đa: b1*xH >= (c1*risk - a1)*xAI
    i = 1
    st.info(f"Với ngành **CN chế biến chế tạo** (rủi ro 42%): hệ số NetJob của AI là "
            f"a₁−c₁·risk = {cAI[i]:.2f} (>0 → AI vẫn tạo việc ròng). Khi đầu tư AI lớn, cần x_H ≥ "
            f"phần bù để giữ NetJob≥0. Mô hình tự cân đối qua ràng buộc Displaced ≤ RetrainCap.")

    if cap5:
        st.warning("⚠️ Ràng buộc 'mất ≤ 5% LĐ/ngành' siết mạnh các ngành rủi ro cao (Tài chính-NH, "
                   "CN chế biến) → giảm dư địa đầu tư AID, tổng NetJob thấp hơn nhưng an sinh tốt hơn.")

    policy_discussion(9)


# ============================================================================
# BÀI 10 — QUY HOẠCH NGẪU NHIÊN HAI GIAI ĐOẠN (VSS & EVPI)
# ============================================================================
J10 = ["I", "D", "AI", "H"]
P10 = {"s1": 0.30, "s2": 0.45, "s3": 0.20, "s4": 0.05}
S10 = list(P10)
BETA0 = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}
BETA_S = {
    "s1": {"I": 1.25, "D": 1.35, "AI": 1.55, "H": 1.05},
    "s2": {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95},
    "s3": {"I": 0.75, "D": 0.85, "AI": 0.90, "H": 1.00},
    "s4": {"I": 0.40, "D": 0.50, "AI": 0.55, "H": 1.10},
}
CAP1 = 30000     # trần hấp thụ mỗi hạng mục GĐ1
B1_TOTAL = 65000
B2_TOTAL = 15000


def _stage2_value(x, s):
    # tối ưu second-stage cho 1 kịch bản, x cố định
    b = BETA_S[s]
    c = -np.array([b[j] for j in J10])
    A_ub = [np.ones(4)]; b_ub = [B2_TOTAL]
    # y_AI <= 0.5 x_H
    row = np.zeros(4); row[2] = 1; A_ub.append(row); b_ub.append(0.5 * x[3])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")
    return -res.fun if res.success else 0.0, (res.x if res.success else np.zeros(4))


def solve_SP():
    # x[4] + y[s,4]*4 = 20 biến
    nv = 20
    c = np.zeros(nv)
    for i, j in enumerate(J10): c[i] = BETA0[j]
    idx = lambda s, j: 4 + S10.index(s) * 4 + J10.index(j)
    for s in S10:
        for j in J10: c[idx(s, j)] = P10[s] * BETA_S[s][j]
    A_ub, b_ub = [], []
    row = np.zeros(nv); row[:4] = 1; A_ub.append(row); b_ub.append(B1_TOTAL)
    for s in S10:
        row = np.zeros(nv)
        for j in J10: row[idx(s, j)] = 1
        A_ub.append(row); b_ub.append(B2_TOTAL)
        row = np.zeros(nv); row[idx(s, "AI")] = 1; row[3] = -0.5; A_ub.append(row); b_ub.append(0)
    bounds = [(0, CAP1)] * 4 + [(0, None)] * 16
    res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    x = res.x[:4]
    return x, -res.fun


def solve_one_scenario(s):
    # first+second stage tối ưu khi biết trước kịch bản s (wait-and-see)
    nv = 8
    c = np.zeros(nv)
    for i, j in enumerate(J10): c[i] = BETA0[j]; c[4 + i] = BETA_S[s][j]
    A_ub, b_ub = [], []
    row = np.zeros(nv); row[:4] = 1; A_ub.append(row); b_ub.append(B1_TOTAL)
    row = np.zeros(nv); row[4:] = 1; A_ub.append(row); b_ub.append(B2_TOTAL)
    row = np.zeros(nv); row[6] = 1; row[3] = -0.5; A_ub.append(row); b_ub.append(0)
    bounds = [(0, CAP1)] * 4 + [(0, None)] * 4
    res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res.x[:4], -res.fun


def solve_EV():
    # kịch bản kỳ vọng: beta_s trung bình theo xác suất
    bbar = {j: sum(P10[s] * BETA_S[s][j] for s in S10) for j in J10}
    nv = 8
    c = np.zeros(nv)
    for i, j in enumerate(J10): c[i] = BETA0[j]; c[4 + i] = bbar[j]
    A_ub, b_ub = [], []
    row = np.zeros(nv); row[:4] = 1; A_ub.append(row); b_ub.append(B1_TOTAL)
    row = np.zeros(nv); row[4:] = 1; A_ub.append(row); b_ub.append(B2_TOTAL)
    row = np.zeros(nv); row[6] = 1; row[3] = -0.5; A_ub.append(row); b_ub.append(0)
    bounds = [(0, CAP1)] * 4 + [(0, None)] * 4
    res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res.x[:4]


@st.cache_data
def bai10_metrics():
    x_sp, RP = solve_SP()
    # WS
    WS = 0.0; x_ws = {}
    for s in S10:
        xs, vs = solve_one_scenario(s); WS += P10[s] * vs; x_ws[s] = xs
    # EEV: dùng x_EV, đánh giá kỳ vọng second-stage
    x_ev = solve_EV()
    first_ev = sum(BETA0[j] * x_ev[i] for i, j in enumerate(J10))
    EEV = first_ev + sum(P10[s] * _stage2_value(x_ev, s)[0] for s in S10)
    VSS = max(RP - EEV, 0.0)
    EVPI = max(WS - RP, 0.0)
    return x_sp, x_ev, RP, EEV, WS, VSS, EVPI


def render_bai10():
    st.title("🎲 Bài 10 · Quy hoạch ngẫu nhiên hai giai đoạn")
    policy_box("<b>Bối cảnh:</b> Việt Nam độ mở thương mại ~180% GDP → tăng trưởng phụ thuộc kịch bản toàn "
               "cầu. Quyết định GĐ1 'here-and-now' (≤65.000 tỷ, giữ 15.000 tỷ dự phòng) trước khi biết kịch "
               "bản; GĐ2 'recourse' điều chỉnh theo 4 kịch bản (lạc quan/cơ sở/bi quan/khủng hoảng).")

    st.dataframe(pd.DataFrame({
        "Kịch bản": ["Lạc quan", "Cơ sở", "Bi quan", "Khủng hoảng"],
        "Xác suất": [0.30, 0.45, 0.20, 0.05],
        "β_AI": [1.55, 1.25, 0.90, 0.55], "β_H": [1.05, 0.95, 1.00, 1.10]}),
        use_container_width=True, hide_index=True)

    x_sp, x_ev, RP, EEV, WS, VSS, EVPI = bai10_metrics()
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("RP (stochastic)", f"{RP:,.0f}", "lời giải SP", "#2e7d32")
    with c2: card("EEV (lời giải kỳ vọng)", f"{EEV:,.0f}", "dùng x_EV", INK)
    with c3: card("VSS = RP − EEV", f"{VSS:,.0f}", "giá trị tư duy xác suất", ACCENT)
    with c4: card("EVPI = WS − RP", f"{EVPI:,.0f}", "giá trị thông tin hoàn hảo", "#6a1b9a")

    section("10.5.1 & 10.5.2 · Quyết định GĐ1: SP vs EV")
    dfx = pd.DataFrame({"Hạng mục": J10,
                        "SP (stochastic)": np.round(x_sp, 0),
                        "EV (kỳ vọng)": np.round(x_ev, 0)})
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.dataframe(dfx, use_container_width=True, hide_index=True)
    with c2:
        fig = px.bar(dfx, x="Hạng mục", y=["SP (stochastic)", "EV (kỳ vọng)"],
                     barmode="group", color_discrete_sequence=[ACCENT, "#94a3b8"])
        fig.update_layout(height=320, template="plotly_white", title="Phân bổ GĐ1")
        st.plotly_chart(fig, use_container_width=True)

    dh = x_sp[3] - x_ev[3]
    st.info(f"💡 Lời giải SP đầu tư nhân lực H {'**nhiều hơn**' if dh>0 else 'khác'} EV "
            f"({dh:+,.0f} ngh.tỷ): H là 'hàng hoá bảo hiểm' — trong kịch bản khủng hoảng β_H cao nhất (1,10) "
            "và H còn mở khoá năng lực AI GĐ2 (ràng buộc y_AI ≤ 0,5·x_H).")

    section("10.5.3 · Phân rã giá trị: WS ≥ RP ≥ EEV")
    fig = go.Figure(go.Bar(x=["EEV", "RP (SP)", "WS"], y=[EEV, RP, WS],
                           marker_color=["#94a3b8", ACCENT, "#2e7d32"], text=[f"{v:,.0f}" for v in [EEV, RP, WS]]))
    fig.update_layout(height=320, template="plotly_white", title="EEV ≤ RP ≤ WS",
                      yaxis_title="GDP gain kỳ vọng")
    st.plotly_chart(fig, use_container_width=True)

    policy_discussion(10)


# ============================================================================
# BÀI 11 — Q-LEARNING CHO CHÍNH SÁCH KINH TẾ THÍCH NGHI
# ============================================================================
ALLOC11 = {0: [0.70, 0.10, 0.10, 0.10], 1: [0.40, 0.25, 0.15, 0.20],
           2: [0.25, 0.45, 0.15, 0.15], 3: [0.20, 0.20, 0.45, 0.15],
           4: [0.30, 0.20, 0.10, 0.40]}
ACT_NAME = ["a0 Truyền thống", "a1 Cân bằng", "a2 Số hoá nhanh", "a3 AI dẫn dắt", "a4 Bao trùm"]
W11 = np.array([0.40, 0.25, 0.20, 0.15])


def step_env(state, action, rng):
    a = np.array(ALLOC11[action])
    g, d, ai, u = state
    # ΔGDP ~ trọng đầu tư K + D + AI; thất nghiệp giảm theo H; cyber theo AI; phát thải theo K
    dGDP = 0.5 * a[0] + 0.8 * a[1] + 1.0 * a[2] + 0.3 * a[3] + 0.05 * rng.standard_normal()
    dUnemp = -0.6 * a[3] + 0.4 * a[2] - 0.1 * a[1]
    cyber = 0.7 * a[2] - 0.3 * a[3]
    emis = 0.6 * a[0] + 0.2 * a[2]
    reward = W11[0] * dGDP - W11[1] * dUnemp - W11[2] * cyber - W11[3] * emis
    # cập nhật trạng thái rời rạc (clip 0..2)
    def upd(v, x): return int(np.clip(v + (1 if x > 0.15 else (-1 if x < -0.05 else 0)), 0, 2))
    ns = np.array([upd(g, dGDP), upd(d, a[1]), upd(ai, a[2]), upd(u, -dUnemp)])
    return ns, reward


@st.cache_data
def train_qlearning(episodes=4000, alpha=0.1, gamma=0.95, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((3, 3, 3, 3, 5))
    curve = []
    for ep in range(episodes):
        s = np.array([1, 1, 0, 1]); tot = 0.0
        eps = max(0.05, 1.0 - ep / (episodes * 0.5))
        for t in range(10):
            a = rng.integers(5) if rng.random() < eps else int(np.argmax(Q[tuple(s)]))
            ns, r = step_env(s, a, rng)
            Q[tuple(s) + (a,)] += alpha * (r + gamma * Q[tuple(ns)].max() - Q[tuple(s) + (a,)])
            s = ns; tot += r
        curve.append(tot)
    # smoothing
    curve = np.convolve(curve, np.ones(50) / 50, mode="valid")
    return Q, curve


def eval_policy(Q, mode, seed=1, episodes=200):
    rng = np.random.default_rng(seed); tot = 0.0
    for _ in range(episodes):
        s = np.array([1, 1, 0, 1])
        for t in range(10):
            if mode == "q": a = int(np.argmax(Q[tuple(s)]))
            elif mode == "a1": a = 1
            elif mode == "a3": a = 3
            else: a = rng.integers(5)
            s, r = step_env(s, a, rng); tot += r
    return tot / episodes


def render_bai11():
    st.title("🤖 Bài 11 · Q-learning cho chính sách kinh tế thích nghi")
    policy_box("<b>Bối cảnh:</b> nền kinh tế = môi trường, chính sách = hành động, phần thưởng = phúc lợi xã "
               "hội. MDP 81 trạng thái (3⁴) × 5 hành động ngân sách. <i>Lưu ý: AI hỗ trợ chứ không thay thế "
               "trách nhiệm chính trị.</i>")
    st.latex(r"R_t = 0{,}40\,\Delta GDP - 0{,}25\,\Delta U - 0{,}20\,Cyber - 0{,}15\,Emission")

    st.sidebar.markdown("**⚙️ Tham số (Bài 11)**")
    episodes = st.sidebar.select_slider("Số episode huấn luyện", [2000, 4000, 8000], 4000)
    gamma = st.sidebar.slider("γ — discount", 0.80, 0.99, 0.95, 0.01)

    with st.spinner("Đang huấn luyện Q-learning..."):
        Q, curve = train_qlearning(episodes, 0.1, gamma)

    section("11.3.2 & 11.3.4 · Learning curve & so sánh chính sách")
    rq = eval_policy(Q, "q"); r1 = eval_policy(Q, "a1"); r3 = eval_policy(Q, "a3"); rr = eval_policy(Q, "rand")
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("π* (Q-learning)", f"{rq:.2f}", "reward TB/episode", "#2e7d32")
    with c2: card("Luôn a1 (cân bằng)", f"{r1:.2f}", "rule-based", INK)
    with c3: card("Luôn a3 (AI dẫn dắt)", f"{r3:.2f}", "rule-based", INK)
    with c4: card("Ngẫu nhiên", f"{rr:.2f}", "baseline", "#94a3b8")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = px.line(x=np.arange(len(curve)), y=curve,
                      labels={"x": "Episode", "y": "Reward (trượt 50)"})
        fig.update_traces(line_color=ACCENT, line_width=2)
        fig.update_layout(height=320, template="plotly_white", title="Learning curve Q-learning")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cmp = pd.DataFrame({"Chính sách": ["π*", "a1", "a3", "random"],
                            "Reward": [rq, r1, r3, rr]})
        fig = px.bar(cmp, x="Chính sách", y="Reward", color="Chính sách",
                     color_discrete_sequence=[ACCENT, "#1565c0", "#6a1b9a", "#94a3b8"])
        fig.update_layout(height=320, template="plotly_white", showlegend=False,
                          title="So sánh phần thưởng")
        st.plotly_chart(fig, use_container_width=True)

    section("11.3.3 · Chính sách tối ưu π*(s) tại các trạng thái tiêu biểu")
    states = {"VN 2026 (G=TB, D=TB, AI=thấp, U=TB)": (1, 1, 0, 1),
              "Suy giảm (G=thấp, D=thấp, U=cao)": (0, 0, 0, 2),
              "Bứt phá (G=cao, AI=cao, U=thấp)": (2, 2, 2, 0),
              "Quá nóng (G=cao, D=cao, U=thấp)": (2, 2, 1, 0)}
    rows = []
    for name, s in states.items():
        a = int(np.argmax(Q[tuple(s)]))
        rows.append({"Trạng thái": name, "π*(s)": ACT_NAME[a]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    policy_discussion(11)


# ============================================================================
# BÀI 12 — ĐỒ ÁN TÍCH HỢP AIDEOM-VN (6 module → 4 tab)
# ============================================================================
SCENARIOS = {
    "S1 Truyền thống": [0.70, 0.10, 0.10, 0.10],
    "S2 Số hoá nhanh": [0.25, 0.45, 0.15, 0.15],
    "S3 AI dẫn dắt":   [0.20, 0.20, 0.45, 0.15],
    "S4 Bao trùm số":  [0.30, 0.20, 0.10, 0.40],
    "S5 Tối ưu cân bằng": [0.35, 0.25, 0.20, 0.20],
}


def m1_forecast():
    """M1 — Dự báo kinh tế Cobb-Douglas 2026-2030."""
    K, L, D, AI, H, Y, A, _, _ = cobb_compute(0.33, 0.42, 0.10, 0.08, 0.07)
    a, b, g, d, th = 0.33, 0.42, 0.10, 0.08, 0.07
    yrs = list(range(2026, 2031))
    Kn, Ln, Dn, AIn, Hn, An = K[-1], L[-1], D[-1], AI[-1], H[-1], A[-1]
    out = []
    for _ in yrs:
        Kn *= 1.06; Ln *= 1.005; Dn = min(Dn + 2.1, 30); AIn += 4; Hn = min(Hn + 1.3, 35); An *= 1.012
        out.append(An * Kn**a * Ln**b * Dn**g * AIn**d * Hn**th)
    return yrs, np.array(out), Y[-1]


def m2_readiness():
    """M2 — TOPSIS sẵn sàng số 6 vùng."""
    w = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
    C = topsis(w)
    return pd.DataFrame({"region_name_vi": REGIONS.region_name_vi, "C*": C}).sort_values("C*", ascending=False)


def m345_scenario(profile, budget=50000):
    """M3/M4/M5 — KPI tổng hợp cho 1 kịch bản phân bổ [K,D,AI,H]."""
    sK, sD, sAI, sH = profile
    # M3: GDP gain xấp xỉ qua hệ số hiệu quả biên trung bình
    gdp_gain = budget * (0.85 * sK + 1.10 * sD + 1.25 * sAI + 0.95 * sH) / 1000  # ngh.tỷ
    # M4: NetJob (đơn giản hoá) — H & AI tạo việc, AI có rủi ro dịch chuyển
    netjob = budget * (35 * sH + 20 * sAI - 18 * sAI) / 100   # nghìn việc
    # M5: rủi ro tổng hợp (cyber theo AI, phát thải theo K, bất bình đẳng nếu thiếu H)
    risk = 100 * (0.5 * sAI + 0.4 * sK - 0.3 * sH + 0.2)
    digital = 100 * (0.6 * sD + 0.4 * sAI)
    inclusion = 100 * (0.7 * sH + 0.3 * sD)
    return dict(GDP_gain=gdp_gain, NetJob=netjob, Risk=max(risk, 0),
                Digital=digital, Inclusion=inclusion)


def render_bai12():
    st.title("🧠 Bài 12 · Đồ án tích hợp AIDEOM-VN")
    policy_box("<b>Đồ án tổng kết:</b> tích hợp <b>6 module</b> (M1 Dự báo · M2 Sẵn sàng số · M3 Phân bổ · "
               "M4 Lao động · M5 Rủi ro · M6 Dashboard) thành hệ hỗ trợ ra quyết định, so sánh "
               "<b>5 kịch bản chính sách</b> S1-S5. Tổ chức theo 4 tab dưới đây.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Tổng quan (M1+M2)", "🗺️ Phân bổ (M3)",
         "⚖️ So sánh kịch bản (M4)", "🚨 Cảnh báo rủi ro (M5+M6)"])

    # ---------- TAB 1: Tổng quan = M1 + M2 ----------
    with tab1:
        st.subheader("Module M1 — Dự báo kinh tế 2026-2030")
        yrs, Yf, Y0 = m1_forecast()
        c1, c2, c3 = st.columns(3)
        with c1: card("GDP 2030 dự báo (ngh.tỷ)", f"{Yf[-1]:,.0f}", f"CAGR {(np.log(Yf[-1]/Y0)/5)*100:.2f}%", "#2e7d32")
        with c2: card("Tăng so với 2025", f"+{(Yf[-1]/Y0-1)*100:.1f}%", "5 năm", INK)
        with c3: card("GDP 2030 (tỷ USD)", f"{Yf[-1]*1000/25.5:,.0f}", "~25.500 VND/USD", ACCENT)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[2025] + yrs, y=[Y0] + list(Yf), mode="lines+markers",
                      line=dict(color=ACCENT, width=3), name="GDP dự báo"))
        fig.update_layout(height=320, template="plotly_white", title="Quỹ đạo GDP 2025-2030")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Module M2 — Bản đồ mức sẵn sàng số (TOPSIS)")
        m2 = m2_readiness()
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.dataframe(m2.rename(columns={"region_name_vi": "Vùng"}).round(4),
                         use_container_width=True, hide_index=True)
        with c2:
            fig = px.bar(m2.sort_values("C*"), x="C*", y="region_name_vi", orientation="h",
                         color="C*", color_continuous_scale="Reds")
            fig.update_layout(height=320, template="plotly_white", yaxis_title="",
                              title="Xếp hạng sẵn sàng AI 6 vùng")
            st.plotly_chart(fig, use_container_width=True)

    # ---------- TAB 2: Phân bổ = M3 ----------
    with tab2:
        st.subheader("Module M3 — Tối ưu phân bổ ngân sách ngành-vùng")

        # Ràng buộc công bằng C5 chỉ khả thi tới một ngưỡng λ_max < 0,70 (xem Bài 4).
        # M3 tự xác định λ_max bằng bisection rồi phân bổ ngay tại ngưỡng đó để vừa
        # tối đa hoá công bằng vừa luôn cho ra một lời giải khả thi (không bao giờ lỗi).
        lo, hi = 0.0, 0.70
        if solve_bai4(True, 0.70).success:
            lam_max = 0.70
        else:
            for _ in range(40):
                mid = (lo + hi) / 2
                if solve_bai4(True, mid).success:
                    lo = mid
                else:
                    hi = mid
            lam_max = round(lo, 3)
        res_m3 = solve_bai4(True, lam_max)
        if not res_m3.success:                     # an toàn tuyệt đối
            res_m3, lam_max = solve_bai4(False), None

        X = res_m3.x[:24].reshape(6, 4)
        c1, c2 = st.columns([1, 1.2])
        with c1:
            card("Z* GDP gain", f"{-res_m3.fun:,.0f}", "tỷ VND", "#2e7d32")
            if lam_max is not None:
                card("Ngưỡng công bằng áp dụng", f"λ = {lam_max:.3f}",
                     "tối đa khả thi của C5", ACCENT)
            st.dataframe(pd.DataFrame(X, columns=["I", "D", "AI", "H"],
                         index=REGION_SHORT).style.format("{:,.0f}"),
                         use_container_width=True)
        with c2:
            fig = px.imshow(X, x=["I", "D", "AI", "H"], y=REGION_SHORT, text_auto=".0f",
                            color_continuous_scale="Reds", aspect="auto")
            fig.update_layout(height=360, template="plotly_white",
                              title=f"Heatmap phân bổ tối ưu — M3 (λ={lam_max:.3f})"
                              if lam_max is not None else "Heatmap phân bổ tối ưu — M3")
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"M3 phân bổ tại **λ = {lam_max:.3f}** — mức công bằng vùng cao nhất mà hệ ràng "
            f"buộc C1–C5 còn khả thi. Mục tiêu lý thuyết λ = 0,70 vượt giới hạn này nên không "
            f"tồn tại lời giải; đây là kết quả phân tích của Bài 4 chứ không phải lỗi mô hình. "
            f"Việc kéo λ về {lam_max:.3f} chính là biểu hiện định lượng của *cái giá của công "
            f"bằng* (cost of fairness) trong phân bổ ngân sách số."
            if lam_max is not None else
            "M3 hiển thị phân bổ tối đa hoá GDP gain (đã nới ràng buộc công bằng C5)."
        )

    # ---------- TAB 3: So sánh 5 kịch bản = M4 + KPI ----------
    with tab3:
        st.subheader("Module M4 — Mô phỏng lao động & So sánh 5 kịch bản chính sách")
        rows = []
        for name, prof in SCENARIOS.items():
            kpi = m345_scenario(prof)
            rows.append({"Kịch bản": name, "Cơ cấu (K/D/AI/H)": "/".join(f"{int(p*100)}" for p in prof),
                         **{k: round(v, 1) for k, v in kpi.items()}})
        dfk = pd.DataFrame(rows)
        st.dataframe(dfk, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(dfk, x="Kịch bản", y="GDP_gain", color="Kịch bản",
                         color_discrete_sequence=PALETTE, title="GDP gain theo kịch bản")
            fig.update_layout(height=320, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(dfk, x="Kịch bản", y="NetJob", color="Kịch bản",
                         color_discrete_sequence=PALETTE, title="NetJob ròng theo kịch bản")
            fig.update_layout(height=320, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        # radar so sánh đa chiều
        cats = ["GDP_gain", "Digital", "Inclusion", "NetJob"]
        fig = go.Figure()
        for i, (name, prof) in enumerate(SCENARIOS.items()):
            kpi = m345_scenario(prof)
            vals = [kpi["GDP_gain"], kpi["Digital"], kpi["Inclusion"], kpi["NetJob"] / 10]
            fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]],
                          name=name, line=dict(color=PALETTE[i])))
        fig.update_layout(height=420, template="plotly_white",
                          title="Radar đa chiều 5 kịch bản (NetJob/10 để cùng thang)")
        st.plotly_chart(fig, use_container_width=True)

    # ---------- TAB 4: Cảnh báo rủi ro = M5 + M6 ----------
    with tab4:
        st.subheader("Module M5 — Đánh giá rủi ro (Cyber · Môi trường · Bất bình đẳng)")
        risk_rows = []
        for name, prof in SCENARIOS.items():
            kpi = m345_scenario(prof)
            level = "🔴 Cao" if kpi["Risk"] > 50 else ("🟡 Trung bình" if kpi["Risk"] > 30 else "🟢 Thấp")
            risk_rows.append({"Kịch bản": name, "Chỉ số rủi ro": round(kpi["Risk"], 1),
                              "Bao trùm": round(kpi["Inclusion"], 1), "Cảnh báo": level})
        dfr = pd.DataFrame(risk_rows)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(dfr, use_container_width=True, hide_index=True)
        with c2:
            fig = px.scatter(dfr, x="Chỉ số rủi ro", y="Bao trùm", text="Kịch bản",
                             color="Chỉ số rủi ro", color_continuous_scale="RdYlGn_r", size_max=20)
            fig.update_traces(textposition="top center", marker_size=16)
            fig.update_layout(height=340, template="plotly_white", title="Rủi ro × Bao trùm")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Module M6 — Dashboard khuyến nghị chính sách")
        best = max(SCENARIOS, key=lambda n: m345_scenario(SCENARIOS[n])["GDP_gain"]
                   - 0.5 * m345_scenario(SCENARIOS[n])["Risk"] + m345_scenario(SCENARIOS[n])["Inclusion"])
        st.success(f"🏆 **Khuyến nghị AIDEOM-VN:** kịch bản **{best}** đạt cân bằng tốt nhất giữa "
                   "tăng trưởng GDP, mức rủi ro chấp nhận được và tính bao trùm xã hội.")
        policy_box("<b>Cảnh báo & khuyến nghị tự động:</b> "
                   "• S1 (Truyền thống) — rủi ro thấp nhưng động lực số yếu, khó đạt 30% KTS/GDP 2030. "
                   "• S3 (AI dẫn dắt) — GDP & số hoá cao nhưng rủi ro cyber/bất bình đẳng cao → cần đệm H. "
                   "• S5 (Tối ưu cân bằng) — bám sát Nghị quyết 57-NQ/TW: tăng trưởng đi cùng bao trùm & an toàn.")
        st.caption("Tiêu chí đánh giá đồ án (Phụ lục F2): mô hình 20% · mã nguồn 20% · dữ liệu VN 15% · "
                   "phân tích chính sách 20% · dashboard 15% · báo cáo 10%.")


# ============================================================================
# BỘ ĐỊNH TUYẾN
# ============================================================================
ROUTER = {
    PAGES[0]: render_home, PAGES[1]: render_bai1, PAGES[2]: render_bai2,
    PAGES[3]: render_bai3, PAGES[4]: render_bai4, PAGES[5]: render_bai5,
    PAGES[6]: render_bai6, PAGES[7]: render_bai7, PAGES[8]: render_bai8,
    PAGES[9]: render_bai9, PAGES[10]: render_bai10, PAGES[11]: render_bai11,
    PAGES[12]: render_bai12,
}
ROUTER.get(choice, render_home)()

st.divider()
st.caption("AIDEOM-VN · Bài tập lớn Các mô hình ra quyết định · Tạ Tuấn Chinh — 23051191 · "
           "Dữ liệu Việt Nam 2020-2025")
