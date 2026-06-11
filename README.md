# AIDEOM-VN: Hệ thống hỗ trợ ra quyết định kinh tế vĩ mô Việt Nam trong Kỷ nguyên AI

Hệ thống **AIDEOM-VN** là một dự án phân tích định lượng toàn diện cho nền kinh tế Việt Nam, tích hợp các mô hình toán học tiên tiến để tối ưu hóa nguồn lực, dự báo tăng trưởng vĩ mô, mô phỏng động lực học thị trường lao động và quản trị rủi ro chính sách trong kỷ nguyên trí tuệ nhân tạo (AI).

Dự án được thiết kế đáp ứng hoàn toàn các định hướng vĩ mô của Chính phủ Việt Nam:
*   **Quyết định 749/QĐ-TTg** (Chương trình Chuyển đổi số quốc gia đến năm 2025, định hướng đến năm 2030).
*   **Quyết định 127/QĐ-TTg** (Chiến lược quốc gia về nghiên cứu, phát triển và ứng dụng Trí tuệ nhân tạo đến năm 2030).
*   **Quyết định 964/QĐ-TTg** (Chiến lược An toàn, An ninh mạng quốc gia).
*   **Cam kết COP26** về trung hòa carbon và phát triển bền vững bao trùm.

---

## 📂 Cấu trúc thư mục dự án

```text
aideom_vn/
├── data/
│   ├── vietnam_macro_2020_2025.csv    # Dữ liệu kinh tế vĩ mô Việt Nam
│   ├── vietnam_priorities.csv          # Trọng số ưu tiên chính sách (MCDA)
│   ├── vietnam_regions_2024.csv        # Dữ liệu 6 vùng kinh tế - xã hội
│   └── vietnam_sectors_2024.csv        # Dữ liệu 10 ngành kinh tế chính
├── notebooks/                          # 11 tệp Jupyter Notebook tự chứa (chạy Colab/Kaggle)
│   ├── bai01_cobb_douglas.ipynb
│   ├── bai02_lp_budget.ipynb
│   ├── bai03_priority_index.ipynb
│   ├── bai04_lp_region_sector.ipynb
│   ├── bai05_mip_project.ipynb
│   ├── bai06_topsis.ipynb
│   ├── bai07_nsga2_pareto.ipynb
│   ├── bai08_dynamic_opt.ipynb
│   ├── bai09_labor_market.ipynb
│   ├── bai10_stochastic_lp.ipynb
│   └── bai11_qlearning.ipynb
├── src/                                # Thư viện mã nguồn lõi Python
│   ├── __init__.py
│   ├── data_loader.py                  # Module nạp và kiểm tra dữ liệu CSV
│   ├── m1_production.py                # Hàm Cobb-Douglas mở rộng, TFP, Scenario 2030
│   ├── m2_readiness.py                 # Thuật toán TOPSIS & Entropy weights
│   ├── m3_allocation.py                # Tối ưu hóa phân bổ ngân sách (PuLP, SciPy, CVXPY)
│   ├── m4_labor.py                     # Mô phỏng thị trường lao động & NetJob
│   ├── m5_risk.py                      # Quy hoạch ngẫu nhiên 2 giai đoạn & Minimax Regret
│   ├── m6_dashboard.py                 # Module layout giao diện Streamlit Dashboard
│   └── rl_env.py                       # Môi trường kinh tế Gymnasium & huấn luyện Q-learning
├── tests/
│   └── test_all_modules.py             # Bộ kiểm thử tự động hệ thống (pytest)
├── app.py                              # Ứng dụng Streamlit Dashboard chính
├── requirements.txt                    # Danh sách các thư viện phụ thuộc
└── README.md                           # Tài liệu hướng dẫn sử dụng (tệp này)
```

---

## 🚀 Hướng dẫn cài đặt & Khởi động nhanh (Local)

Để thiết lập môi trường chạy thử nghiệm hệ thống tại máy tính cá nhân của bạn:

1.  **Cài đặt các thư viện phụ thuộc:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Chạy kiểm thử tự động bằng Pytest:**
    Để đảm bảo toàn bộ 10 ca kiểm thử định lượng hệ thống hoạt động chính xác:
    ```bash
    # Trên Windows PowerShell
    $env:PYTHONPATH="."; pytest
    ```

3.  **Khởi động Streamlit Dashboard:**
    ```bash
    streamlit run app.py
    ```

---

## 📓 Hướng dẫn chạy trên Google Colab hoặc Kaggle

Toàn bộ **11 tệp Jupyter Notebook** trong thư mục `notebooks/` được thiết kế theo tiêu chuẩn **Tự chứa hoàn toàn (Self-contained)**. Người dùng có thể dễ dàng tải lên Google Colab hoặc Kaggle để chạy trực tiếp mà không cần cấu hình trước môi trường trên máy tính cá nhân:

1.  Mở Google Colab hoặc Kaggle Notebook.
2.  Tải lên (Upload) tệp notebook mà bạn muốn học (ví dụ: `bai01_cobb_douglas.ipynb` hoặc `bai11_qlearning.ipynb`).
3.  **Bấm chạy ô đầu tiên (Cell #1 - "Setup Environment"):**
    *   Ô này sẽ tự động cài đặt tất cả thư viện cần thiết như `pulp`, `cvxpy`, `gymnasium`, v.v.
    *   Tự động giải nén và tạo đúng cấu trúc thư mục dữ liệu `aideom_vn/data` và mã nguồn lõi `aideom_vn/src` trực tiếp vào không gian làm việc đám mây của Colab/Kaggle.
    *   Giúp tất cả các câu lệnh import như `from aideom_vn.src.m1_production import compute_tfp` hoạt động trơn tru 100% không gặp lỗi thiếu tệp.
4.  Tiến hành thực thi các ô phân tích, mô phỏng đồ thị trực quan và đọc các phân tích thảo luận chính sách chi tiết bằng tiếng Việt đi kèm trong từng tệp.

---

## 🛠️ Mô tả các mô hình quyết định tích hợp

1.  **Bài 1 (Cobb-Douglas mở rộng):** Đo lường đóng góp tăng trưởng của Vốn (K), Lao động (L), Công nghệ Số (D), Doanh nghiệp AI và Nhân lực qua đào tạo (H) đối với TFP và dự báo GDP 2030 theo 3 kịch bản chính sách.
2.  **Bài 2 (LP Budget Allocation):** Giải bài toán phân bổ tối ưu ngân sách công nghệ sử dụng solver CBC mạnh mẽ, trích xuất giá đối ngẫu (Shadow Prices) để đo lường lợi ích biên vĩ mô.
3.  **Bài 3 & 6 (TOPSIS & Entropy Weights):** Đánh giá đa tiêu chí khách quan năng lực chuyển đổi số vùng miền, phân tích độ nhạy của biến AI nhằm đưa ra các lộ trình giảm thiểu khoảng cách số.
4.  **Bài 4 (LP Regional Equity):** Phân bổ ngân sách liên vùng có ràng buộc công bằng phi tuyến (được tuyến tính hóa) kiểm soát chênh lệch Gini của chỉ số phát triển số hóa giữa các cực tăng trưởng.
5.  **Bài 5 (MIP Project Portfolio):** Tuyển chọn dự án chuyển đổi số công nghệ cao sử dụng quy hoạch nguyên hỗn hợp nhị phân (MIP) dưới các ràng buộc loại trừ lẫn nhau, tiên quyết đầu tư nhân lực và rủi ro thất bại thực tế.
6.  **Bài 7 (Biên tối ưu Pareto):** Khảo sát đa mục tiêu phân bổ ngân sách dung hòa giữa Tăng trưởng kinh tế tối đa và Đảm bảo an sinh xã hội/Bình đẳng vùng miền, xác định điểm thỏa hiệp tối ưu Nash (Nash Bargaining).
7.  **Bài 8 (Dynamic Programming):** Quy hoạch động tối ưu hóa quỹ đạo tích lũy vốn vật lý và công nghệ số trong dải thời gian 10 năm (2026-2035) dưới ảnh hưởng chênh lệch khấu hao kinh tế.
8.  **Bài 9 (Mô phỏng Lao động):** Dự báo cung cầu việc làm số, sa thải tự động hóa, tối ưu hóa ngân sách đào tạo lại (retraining) vĩ mô và xác định ngưỡng đầu tư tối thiểu để giữ việc làm ròng không bị âm cho ngành Chế tạo.
9.  **Bài 10 (Stochastic Two-stage LP):** Ra quyết định đầu tư dưới sự không chắc chắn của thị trường toàn cầu thông qua tối ưu hóa ngẫu nhiên hai giai đoạn có quyết định bù đắp (recourse), tính các trị số EVPI, VSS và giải bài toán vững chắc cực tiểu hóa hối tiếc cực đại (Minimax Regret).
10. **Bài 11 (Gymnasium Reinforcement Learning):** Huấn luyện trí tuệ nhân tạo (Agent) bằng thuật toán Q-learning trên môi trường Gymnasium mô phỏng động lực học kinh tế vĩ mô Việt Nam để tìm kiếm chính sách điều hành tối ưu bền vững dài hạn hướng tới tầm nhìn 2045.
