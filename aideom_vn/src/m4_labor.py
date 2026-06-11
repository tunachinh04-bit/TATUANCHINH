import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

class LaborMarketModel:
    """
    Lớp định nghĩa mô hình thị trường lao động Việt Nam dưới tác động của AI & Số hóa.
    Bao gồm 8 ngành kinh tế chính với các tham số tương ứng.
    """
    def __init__(self):
        self.sectors = {
            1: "Nông-Lâm-Thủy sản",
            2: "CN chế biến chế tạo",
            3: "Xây dựng",
            4: "Bán buôn-bán lẻ",
            5: "Tài chính-Ngân hàng",
            6: "Logistics-Vận tải",
            7: "CNTT-Truyền thông",
            8: "Giáo dục-Đào tạo"
        }
        
        # Số lao động của từng ngành (triệu người)
        self.L = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
        
        # Tỷ lệ rủi ro tự động hóa (%) của từng ngành
        self.risk = np.array([18.0, 42.0, 25.0, 38.0, 52.0, 35.0, 28.0, 22.0]) / 100.0
        
        # Hệ số tạo việc làm mới, nâng cấp, dịch chuyển và đào tạo lại (số việc / tỷ VND đầu tư)
        self.a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
        self.b1 = np.array([45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0])
        self.c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
        self.d1 = np.array([50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0])

def solve_labor_optimization(budget: float = 30000.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Tối đa hóa tổng NetJob ròng trên toàn bộ 8 ngành dưới ngân sách đào tạo & AI cho trước.
    
    Args:
        budget: Tổng ngân sách phân bổ cho cả 8 ngành (tỷ VND).
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            - x_AI_opt: Phân bổ đầu tư AI tối ưu từng ngành.
            - x_H_opt: Phân bổ đầu tư đào tạo lại H tối ưu từng ngành.
            - net_jobs_by_sector: Số việc làm ròng tăng thêm của mỗi ngành.
            - max_net_jobs: Tổng số việc làm ròng cực đại.
    """
    model = LaborMarketModel()
    N = len(model.sectors)
    
    import pulp
    prob = pulp.LpProblem("VN_Labor_Optimization", pulp.LpMaximize)
    
    # Biến quyết định
    x_AI = pulp.LpVariable.dicts("x_AI", range(N), lowBound=0)
    x_H = pulp.LpVariable.dicts("x_H", range(N), lowBound=0)
    
    # Định nghĩa các biểu thức trung gian cho mỗi ngành i
    new_jobs = [model.a1[i] * x_AI[i] for i in range(N)]
    upgrade_jobs = [model.b1[i] * x_H[i] for i in range(N)]
    displaced_jobs = [model.c1[i] * model.risk[i] * x_AI[i] for i in range(N)]
    retrain_caps = [model.d1[i] * x_H[i] for i in range(N)]
    
    # Hàm mục tiêu: max sum(NetJob_i)
    prob += pulp.lpSum(new_jobs[i] + upgrade_jobs[i] - displaced_jobs[i] for i in range(N))
    
    # Ràng buộc
    # 1. Ràng buộc ngân sách tổng
    prob += pulp.lpSum(x_AI[i] + x_H[i] for i in range(N)) <= budget, "Total_Budget"
    
    # 2. NetJob_i >= 0 và Displaced_i <= RetrainCap_i cho mỗi ngành
    for i in range(N):
        prob += new_jobs[i] + upgrade_jobs[i] - displaced_jobs[i] >= 0, f"NetJob_NonNegative_Sector_{i+1}"
        prob += displaced_jobs[i] <= retrain_caps[i], f"Retrain_Capacity_Sector_{i+1}"
        
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if prob.status != pulp.LpStatusOptimal:
        raise ValueError("Không giải được bài toán tối ưu hóa thị trường lao động.")
        
    x_AI_opt = np.array([x_AI[i].varValue for i in range(N)])
    x_H_opt = np.array([x_H[i].varValue for i in range(N)])
    
    # Tính toán số việc làm ròng cụ thể của mỗi ngành
    net_jobs = np.array([
        (model.a1[i] * x_AI_opt[i]) + (model.b1[i] * x_H_opt[i]) - (model.c1[i] * model.risk[i] * x_AI_opt[i])
        for i in range(N)
    ])
    
    return x_AI_opt, x_H_opt, net_jobs, pulp.value(prob.objective)

def minimum_training_threshold(sector_id: int = 2) -> float:
    """
    Tính toán ngưỡng đầu tư đào tạo lại tối thiểu (x_H) cho ngành Chế biến Chế tạo (sector_id = 2)
    để đảm bảo NetJob >= 0 ngay cả khi phân bổ đầu tư AI tối đa.
    Nguyên tắc: Tỷ lệ đầu tư đào tạo so với đầu tư AI phải đủ lớn để bù đắp rủi ro mất việc.
    
    Args:
        sector_id: ID của ngành cần tính toán (1-indexed).
        
    Returns:
        float: Tỷ trọng đầu tư x_H cần thiết tối thiểu trên mỗi đồng đầu tư x_AI (tỷ lệ x_H / x_AI).
    """
    model = LaborMarketModel()
    idx = sector_id - 1 # Chuyển về 0-indexed
    
    a1_i = model.a1[idx]
    b1_i = model.b1[idx]
    c1_i = model.c1[idx]
    risk_i = model.risk[idx]
    
    # NetJob_i = a1_i * x_AI_i + b1_i * x_H_i - c1_i * risk_i * x_AI_i >= 0
    # -> x_H_i >= ((c1_i * risk_i - a1_i) / b1_i) * x_AI_i
    val = (c1_i * risk_i - a1_i) / b1_i
    return max(0.0, float(val))

def scenario_displacement_cap(
    budget: float = 30000.0,
    max_displacement_pct: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """
    Mô phỏng thêm kịch bản ràng buộc bảo an xã hội: số lao động mất việc ở mỗi ngành
    không vượt quá 5% tổng số lao động ban đầu của ngành đó (DisplacedJob_i <= 0.05 * L_i).
    
    Args:
        budget: Tổng ngân sách.
        max_displacement_pct: Tỷ lệ mất việc tối đa cho phép của ngành (ví dụ 0.05 = 5%).
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
            - x_AI_opt, x_H_opt: Phương án đầu tư.
            - net_jobs: Số việc làm ròng từng ngành.
            - max_net_jobs: Tổng số việc làm ròng.
            - feasible: Trạng thái tìm được nghiệm khả thi (True/False).
    """
    model = LaborMarketModel()
    N = len(model.sectors)
    
    import pulp
    prob = pulp.LpProblem("VN_Labor_Displacement_Cap", pulp.LpMaximize)
    
    x_AI = pulp.LpVariable.dicts("x_AI", range(N), lowBound=0)
    x_H = pulp.LpVariable.dicts("x_H", range(N), lowBound=0)
    
    # Định nghĩa các biểu thức trung gian
    new_jobs = [model.a1[i] * x_AI[i] for i in range(N)]
    upgrade_jobs = [model.b1[i] * x_H[i] for i in range(N)]
    displaced_jobs = [model.c1[i] * model.risk[i] * x_AI[i] for i in range(N)]
    retrain_caps = [model.d1[i] * x_H[i] for i in range(N)]
    
    # Hàm mục tiêu
    prob += pulp.lpSum(new_jobs[i] + upgrade_jobs[i] - displaced_jobs[i] for i in range(N))
    
    # Ràng buộc
    prob += pulp.lpSum(x_AI[i] + x_H[i] for i in range(N)) <= budget
    
    for i in range(N):
        prob += new_jobs[i] + upgrade_jobs[i] - displaced_jobs[i] >= 0
        prob += displaced_jobs[i] <= retrain_caps[i]
        # Thêm ràng buộc bảo an xã hội (Đổi đơn vị triệu người sang số người nếu cần,
        # nhưng ở đây cả lao động và hệ số việc làm đều đồng bộ ở quy mô triệu người / việc làm)
        # Vì L_i đo bằng triệu người, còn x_AI đo bằng tỷ đồng, hệ số việc làm đo bằng việc/tỷ đồng.
        # Lưu ý: L_i triệu người = L_i * 1e6 người. Việc làm tạo ra = việc/tỷ VND * x_AI tỷ = số việc làm.
        # Do đó, DisplacedJob_i (số việc làm bị thay thế) <= max_displacement_pct * (L_i * 1,000,000)
        prob += displaced_jobs[i] <= max_displacement_pct * (model.L[i] * 1000000.0)
        
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if prob.status != pulp.LpStatusOptimal:
        return np.zeros(N), np.zeros(N), np.zeros(N), 0.0, False
        
    x_AI_opt = np.array([x_AI[i].varValue for i in range(N)])
    x_H_opt = np.array([x_H[i].varValue for i in range(N)])
    
    net_jobs = np.array([
        (model.a1[i] * x_AI_opt[i]) + (model.b1[i] * x_H_opt[i]) - (model.c1[i] * model.risk[i] * x_AI_opt[i])
        for i in range(N)
    ])
    
    return x_AI_opt, x_H_opt, net_jobs, pulp.value(prob.objective), True

def get_sankey_data(x_AI_opt: np.ndarray, x_H_opt: np.ndarray) -> Dict[str, Any]:
    """
    Tạo dữ liệu cấu trúc cho biểu đồ Sankey thể hiện luồng chuyển dịch lao động
    của các ngành dễ bị tổn thương (1 = Nông-Lâm-Thủy sản, 3 = Xây dựng, 4 = Bán buôn-bán lẻ).
    
    Returns:
        Dict[str, Any]: Từ điển chứa các danh sách labels, sources, targets, values cho Sankey.
    """
    model = LaborMarketModel()
    vulnerable_indices = [0, 2, 3] # Ngành 1, 3, 4 (0-indexed)
    
    labels = []
    sources = []
    targets = []
    values = []
    
    # Các nút:
    # 0, 1, 2: Lao động bị ảnh hưởng bởi tự động hóa (Ngành 1, 3, 4)
    # 3, 4, 5: Lao động được đào tạo lại thành công (Upgrade/Retrained)
    # 6: Lao động thất nghiệp lâm thời (Unemployed)
    # 7: Thị trường việc làm mới từ AI (New Jobs)
    
    node_labels = [
        "LĐ bị thay thế (Nông-Lâm-Thủy sản)",
        "LĐ bị thay thế (Xây dựng)",
        "LĐ bị thay thế (Bán buôn-bán lẻ)",
        "LĐ đào tạo lại (Nông nghiệp)",
        "LĐ đào tạo lại (Xây dựng)",
        "LĐ đào tạo lại (Bán buôn-bán lẻ)",
        "Lao động Thất nghiệp",
        "Việc làm mới phát sinh từ AI"
    ]
    
    for local_idx, idx in enumerate(vulnerable_indices):
        # Tính toán chi tiết lao động của ngành idx
        displaced = model.c1[idx] * model.risk[idx] * x_AI_opt[idx]
        retrain_cap = model.d1[idx] * x_H_opt[idx]
        
        # Lao động đào tạo thành công tối đa bằng năng lực đào tạo hoặc phần bị thay thế
        retrained = min(displaced, retrain_cap)
        unemployed = max(0.0, displaced - retrained)
        
        # Việc làm mới từ AI trong ngành
        new_job_ai = model.a1[idx] * x_AI_opt[idx]
        
        # Thêm luồng từ "LĐ bị thay thế" -> "LĐ đào tạo lại"
        if retrained > 0:
            sources.append(local_idx)
            targets.append(3 + local_idx)
            values.append(retrained)
            
        # Thêm luồng từ "LĐ bị thay thế" -> "Lao động Thất nghiệp"
        if unemployed > 0:
            sources.append(local_idx)
            targets.append(6)
            values.append(unemployed)
            
        # Thêm luồng từ "LĐ đào tạo lại" -> "Việc làm mới phát sinh từ AI"
        if retrained > 0:
            sources.append(3 + local_idx)
            targets.append(7)
            values.append(retrained)
            
    return {
        'labels': node_labels,
        'sources': sources,
        'targets': targets,
        'values': values
    }

# ==============================================================================
# BỔ SUNG CÁC HÀM CHO 10 NGÀNH KINH TẾ (Notebooks, Tests & Dashboard)
# ==============================================================================

def simulate_labor_dynamics(df_sectors: pd.DataFrame, x_inv: np.ndarray) -> pd.DataFrame:
    """
    Mô phỏng luồng biến động lao động của 10 ngành kinh tế dưới tác động của đầu tư AI/Số hóa.
    
    Args:
        df_sectors: DataFrame chứa thông tin 10 ngành từ load_sectors().
        x_inv: Mức đầu tư phân bổ cho mỗi ngành (billion VND).
        
    Returns:
        pd.DataFrame: DataFrame bổ sung các cột jobs_created_thousand, jobs_displaced_thousand, net_jobs_thousand.
    """
    df = df_sectors.copy()
    n_sectors = len(df)
    
    # Đảm bảo x_inv có cùng chiều dài với số ngành
    if len(x_inv) < n_sectors:
        x_inv_full = np.zeros(n_sectors)
        x_inv_full[:len(x_inv)] = x_inv
        x_inv = x_inv_full
        
    jobs_created = []
    jobs_displaced = []
    
    for i in range(n_sectors):
        row = df.iloc[i]
        x_i = x_inv[i]
        
        # Hệ số tạo việc làm mới: phụ thuộc vào ai_readiness
        ar_i = row.get('ai_readiness_0_100', 50.0)
        c_create = 0.12 * (ar_i / 100.0)
        jobs_c = x_i * c_create
        
        # Hệ số sa thải/thay thế: phụ thuộc vào automation_risk và labor_million
        risk_i = row.get('automation_risk_pct', 30.0)
        l_i = row.get('labor_million', 2.0)
        c_displace = 0.005 * (risk_i / 100.0) * l_i
        jobs_d = x_i * c_displace
        
        jobs_created.append(jobs_c)
        jobs_displaced.append(jobs_d)
        
    df['jobs_created_thousand'] = jobs_created
    df['jobs_displaced_thousand'] = jobs_displaced
    df['net_jobs_thousand'] = df['jobs_created_thousand'] - df['jobs_displaced_thousand']
    
    return df

def solve_netjob_maximization(budget_total: float = 60000.0) -> Tuple[np.ndarray, float]:
    """
    Tối đa hóa tổng tạo việc làm ròng toàn nền kinh tế (cho 10 ngành) dưới ràng buộc ngân sách.
    
    Args:
        budget_total: Tổng ngân sách đầu tư công nghệ & lao động (tỷ VND).
        
    Returns:
        Tuple[np.ndarray, float]:
            - x_opt: Phân bổ đầu tư tối ưu cho 10 ngành kinh tế.
            - Z_opt: Tổng việc làm ròng cực đại (nghìn người).
    """
    from aideom_vn.src.data_loader import load_sectors
    df_sectors = load_sectors()
    n_sectors = len(df_sectors)
    
    import pulp
    prob = pulp.LpProblem("NetJob_Maximization", pulp.LpMaximize)
    
    # Biến quyết định: Phân bổ ngân sách cho từng ngành
    x = [pulp.LpVariable(f'x_netjob_{i}', lowBound=1000.0, upBound=20000.0) for i in range(n_sectors)]
    
    # Hàm mục tiêu: Max maximize sum(NetJob_i)
    net_jobs = []
    for i in range(n_sectors):
        row = df_sectors.iloc[i]
        ar_i = row.get('ai_readiness_0_100', 50.0)
        risk_i = row.get('automation_risk_pct', 30.0)
        l_i = row.get('labor_million', 2.0)
        
        c_create = 0.12 * (ar_i / 100.0)
        c_displace = 0.005 * (risk_i / 100.0) * l_i
        
        net_jobs.append(x[i] * (c_create - c_displace))
        
    prob += pulp.lpSum(net_jobs)
    
    # Ràng buộc ngân sách tổng
    prob += pulp.lpSum(x) == budget_total, "Budget_Constraint"
    
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if prob.status != pulp.LpStatusOptimal:
        # Fallback chia đều ngân sách
        x_opt = np.ones(n_sectors) * (budget_total / n_sectors)
        Z_opt = 0.0
        return x_opt, Z_opt
        
    x_opt = np.array([x[i].varValue for i in range(n_sectors)])
    Z_opt = pulp.value(prob.objective)
    
    return x_opt, Z_opt

def get_minimum_retraining_threshold(
    x_inv: float,
    alpha_J: float = 0.12,
    beta_J: float = 0.08,
    k_displace: float = 0.005,
    omega: float = 0.003,
    c_re: float = 1.5
) -> float:
    """
    Tính toán ngân sách đào tạo lại tối thiểu (x_H) cần thiết cho ngành Chế biến Chế tạo (sector_id = 2)
    để đảm bảo số việc làm ròng không bị âm.
    """
    from aideom_vn.src.data_loader import load_sectors
    df_sectors = load_sectors()
    mfg = df_sectors[df_sectors['sector_id'] == 2].iloc[0]
    R_i = mfg['automation_risk_pct']
    L_i = mfg['labor_million']
    
    # Số việc làm bị thay thế
    jobs_displaced = x_inv * (k_displace * R_i + omega * L_i) / c_re
    # Số việc làm mới tạo ra
    jobs_created = x_inv * alpha_J
    
    # net_jobs = jobs_created + beta_J * x_H - jobs_displaced >= 0
    # => beta_J * x_H >= jobs_displaced - jobs_created
    min_x_H = (jobs_displaced - jobs_created) / beta_J
    return max(0.0, float(min_x_H))


def simulate_labor_market(sectors_df=None, allocation_dict=None):
    """
    Wrapper function cho dashboard tích hợp — mô phỏng tác động thị trường lao động.

    Sử dụng LaborMarketModel với tham số thực tế của 8 ngành để tính toán
    số việc làm ròng tạo ra/mất đi một cách xác định (không có nhiễu ngẫu nhiên),
    dựa trên cơ cấu phân bổ ngân sách AI và Nhân lực (H) cho từng kịch bản.

    Công thức: NetJob_i = a1_i·x_AI_i + b1_i·x_H_i - c1_i·risk_i·x_AI_i
    Trong đó x_AI_i và x_H_i là phân bổ cho mỗi ngành theo tỷ trọng lao động.

    Args:
        sectors_df: DataFrame với thông tin ngành (tùy chọn, để lấy tên ngành).
        allocation_dict: Dict phân bổ ngân sách. Chấp nhận 2 dạng:
            - Tỷ lệ (ratio): {'K': 0.70, 'D': 0.10, 'AI': 0.10, 'H': 0.10}  — tổng = 1.0
            - Giá trị tuyệt đối (tỷ VND): {'K': 56000, 'D': 8000, 'AI': 8000, 'H': 8000}

    Returns:
        dict: {'net_job': np.ndarray (8 ngành, đơn vị: số việc làm), 'sectors': list tên ngành}
    """
    # Tổng ngân sách tham chiếu (tỷ VND) cho toàn bộ giai đoạn 2026-2030
    REFERENCE_BUDGET_TY = 80_000.0   # 80.000 tỷ VND

    try:
        model = LaborMarketModel()
        N = len(model.sectors)

        # Lấy tên ngành từ model nếu không có input
        if sectors_df is None:
            sectors_list = list(model.sectors.values())
        else:
            sectors_list = sectors_df['sector_name_vi'].tolist()[:N]

        # Giá trị phân bổ mặc định nếu không có input
        if allocation_dict is None:
            allocation_dict = {'K': 0.35, 'D': 0.25, 'AI': 0.20, 'H': 0.20}

        if isinstance(allocation_dict, dict):
            ai_alloc = float(allocation_dict.get('AI', 0.20))
            h_alloc = float(allocation_dict.get('H', 0.20))
            total_alloc = sum(float(allocation_dict.get(k, 0.0)) for k in ['K', 'D', 'AI', 'H'])

            if 0.0 < total_alloc <= 1.0:
                x_AI = np.full(N, (ai_alloc * REFERENCE_BUDGET_TY) / N)
                x_H = np.full(N, (h_alloc * REFERENCE_BUDGET_TY) / N)
            else:
                x_AI = np.full(N, ai_alloc / N)
                x_H = np.full(N, h_alloc / N)
        else:
            x_AI = np.full(N, 0.20 * REFERENCE_BUDGET_TY / N)
            x_H = np.full(N, 0.20 * REFERENCE_BUDGET_TY / N)

        # Tính số việc làm ròng (đơn vị: số việc làm)
        net_jobs = np.array([
            model.a1[i] * x_AI[i]
            + model.b1[i] * x_H[i]
            - model.c1[i] * model.risk[i] * x_AI[i]
            for i in range(N)
        ])

        return {
            'net_job': net_jobs,
            'sectors': sectors_list
        }

    except Exception as e:
        print(f"Warning in simulate_labor_market: {e}")
        # Fallback xác định (không dùng random) — tính trên cơ sở tham số mặc định
        model_fb = LaborMarketModel()
        x_fb = np.array([375.0] * 8)   # 375 tỷ VND / ngành (≈ 3000 tỷ / 8)
        nj_fb = np.array([
            model_fb.a1[i] * x_fb[i]
            + model_fb.b1[i] * x_fb[i] * 0.5
            - model_fb.c1[i] * model_fb.risk[i] * x_fb[i]
            for i in range(8)
        ])
        return {
            'net_job': nj_fb,
            'sectors': list(model_fb.sectors.values())
        }
