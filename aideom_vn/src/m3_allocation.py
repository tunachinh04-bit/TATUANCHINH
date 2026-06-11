import numpy as np
import scipy.optimize as opt
from typing import Tuple, Dict, Any, List, Optional

def solve_lp_scipy(budget_total: float = 100.0) -> Tuple[np.ndarray, float]:
    """
    Giải bài toán phân bổ ngân sách đơn giản (Bài 2) bằng scipy.optimize.linprog.
    
    Args:
        budget_total: Tổng ngân sách đầu tư (nghìn tỷ VND).
        
    Returns:
        Tuple[np.ndarray, float]:
            - x_opt: Vector đầu tư tối ưu cho 4 hạng mục (I, AI, H, R&D).
            - Z_opt: GDP tăng thêm cực đại (nghìn tỷ VND).
    """
    # Hàm mục tiêu: max Z = 0.85x1 + 1.20x2 + 0.95x3 + 1.35x4
    # scipy tối thiểu hóa, nên ta đảo dấu hệ số
    c = [-0.85, -1.20, -0.95, -1.35]
    
    # Ràng buộc bất đẳng thức A_ub * x <= b_ub
    # x1 + x2 + x3 + x4 <= budget_total
    # -x1 <= -25 (x1 >= 25)
    # -x2 <= -15 (x2 >= 15)
    # -x3 <= -20 (x3 >= 20)
    # -x4 <= -10 (x4 >= 10)
    # x2 + x4 >= 0.35(x1+x2+x3+x4) -> 0.35x1 - 0.65x2 + 0.35x3 - 0.65x4 <= 0
    A_ub = [
        [1.0, 1.0, 1.0, 1.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.35, -0.65, 0.35, -0.65]
    ]
    b_ub = [budget_total, -25.0, -15.0, -20.0, -10.0, 0.0]
    
    bounds = [(0, None)] * 4
    
    # Giải bằng phương pháp HiGHS hiệu suất cao
    res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        raise ValueError(f"Không tìm thấy lời giải khả thi cho LP SciPy: {res.message}")
        
    return res.x, -res.fun

def solve_lp_pulp(budget_total: float = 100.0) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """
    Giải bài toán phân bổ ngân sách đơn giản (Bài 2) bằng PuLP và in ra giá đối ngẫu.
    
    Args:
        budget_total: Tổng ngân sách đầu tư.
        
    Returns:
        Tuple[np.ndarray, float, Dict[str, float]]:
            - x_opt: Vector đầu tư tối ưu.
            - Z_opt: GDP tăng cực đại.
            - shadow_prices: Từ điển chứa giá đối ngẫu (shadow prices) của từng ràng buộc.
    """
    import pulp
    # Khởi tạo bài toán tối đa hóa
    prob = pulp.LpProblem("Simple_Budget_Allocation", pulp.LpMaximize)
    
    # Biến quyết định
    x1 = pulp.LpVariable('x1_Infra', lowBound=0)
    x2 = pulp.LpVariable('x2_AI', lowBound=0)
    x3 = pulp.LpVariable('x3_Human', lowBound=0)
    x4 = pulp.LpVariable('x4_RD', lowBound=0)
    
    # Hàm mục tiêu
    prob += 0.85*x1 + 1.20*x2 + 0.95*x3 + 1.35*x4, "Total_GDP_Gain"
    
    # Ràng buộc
    c_budget = x1 + x2 + x3 + x4 <= budget_total
    prob += c_budget, "Budget_Constraint"
    
    c_infra = x1 >= 25
    prob += c_infra, "Min_Infra"
    
    c_ai = x2 >= 15
    prob += c_ai, "Min_AI"
    
    c_human = x3 >= 20
    prob += c_human, "Min_Human"
    
    c_rd = x4 >= 10
    prob += c_rd, "Min_RD"
    
    c_strategic = x2 + x4 >= 0.35 * (x1 + x2 + x3 + x4)
    prob += c_strategic, "Strategic_Tech_Ratio"
    
    # Giải bài toán bằng solver CBC mặc định của PuLP (đã đi kèm)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if prob.status != pulp.LpStatusOptimal:
        raise ValueError(f"Không giải được LP tối ưu bằng PuLP: Status = {pulp.LpStatus[prob.status]}")
        
    x_opt = np.array([x1.varValue, x2.varValue, x3.varValue, x4.varValue])
    Z_opt = pulp.value(prob.objective)
    
    # Trích xuất giá đối ngẫu (shadow prices) của các ràng buộc
    shadow_prices = {
        'budget': c_budget.pi,
        'min_infra': c_infra.pi,
        'min_ai': c_ai.pi,
        'min_human': c_human.pi,
        'min_rd': c_rd.pi,
        'strategic': c_strategic.pi
    }
    
    return x_opt, Z_opt, shadow_prices

def solve_allocation_pulp(
    budget: float = 50000.0,
    gamma: float = 0.002,
    lam: float = 0.7,
    with_fairness: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Giải bài toán phân bổ ngân sách 6 vùng x 4 hạng mục (Bài 4) bằng PuLP.
    
    Args:
        budget: Tổng ngân sách đầu tư vùng miền.
        gamma: Hệ số cải thiện số hóa trên mỗi tỷ đầu tư.
        lam: Tham số mức độ công bằng (lambda in [0, 1]).
        with_fairness: Nếu True, áp dụng ràng buộc công bằng vùng miền C5.
        
    Returns:
        Tuple[np.ndarray, float]:
            - x_matrix: Ma trận đầu tư tối ưu (6 vùng x 4 hạng mục).
            - Z_opt: GDP tăng cực đại vùng miền (tỷ VND).
    """
    regions = ['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD']
    items = ['I', 'D', 'AI', 'H']
    
    # Bảng hệ số tác động biên beta
    beta = {
        ('NMM','I'):1.15,('NMM','D'):0.85,('NMM','AI'):0.55,('NMM','H'):1.30,
        ('RRD','I'):0.95,('RRD','D'):1.25,('RRD','AI'):1.40,('RRD','H'):1.05,
        ('NCC','I'):1.05,('NCC','D'):0.95,('NCC','AI'):0.85,('NCC','H'):1.15,
        ('CH','I') :1.20,('CH','D') :0.75,('CH','AI') :0.45,('CH','H') :1.35,
        ('SE','I') :0.90,('SE','D') :1.30,('SE','AI') :1.55,('SE','H') :1.00,
        ('MD','I') :1.10,('MD','D') :0.85,('MD','AI') :0.65,('MD','H') :1.25
    }
    
    # Chỉ số số hóa ban đầu D_0
    D0 = {'NMM':38.0, 'RRD':78.0, 'NCC':55.0, 'CH':32.0, 'SE':82.0, 'MD':48.0}
    
    import pulp
    prob = pulp.LpProblem("VN_Region_Budget", pulp.LpMaximize)
    
    # Biến quyết định
    x = pulp.LpVariable.dicts('x', (regions, items), lowBound=0)
    
    # Hàm mục tiêu
    prob += pulp.lpSum(beta[(r,j)] * x[r][j] for r in regions for j in items)
    
    # Ràng buộc
    # C1: Ngân sách tổng
    prob += pulp.lpSum(x[r][j] for r in regions for j in items) <= budget
    
    # C2 & C3: Ngưỡng sàn/trần của từng vùng miền
    for r in regions:
        prob += pulp.lpSum(x[r][j] for j in items) >= 5000
        prob += pulp.lpSum(x[r][j] for j in items) <= 12000
        
    # C4: Sàn đào tạo nhân lực số tổng thể
    prob += pulp.lpSum(x[r]['H'] for r in regions) >= 12000
    
    # C5: Ràng buộc công bằng vùng (Linearized bằng biến phụ M)
    if with_fairness:
        M = pulp.LpVariable('Dmax')
        for r in regions:
            prob += D0[r] + gamma * x[r]['D'] <= M
        for r in regions:
            prob += D0[r] + gamma * x[r]['D'] >= lam * M
            
    # Giải LP
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if prob.status != pulp.LpStatusOptimal:
        raise ValueError("Không tìm thấy nghiệm tối ưu cho Bài 4 LP PuLP.")
        
    # Chuyển kết quả về dạng ma trận numpy 6x4
    res_matrix = np.zeros((6, 4))
    for r_idx, r in enumerate(regions):
        for j_idx, j in enumerate(items):
            res_matrix[r_idx, j_idx] = x[r][j].varValue
            
    return res_matrix, pulp.value(prob.objective)

def solve_allocation_cvxpy(
    budget: float = 50000.0,
    gamma: float = 0.002,
    lam: float = 0.7,
    with_fairness: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Giải bài toán phân bổ ngân sách 6 vùng x 4 hạng mục (Bài 4) bằng CVXPY.
    
    Returns:
        Tuple[np.ndarray, float]:
            - x_matrix: Ma trận đầu tư tối ưu (6 vùng x 4 hạng mục).
            - Z_opt: GDP tăng cực đại vùng miền (tỷ VND).
    """
    # Ma trận beta: 6 vùng x 4 hạng mục
    beta_mat = np.array([
        [1.15, 0.85, 0.55, 1.30], # NMM
        [0.95, 1.25, 1.40, 1.05], # RRD
        [1.05, 0.95, 0.85, 1.15], # NCC
        [1.20, 0.75, 0.45, 1.35], # CH
        [0.90, 1.30, 1.55, 1.00], # SE
        [1.10, 0.85, 0.65, 1.25]  # MD
    ])
    
    D0_vec = np.array([38.0, 78.0, 55.0, 32.0, 82.0, 48.0])
    
    import cvxpy as cp
    # Biến quyết định trong CVXPY
    X = cp.Variable((6, 4), nonneg=True)
    
    # Hàm mục tiêu
    objective = cp.Maximize(cp.sum(cp.multiply(beta_mat, X)))
    
    constraints = []
    # C1: Ngân sách tổng
    constraints.append(cp.sum(X) <= budget)
    
    # C2 & C3: Sàn/Trần đầu tư mỗi vùng
    regional_sums = cp.sum(X, axis=1)
    constraints.append(regional_sums >= 5000)
    constraints.append(regional_sums <= 12000)
    
    # C4: Sàn đào tạo nhân lực số tổng thể
    constraints.append(cp.sum(X[:, 3]) >= 12000)
    
    # C5: Ràng buộc công bằng vùng miền
    if with_fairness:
        # Trong CVXPY ta dùng cp.max để biểu diễn tối đa trực tiếp
        # D_final_vec = D0_vec + gamma * X[:, 1]
        D_final = D0_vec + gamma * X[:, 1]
        max_D = cp.max(D_final)
        
        # Ràng buộc: D_final_r >= lam * max_D cho mọi r
        for r in range(6):
            constraints.append(D_final[r] >= lam * max_D)
            
    # Giải bài toán tối ưu lồi
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Không giải được LP bằng CVXPY: Status = {prob.status}")
        
    return X.value, prob.value

def solve_project_selection(
    budget_total: float = 80000.0,
    budget_12: float = 40000.0,
    force_p1_p2: bool = False,
    use_risk: bool = False,
    probabilities_dict: Optional[Dict[int, float]] = None
) -> Tuple[List[int], float, float]:
    """
    Giải bài toán quy hoạch nguyên MIP lựa chọn dự án chuyển đổi số (Bài 5) bằng PuLP.
    
    Args:
        budget_total: Tổng ngân sách 5 năm (tỷ VND).
        budget_12: Ngân sách tối đa cho năm 1-2 (tỷ VND).
        force_p1_p2: Nếu True, bỏ qua ràng buộc loại trừ C3 và ép chọn cả P1 và P2.
        use_risk: Nếu True, tối ưu hóa lợi ích kỳ vọng E[Z] = sum(p_i * B_i * y_i).
        probabilities_dict: Từ điển chứa xác suất thành công p_i của mỗi dự án.
        
    Returns:
        Tuple[List[int], float, float]:
            - selected_projects: Danh sách các dự án được chọn (1-indexed).
            - total_cost: Tổng chi phí thực tế của các dự án được chọn.
            - total_benefit: Tổng lợi ích (hoặc lợi ích kỳ vọng) Z*.
    """
    P = list(range(1, 16))
    
    # Bảng chi phí C, chi phí 2 năm đầu C1, và lợi ích B của 15 dự án
    C = {1:12000, 2:11500, 3:18000, 4:4500, 5:3200, 6:5800, 7:6500, 8:15000,
         9:2500, 10:7200, 11:4800, 12:8500, 13:20000, 14:3800, 15:1500}
         
    C1 = {1:8500, 2:7500, 3:12000, 4:3500, 5:2500, 6:4000, 7:4500, 8:9000,
          9:1800, 10:5000, 11:3500, 12:5500, 13:13000, 14:2800, 15:1200}
          
    B = {1:21500, 2:20800, 3:32500, 4:9200, 5:6800, 6:11400, 7:12200, 8:28500,
         9:5800, 10:13800, 11:8500, 12:16200, 13:35000, 14:7500, 15:3800}
         
    import pulp
    # Khởi tạo bài toán MIP
    prob = pulp.LpProblem("VN_Project_Selection", pulp.LpMaximize)
    
    # Biến quyết định nhị phân
    y = pulp.LpVariable.dicts('y', P, cat='Binary')
    
    # Xác định hàm mục tiêu
    if use_risk:
        if probabilities_dict is None:
            # Xác suất mặc định nếu không truyền
            probabilities_dict = {
                1:0.85, 2:0.85, 3:0.85, # Hạ tầng
                4:0.75, 5:0.75,         # Chính phủ
                8:0.65, 12:0.65, 13:0.65, # AI & bán dẫn
                6:0.80, 7:0.80, 9:0.80, 10:0.80, 11:0.80, 14:0.80, 15:0.80 # Còn lại
            }
        prob += pulp.lpSum(probabilities_dict[i] * B[i] * y[i] for i in P), "Expected_Total_Benefit"
    else:
        prob += pulp.lpSum(B[i] * y[i] for i in P), "Total_NPV_Benefit"
        
    # C1 & C2: Ràng buộc ngân sách
    prob += pulp.lpSum(C[i] * y[i] for i in P) <= budget_total, "Budget_Total"
    prob += pulp.lpSum(C1[i] * y[i] for i in P) <= budget_12, "Budget_12_Years"
    
    # C3: Ràng buộc loại trừ
    if not force_p1_p2:
        prob += y[1] + y[2] <= 1, "Exclusive_Datacenter"
    else:
        prob += y[1] == 1
        prob += y[2] == 1
        
    # C4 & C5: Ràng buộc tiên quyết (phải đào tạo nhân lực y12 trước khi triển khai AI y8 và bán dẫn y13)
    prob += y[8] <= y[12], "Prereq_AI_Training"
    prob += y[13] <= y[12], "Prereq_Semi_Training"
    
    # C6: Ràng buộc cân đối lĩnh vực (Chính phủ số tối thiểu 1 dự án và An ninh mạng là bắt buộc)
    prob += y[4] + y[5] >= 1, "Min_One_Gov"
    prob += y[14] == 1, "Mandatory_CyberSecurity"
    
    # C7: Số lượng dự án chọn giới hạn từ 7 đến 11
    prob += pulp.lpSum(y[i] for i in P) >= 7, "Min_Projects"
    prob += pulp.lpSum(y[i] for i in P) <= 11, "Max_Projects"
    
    # Giải MIP bằng solver CBC mặc định
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Trả về kết quả nếu tìm được lời giải khả thi
    if prob.status != pulp.LpStatusOptimal:
        # Nếu không khả thi và chúng ta bị ép buộc, trả về rỗng
        return [], 0.0, 0.0
        
    selected = [i for i in P if y[i].varValue > 0.5]
    total_cost = sum(C[i] for i in selected)
    total_benefit = pulp.value(prob.objective)
    
    return selected, total_cost, total_benefit


def optimize_budget_allocation(budget_dict, sectors_df=None, regions_df=None):
    """
    Wrapper function for dashboard integration - optimizes budget allocation.
    """
    import pandas as pd
    import numpy as np
    import pulp
    
    regions = ['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD']
    items = ['I', 'D', 'AI', 'H']
    
    # Bảng hệ số tác động biên beta
    beta = {
        ('NMM','I'):1.15,('NMM','D'):0.85,('NMM','AI'):0.55,('NMM','H'):1.30,
        ('RRD','I'):0.95,('RRD','D'):1.25,('RRD','AI'):1.40,('RRD','H'):1.05,
        ('NCC','I'):1.05,('NCC','D'):0.95,('NCC','AI'):0.85,('NCC','H'):1.15,
        ('CH','I') :1.20,('CH','D') :0.75,('CH','AI') :0.45,('CH','H') :1.35,
        ('SE','I') :0.90,('SE','D') :1.30,('SE','AI') :1.55,('SE','H') :1.00,
        ('MD','I') :1.10,('MD','D') :0.85,('MD','AI') :0.65,('MD','H') :1.25
    }
    D0 = {'NMM':38.0, 'RRD':78.0, 'NCC':55.0, 'CH':32.0, 'SE':82.0, 'MD':48.0}
    
    try:
        # Map budget categories: K -> I
        budget_map = {
            'I': budget_dict.get('K', budget_dict.get('I', 20000.0)),
            'D': budget_dict.get('D', 15000.0),
            'AI': budget_dict.get('AI', 15000.0),
            'H': budget_dict.get('H', 15000.0)
        }
        total_budget = sum(budget_map.values())
        
        prob = pulp.LpProblem("Region_Budget_Dashboard", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x_dash", (regions, items), lowBound=0)
        M = pulp.LpVariable("M_dash")
        
        prob += pulp.lpSum(beta[(r,j)] * x[r][j] for r in regions for j in items)
        
        # Constraints
        # Budget for each category
        for j in items:
            prob += pulp.lpSum(x[r][j] for r in regions) == budget_map[j]
            
        # Regional bounds adjusted to total budget to guarantee feasibility
        min_reg = max(0.0, min(5000.0, total_budget / 6.0 - 1000.0))
        max_reg = max(12000.0, total_budget / 6.0 + 3000.0)
        for r in regions:
            prob += pulp.lpSum(x[r][j] for j in items) >= min_reg
            prob += pulp.lpSum(x[r][j] for j in items) <= max_reg
            
        # Equity constraint
        gamma = 0.002
        lam = 0.70
        for r in regions:
            prob += D0[r] + gamma * x[r]['D'] <= M
        for r in regions:
            prob += D0[r] + gamma * x[r]['D'] >= lam * M
            
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if prob.status == pulp.LpStatusOptimal:
            allocation_by_region = np.array([
                sum(x[r][j].varValue for j in items) for r in regions
            ])
            expected_benefit = pulp.value(prob.objective)
            return {
                'allocation_by_region': allocation_by_region,
                'expected_benefit': expected_benefit
            }
            
    except Exception as e:
        print(f"Error in optimize_budget_allocation LP: {e}")
        
    # Heuristic fallback (proportional to initial digital index or uniform)
    total_b = sum(budget_dict.values()) if isinstance(budget_dict, dict) else budget_dict
    shares = np.array([38.0, 78.0, 55.0, 32.0, 82.0, 48.0])
    shares = shares / shares.sum()
    allocation_by_region = shares * total_b
    expected_benefit = total_b * 1.15
    return {
        'allocation_by_region': allocation_by_region,
        'expected_benefit': expected_benefit
    }
