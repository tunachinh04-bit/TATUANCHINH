import numpy as np
from typing import Dict, Any, Tuple, List, Optional

# Thiết lập các tham số mặc định của bài toán Tối ưu hóa ngẫu nhiên 2 giai đoạn (Bài 10)
DEFAULT_CONFIG = {
    'scenarios': ['Good', 'Base', 'Bad'],
    'probabilities': [0.3, 0.5, 0.2],
    'categories': ['Infra', 'AI', 'Human'],
    'first_stage_cost': [8.0, 12.0, 10.0],  # c_i
    'budget': 100.0,
    'min_investment': [10.0, 10.0, 10.0],  # x_i >= 10
    # Nhu cầu thị trường d_{s,i}
    'demands': {
        'Good': [50.0, 40.0, 45.0],
        'Base': [30.0, 25.0, 30.0],
        'Bad': [15.0, 10.0, 15.0]
    },
    # Doanh thu biên ở giai đoạn 2 q_{s,i}
    'revenues': {
        'Good': [15.0, 22.0, 18.0],
        'Base': [12.0, 18.0, 14.0],
        'Bad': [8.0, 10.0, 9.0]
    }
}

def solve_stochastic_rp_pulp(config: Dict[str, Any] = DEFAULT_CONFIG) -> Tuple[np.ndarray, Dict[str, np.ndarray], float]:
    """
    Giải Bài toán Lập trình Ngẫu nhiên với Quyết định bù đắp (Recourse Problem - RP) bằng PuLP.
    Mục tiêu: Cực đại hóa Lợi nhuận Kỳ vọng ròng.
    E[Net Profit] = sum_s p_s * (sum_i q_{s,i} * y_{s,i}) - sum_i c_i * x_i
    
    Returns:
        Tuple[np.ndarray, Dict[str, np.ndarray], float]:
            - x_opt: Quyết định giai đoạn 1 tối ưu (Đầu tư trước khi biết kịch bản).
            - y_opt_s: Quyết định giai đoạn 2 cho từng kịch bản s.
            - Z_RP: Lợi nhuận kỳ vọng ròng cực đại.
    """
    scenarios = config['scenarios']
    probs = config['probabilities']
    categories = config['categories']
    c = config['first_stage_cost']
    budget = config['budget']
    min_x = config['min_investment']
    demands = config['demands']
    revenues = config['revenues']
    
    n_cats = len(categories)
    
    import pulp
    # Khởi tạo bài toán
    prob = pulp.LpProblem("Stochastic_Recourse_Problem", pulp.LpMaximize)
    
    # Biến quyết định giai đoạn 1
    x = [pulp.LpVariable(f'x_{cat}', lowBound=min_x[i]) for i, cat in enumerate(categories)]
    
    # Biến quyết định giai đoạn 2 (phụ thuộc vào từng kịch bản s)
    y = {}
    for s in scenarios:
        y[s] = [pulp.LpVariable(f'y_{s}_{cat}', lowBound=0) for cat in categories]
        
    # Hàm mục tiêu: Lợi nhuận Kỳ vọng ròng
    # Doanh thu giai đoạn 2 kỳ vọng
    expected_revenue = pulp.lpSum(
        probs[s_idx] * pulp.lpSum(revenues[s][i] * y[s][i] for i in range(n_cats))
        for s_idx, s in enumerate(scenarios)
    )
    # Chi phí giai đoạn 1
    first_stage_cost = pulp.lpSum(c[i] * x[i] for i in range(n_cats))
    
    prob += expected_revenue - first_stage_cost, "Expected_Net_Profit"
    
    # Ràng buộc giai đoạn 1: Ngân sách đầu tư
    prob += pulp.lpSum(x[i] for i in range(n_cats)) <= budget, "Budget_Constraint"
    
    # Ràng buộc giai đoạn 2:
    for s_idx, s in enumerate(scenarios):
        for i in range(n_cats):
            # y_{s,i} <= x_i (Quyết định vận hành không vượt quá năng lực đã đầu tư)
            prob += y[s][i] <= x[i], f"Capacity_Limit_{s}_{categories[i]}"
            # y_{s,i} <= d_{s,i} (Quyết định vận hành không vượt quá nhu cầu thị trường)
            prob += y[s][i] <= demands[s][i], f"Demand_Limit_{s}_{categories[i]}"
            
    # Giải bài toán
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if prob.status != pulp.LpStatusOptimal:
        raise ValueError("Không tìm thấy nghiệm tối ưu cho bài toán RP.")
        
    x_opt = np.array([x[i].varValue for i in range(n_cats)])
    
    y_opt_s = {}
    for s in scenarios:
        y_opt_s[s] = np.array([y[s][i].varValue for i in range(n_cats)])
        
    Z_RP = pulp.value(prob.objective)
    
    return x_opt, y_opt_s, Z_RP

def solve_expected_value_problem_pulp(config: Dict[str, Any] = DEFAULT_CONFIG) -> Tuple[np.ndarray, float, float]:
    """
    1. Giải bài toán Tất định Giá trị Kỳ vọng (Expected Value problem - EV) bằng cách
       thay thế các yếu tố ngẫu nhiên bằng kỳ vọng toán học của chúng.
    2. Sử dụng quyết định x_EV tìm được để tính EEV (Expected Result of Expected Value)
       bằng cách cố định x = x_EV và giải bài toán tối ưu giai đoạn 2 trên từng kịch bản.
       
    Returns:
        Tuple[np.ndarray, float, float]:
            - x_EV: Quyết định giai đoạn 1 từ bài toán EV.
            - Z_EV: Mục tiêu của bài toán EV (không phải EEV).
            - Z_EEV: Lợi nhuận kỳ vọng thực tế khi áp dụng x_EV (EEV).
    """
    scenarios = config['scenarios']
    probs = config['probabilities']
    categories = config['categories']
    c = config['first_stage_cost']
    budget = config['budget']
    min_x = config['min_investment']
    demands = config['demands']
    revenues = config['revenues']
    
    n_cats = len(categories)
    
    # ------------------ BƯỚC 1: GIẢI BÀI TOÁN EV (EXPECTED VALUE) ------------------
    # Tính nhu cầu kỳ vọng và doanh thu kỳ vọng
    mean_demand = np.zeros(n_cats)
    mean_revenue = np.zeros(n_cats)
    for s_idx, s in enumerate(scenarios):
        mean_demand += probs[s_idx] * np.array(demands[s])
        mean_revenue += probs[s_idx] * np.array(revenues[s])
        
    import pulp
    prob_ev = pulp.LpProblem("Expected_Value_Problem", pulp.LpMaximize)
    
    x_ev_vars = [pulp.LpVariable(f'x_ev_{cat}', lowBound=min_x[i]) for i, cat in enumerate(categories)]
    y_ev_vars = [pulp.LpVariable(f'y_ev_{cat}', lowBound=0) for cat in categories]
    
    # Hàm mục tiêu bài toán EV
    ev_revenue = pulp.lpSum(mean_revenue[i] * y_ev_vars[i] for i in range(n_cats))
    ev_cost = pulp.lpSum(c[i] * x_ev_vars[i] for i in range(n_cats))
    prob_ev += ev_revenue - ev_cost, "EV_Objective"
    
    # Ràng buộc
    prob_ev += pulp.lpSum(x_ev_vars[i] for i in range(n_cats)) <= budget, "EV_Budget"
    for i in range(n_cats):
        prob_ev += y_ev_vars[i] <= x_ev_vars[i]
        prob_ev += y_ev_vars[i] <= mean_demand[i]
        
    prob_ev.solve(pulp.PULP_CBC_CMD(msg=False))
    
    x_EV = np.array([x_ev_vars[i].varValue for i in range(n_cats)])
    Z_EV = pulp.value(prob_ev.objective)
    
    # ------------------ BƯỚC 2: TÍNH EEV VỚI x CỐ ĐỊNH = x_EV ------------------
    # Với mỗi kịch bản s, ta cố định x = x_EV và tìm y_s tối ưu
    eev_revenues = []
    
    for s in scenarios:
        import pulp
        prob_s = pulp.LpProblem(f"EEV_Scenario_{s}", pulp.LpMaximize)
        y_s_vars = [pulp.LpVariable(f'y_eev_{s}_{cat}', lowBound=0) for cat in categories]
        
        prob_s += pulp.lpSum(revenues[s][i] * y_s_vars[i] for i in range(n_cats)), "Scenario_Revenue"
        
        for i in range(n_cats):
            prob_s += y_s_vars[i] <= x_EV[i]
            prob_s += y_s_vars[i] <= demands[s][i]
            
        prob_s.solve(pulp.PULP_CBC_CMD(msg=False))
        eev_revenues.append(pulp.value(prob_s.objective))
        
    # Lợi nhuận kỳ vọng ròng EEV
    first_stage_cost_ev = np.sum(c * x_EV)
    expected_eev_revenue = np.sum(np.array(probs) * np.array(eev_revenues))
    Z_EEV = expected_eev_revenue - first_stage_cost_ev
    
    return x_EV, Z_EV, Z_EEV

def solve_wait_and_see_pulp(config: Dict[str, Any] = DEFAULT_CONFIG) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Giải bài toán Chờ và Xem (Wait-and-See - WS).
    Giả định nhà hoạch định chính sách có thông tin hoàn hảo về kịch bản trước khi ra quyết định đầu tư x.
    Tương đương giải bài toán tất định riêng lẻ cho từng kịch bản s rồi tính kỳ vọng.
    
    Returns:
        Tuple[Dict[str, np.ndarray], float]:
            - x_opt_s: Quyết định đầu tư tối ưu cho từng kịch bản s.
            - Z_WS: Lợi nhuận kỳ vọng của Wait-and-See.
    """
    scenarios = config['scenarios']
    probs = config['probabilities']
    categories = config['categories']
    c = config['first_stage_cost']
    budget = config['budget']
    min_x = config['min_investment']
    demands = config['demands']
    revenues = config['revenues']
    
    n_cats = len(categories)
    
    ws_profits = []
    x_opt_s = {}
    
    for s_idx, s in enumerate(scenarios):
        import pulp
        prob_s = pulp.LpProblem(f"WS_Scenario_{s}", pulp.LpMaximize)
        
        x_vars = [pulp.LpVariable(f'x_ws_{s}_{cat}', lowBound=min_x[i]) for i, cat in enumerate(categories)]
        y_vars = [pulp.LpVariable(f'y_ws_{s}_{cat}', lowBound=0) for cat in categories]
        
        # Lợi nhuận ròng của kịch bản s
        revenue = pulp.lpSum(revenues[s][i] * y_vars[i] for i in range(n_cats))
        cost = pulp.lpSum(c[i] * x_vars[i] for i in range(n_cats))
        prob_s += revenue - cost, "Scenario_Net_Profit"
        
        prob_s += pulp.lpSum(x_vars[i] for i in range(n_cats)) <= budget, "Budget_Constraint"
        for i in range(n_cats):
            prob_s += y_vars[i] <= x_vars[i]
            prob_s += y_vars[i] <= demands[s][i]
            
        prob_s.solve(pulp.PULP_CBC_CMD(msg=False))
        
        ws_profits.append(pulp.value(prob_s.objective))
        x_opt_s[s] = np.array([x_vars[i].varValue for i in range(n_cats)])
        
    Z_WS = np.sum(np.array(probs) * np.array(ws_profits))
    
    return x_opt_s, Z_WS

def solve_minimax_regret_pulp(config: Dict[str, Any] = DEFAULT_CONFIG) -> Tuple[np.ndarray, float]:
    """
    Giải bài toán Tối thiểu hóa Hối tiếc Cực đại (Robust Minimax Regret).
    Mục tiêu: Lựa chọn x sao cho hối tiếc lớn nhất giữa lợi nhuận tối ưu của kịch bản s (Z*_s)
    và lợi nhuận thực tế đạt được dưới quyết định x là nhỏ nhất.
    Minimizes H, subject to:
    H >= Z*_s - [sum_i q_{s,i} * y_{s,i} - sum_i c_i * x_i]  với mọi kịch bản s.
    
    Returns:
        Tuple[np.ndarray, float]:
            - x_opt: Quyết định đầu tư cực kỳ vững chắc (robust).
            - max_regret: Giá trị hối tiếc cực đại tối thiểu hóa.
    """
    scenarios = config['scenarios']
    categories = config['categories']
    c = config['first_stage_cost']
    budget = config['budget']
    min_x = config['min_investment']
    demands = config['demands']
    revenues = config['revenues']
    
    n_cats = len(categories)
    
    # Bước 1: Tính lợi nhuận tối ưu tuyệt đối Z*_s cho từng kịch bản s độc lập
    x_opt_s, _ = solve_wait_and_see_pulp(config)
    Z_star_s = {}
    
    for s in scenarios:
        import pulp
        # Giải lại nhanh để lấy trị số mục tiêu chính xác của kịch bản s
        prob_s = pulp.LpProblem(f"Z_star_{s}", pulp.LpMaximize)
        x_vars = [pulp.LpVariable(f'x_star_{s}_{cat}', lowBound=min_x[i]) for i, cat in enumerate(categories)]
        y_vars = [pulp.LpVariable(f'y_star_{s}_{cat}', lowBound=0) for cat in categories]
        
        prob_s += pulp.lpSum(revenues[s][i] * y_vars[i] for i in range(n_cats)) - pulp.lpSum(c[i] * x_vars[i] for i in range(n_cats))
        prob_s += pulp.lpSum(x_vars[i] for i in range(n_cats)) <= budget
        for i in range(n_cats):
            prob_s += y_vars[i] <= x_vars[i]
            prob_s += y_vars[i] <= demands[s][i]
            
        prob_s.solve(pulp.PULP_CBC_CMD(msg=False))
        Z_star_s[s] = pulp.value(prob_s.objective)
        
    import pulp
    # Bước 2: Thiết lập bài toán tối thiểu hóa hối tiếc cực đại
    prob_regret = pulp.LpProblem("Minimax_Regret_Problem", pulp.LpMinimize)
    
    # Biến quyết định giai đoạn 1
    x = [pulp.LpVariable(f'x_regret_{cat}', lowBound=min_x[i]) for i, cat in enumerate(categories)]
    
    # Biến quyết định giai đoạn 2 cho từng kịch bản
    y = {}
    for s in scenarios:
        y[s] = [pulp.LpVariable(f'y_regret_{s}_{cat}', lowBound=0) for cat in categories]
        
    # Biến phụ biểu thị hối tiếc cực đại H
    H = pulp.LpVariable("Max_Regret", lowBound=0)
    
    prob_regret += H, "Minimize_Max_Regret"
    
    # Ràng buộc ngân sách giai đoạn 1
    prob_regret += pulp.lpSum(x[i] for i in range(n_cats)) <= budget, "Budget_Constraint"
    
    first_stage_cost = pulp.lpSum(c[i] * x[i] for i in range(n_cats))
    
    # Ràng buộc giai đoạn 2 và Ràng buộc Hối tiếc cho từng kịch bản
    for s in scenarios:
        for i in range(n_cats):
            prob_regret += y[s][i] <= x[i], f"Cap_Limit_{s}_{categories[i]}"
            prob_regret += y[s][i] <= demands[s][i], f"Dem_Limit_{s}_{categories[i]}"
            
        # Regret_s = Z*_s - (Revenue_s - Cost_1st)
        profit_s = pulp.lpSum(revenues[s][i] * y[s][i] for i in range(n_cats)) - first_stage_cost
        prob_regret += H >= Z_star_s[s] - profit_s, f"Regret_Constraint_{s}"
        
    prob_regret.solve(pulp.PULP_CBC_CMD(msg=False))
    
    x_opt = np.array([x[i].varValue for i in range(n_cats)])
    max_regret = pulp.value(H)
    
    return x_opt, max_regret

def calculate_risk_metrics(config: Dict[str, Any] = DEFAULT_CONFIG) -> Dict[str, Any]:
    """
    Tính toán toàn bộ các chỉ số đo lường rủi ro kinh tế:
    - RP (Recourse Problem)
    - EV (Expected Value Problem)
    - EEV (Expected result of Expected Value)
    - WS (Wait-and-See)
    - VSS (Value of Stochastic Solution) = RP - EEV
    - EVPI (Expected Value of Perfect Information) = WS - RP
    
    Returns:
        Dict[str, Any]: Từ điển chứa các kết quả tính toán chi tiết.
    """
    x_RP, _, Z_RP = solve_stochastic_rp_pulp(config)
    x_EV, Z_EV, Z_EEV = solve_expected_value_problem_pulp(config)
    _, Z_WS = solve_wait_and_see_pulp(config)
    
    VSS = Z_RP - Z_EEV
    EVPI = Z_WS - Z_RP
    
    return {
        'x_RP': x_RP,
        'Z_RP': Z_RP,
        'x_EV': x_EV,
        'Z_EV': Z_EV,
        'Z_EEV': Z_EEV,
        'Z_WS': Z_WS,
        'VSS': VSS,
        'EVPI': EVPI
    }

# ----------------- PHẦN PYOMO HOÀN THIỆN (CÓ CHẾ CHẾ FALLBACK SANG PULP) -----------------
try:
    import pyomo.environ as pyo
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False

def solve_stochastic_pyomo(config: Dict[str, Any] = DEFAULT_CONFIG) -> Tuple[np.ndarray, float]:
    """
    Giải bài toán lập trình ngẫu nhiên bằng Pyomo.
    Nếu hệ thống không có sẵn solver thích hợp (như glpk hoặc cbc) được cài ngoài,
    hàm sẽ tự động chuyển hướng và giải bằng PuLP để đảm bảo hệ thống không bị crash.
    """
    if not PYOMO_AVAILABLE:
        # Fallback sang PuLP nếu không import được pyomo
        x_opt, _, Z_RP = solve_stochastic_rp_pulp(config)
        return x_opt, Z_RP
        
    scenarios = config['scenarios']
    probs = config['probabilities']
    categories = config['categories']
    c = config['first_stage_cost']
    budget = config['budget']
    min_x = config['min_investment']
    demands = config['demands']
    revenues = config['revenues']
    
    n_cats = len(categories)
    
    model = pyo.ConcreteModel()
    
    # Khai báo tập chỉ số
    model.I = pyo.RangeSet(0, n_cats - 1)
    model.S = pyo.Set(initialize=scenarios)
    
    # Biến quyết định giai đoạn 1
    def x_bounds(model, i):
        return (min_x[i], None)
    model.x = pyo.Var(model.I, bounds=x_bounds)
    
    # Biến quyết định giai đoạn 2
    model.y = pyo.Var(model.S, model.I, bounds=(0, None))
    
    # Hàm mục tiêu: Kỳ vọng lợi nhuận ròng
    def obj_rule(model):
        exp_revenue = sum(
            probs[s_idx] * sum(revenues[s][i] * model.y[s, i] for i in model.I)
            for s_idx, s in enumerate(scenarios)
        )
        cost = sum(c[i] * model.x[i] for i in model.I)
        return exp_revenue - cost
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    
    # Ràng buộc ngân sách giai đoạn 1
    def budget_rule(model):
        return sum(model.x[i] for i in model.I) <= budget
    model.budget_constraint = pyo.Constraint(rule=budget_rule)
    
    # Ràng buộc giai đoạn 2 cho biến y
    def cap_rule(model, s, i):
        return model.y[s, i] <= model.x[i]
    model.cap_constraints = pyo.Constraint(model.S, model.I, rule=cap_rule)
    
    def demand_rule(model, s, i):
        return model.y[s, i] <= demands[s][i]
    model.demand_constraints = pyo.Constraint(model.S, model.I, rule=demand_rule)
    
    # Thử giải bằng solver glpk hoặc cbc qua Pyomo
    # Nếu thất bại, tự động fallback sang PuLP
    solved = False
    for solver_name in ['glpk', 'cbc']:
        try:
            solver = pyo.SolverFactory(solver_name)
            results = solver.solve(model, tee=False)
            if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
                solved = True
                break
        except Exception:
            continue
            
    if solved:
        x_opt = np.array([pyo.value(model.x[i]) for i in model.I])
        Z_RP = pyo.value(model.obj)
        return x_opt, Z_RP
    else:
        # Fallback sang PuLP solver
        x_opt, _, Z_RP = solve_stochastic_rp_pulp(config)
        return x_opt, Z_RP


def assess_risk_profile(sectors_df=None, allocation_dict=None):
    """
    Wrapper function for dashboard integration - assesses risk profile.
    
    Args:
        sectors_df: DataFrame with sectors (optional)
        allocation_dict: Dict with K, D, AI, H allocations (optional)
    
    Returns:
        dict: Contains 'cyber_risk', 'environmental_impact', 'regional_inequality'
    """
    try:
        # Default allocation weights if not provided
        if allocation_dict is None:
            allocation_dict = {'K': 0.35, 'D': 0.25, 'AI': 0.20, 'H': 0.20}
        
        # Calculate risk profile based on allocation
        # More AI/D = higher cyber risk, higher environmental impact potentially
        # More K = more environmental impact
        # More H = lower regional inequality
        
        if isinstance(allocation_dict, dict):
            k_weight = allocation_dict.get('K', 0.35)
            d_weight = allocation_dict.get('D', 0.25)
            ai_weight = allocation_dict.get('AI', 0.20)
            h_weight = allocation_dict.get('H', 0.20)
        else:
            k_weight = d_weight = ai_weight = h_weight = 0.25
        
        # Cyber risk: increases with AI/D investment, baseline 0.4-0.6
        base_cyber = 0.5
        cyber_risk = base_cyber + (ai_weight * 0.15) + (d_weight * 0.1) - (h_weight * 0.08)
        cyber_risk = np.clip(cyber_risk, 0.25, 0.85)
        
        # Environmental impact: increases with K (infrastructure), decreases with H (awareness)
        base_environ = 0.45
        environ_impact = base_environ + (k_weight * 0.1) - (h_weight * 0.05)
        environ_impact = np.clip(environ_impact, 0.3, 0.75)
        
        # Regional inequality: decreases with H investment (human capital/education)
        # Baseline Gini ~0.38, target: lower with inclusive policies
        base_gini = 0.38
        regional_inequality = base_gini - (h_weight * 0.05)
        regional_inequality = np.clip(regional_inequality, 0.33, 0.48)
        
        return {
            'cyber_risk': float(cyber_risk),
            'environmental_impact': float(environ_impact),
            'regional_inequality': float(regional_inequality),
            'data_security_score': 1 - cyber_risk,
            'sustainability_score': 1 - environ_impact,
            'inclusivity_score': 1 - regional_inequality
        }
    except Exception as e:
        # Fallback
        print(f"Warning in assess_risk_profile: {e}")
        return {
            'cyber_risk': 0.55,
            'environmental_impact': 0.45,
            'regional_inequality': 0.38,
            'data_security_score': 0.45,
            'sustainability_score': 0.55,
            'inclusivity_score': 0.62
        }
