import numpy as np
import pandas as pd
import pytest
from aideom_vn.src.m1_production import compute_tfp, scenario_2030
from aideom_vn.src.m2_readiness import topsis, entropy_weights
from aideom_vn.src.m3_allocation import solve_lp_pulp, solve_project_selection
from aideom_vn.src.m4_labor import simulate_labor_dynamics, solve_netjob_maximization
from aideom_vn.src.m5_risk import calculate_risk_metrics, solve_minimax_regret_pulp
from aideom_vn.src.rl_env import VietnamEconomyEnv

# ==============================================================================
# 1. KIỂM THỬ MODULE M1 (PRODUCTION)
# ==============================================================================
def test_compute_tfp():
    """Kiểm tra tính toán TFP từ hàm Cobb-Douglas mở rộng."""
    Y, K, L, D, AI, H = 1000.0, 5000.0, 10.0, 20.0, 50.0, 30.0
    tfp = compute_tfp(Y, K, L, D, AI, H)
    assert tfp > 0
    # Kiểm tra tính đồng dạng: TFP nhân với mẫu số phải bằng Y
    alpha, beta, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07
    denom = (K ** alpha) * (L ** beta) * (D ** gamma) * (AI ** delta) * (H ** theta)
    assert np.allclose(tfp * denom, Y)

def test_scenario_2030():
    """Kiểm tra dự báo GDP kịch bản năm 2030."""
    res = scenario_2030(A_trend=1.2, D_target=30.0, AI_target=100.0, H_target=35.0)
    assert 'GDP_2030' in res
    assert res['GDP_2030'] > 0
    assert res['K_2030'] > 25900.0  # Vốn phải tăng lũy kế

# ==============================================================================
# 2. KIỂM THỬ MODULE M2 (READINESS / TOPSIS)
# ==============================================================================
def test_topsis():
    """Kiểm tra thuật toán TOPSIS từ đầu bằng NumPy."""
    X = np.array([
        [80, 20],
        [50, 50],
        [20, 80]
    ])
    w = np.array([0.6, 0.4])
    is_benefit = [True, True]
    scores, ranks = topsis(X, w, is_benefit)
    assert len(scores) == 3
    assert len(ranks) == 3
    # Phương án có điểm cao nhất phải xếp hạng 1
    best_idx = np.argmax(scores)
    assert ranks[best_idx] == 1

def test_entropy_weights():
    """Kiểm tra thuật toán tính trọng số Entropy khách quan."""
    X = np.array([
        [10.0, 100.0],
        [20.0, 100.0],
        [30.0, 100.0]
    ])
    weights = entropy_weights(X)
    assert len(weights) == 2
    assert np.allclose(np.sum(weights), 1.0)
    # Cột 2 toàn trị số giống nhau (100.0), entropy cực đại, độ phân kỳ bằng 0, trọng số phải rất nhỏ hoặc bằng 0
    assert weights[1] < 1e-3

# ==============================================================================
# 3. KIỂM THỬ MODULE M3 (ALLOCATION / LP / MIP)
# ==============================================================================
def test_solve_lp_pulp():
    """Kiểm tra quy hoạch tuyến tính phân bổ ngân sách LP."""
    x_opt, Z_opt, shadow_prices = solve_lp_pulp(100.0)
    assert len(x_opt) == 4
    assert np.allclose(np.sum(x_opt), 100.0)  # Phải dùng hết ngân sách
    assert Z_opt > 0
    assert 'budget' in shadow_prices

def test_solve_project_selection():
    """Kiểm tra quy hoạch nguyên MIP lựa chọn dự án chuyển đổi số."""
    selected, cost, benefit = solve_project_selection(budget_total=80000.0)
    assert len(selected) >= 7  # Ràng buộc số lượng dự án tối thiểu
    assert len(selected) <= 11 # Ràng buộc số lượng dự án tối đa
    assert cost <= 80000.0     # Ràng buộc ngân sách
    assert benefit > 0
    # Dự án 14 (Cyber Security) bắt buộc phải chọn
    assert 14 in selected
    # Ràng buộc loại trừ: không chọn cả 1 và 2 cùng lúc
    assert not (1 in selected and 2 in selected)

# ==============================================================================
# 4. KIỂM THỬ MODULE M4 (LABOR DYNAMICS)
# ==============================================================================
def test_simulate_labor_dynamics():
    """Kiểm tra mô phỏng thị trường lao động."""
    df_sectors = pd.DataFrame({
        'sector_id': [1, 2],
        'sector_name_vi': ['Nông nghiệp', 'Chế tạo'],
        'labor_million': [10.0, 8.0],
        'ai_readiness_0_100': [20, 60],
        'automation_risk_pct': [15, 40]
    })
    x_inv = np.array([5000.0, 10000.0])
    df_res = simulate_labor_dynamics(df_sectors, x_inv)
    assert 'jobs_created_thousand' in df_res
    assert 'jobs_displaced_thousand' in df_res
    assert 'net_jobs_thousand' in df_res

def test_solve_netjob_maximization():
    """Kiểm tra tối đa hóa tạo việc làm ròng toàn nền kinh tế."""
    x_opt, Z_opt = solve_netjob_maximization(budget_total=60000.0)
    assert len(x_opt) == 10
    assert np.allclose(np.sum(x_opt), 60000.0)
    assert Z_opt > 0

# ==============================================================================
# 5. KIỂM THỬ MODULE M5 (RISK DECISION / STOCHASTIC LP)
# ==============================================================================
def test_calculate_risk_metrics():
    """Kiểm tra tính toán toàn bộ các chỉ số đo lường rủi ro kinh tế."""
    metrics = calculate_risk_metrics()
    assert 'Z_RP' in metrics
    assert 'VSS' in metrics
    assert 'EVPI' in metrics
    # Giá trị giải pháp ngẫu nhiên (VSS) và giá trị thông tin hoàn hảo (EVPI) phải luôn >= 0
    assert metrics['VSS'] >= -1e-5
    assert metrics['EVPI'] >= -1e-5

def test_solve_minimax_regret():
    """Kiểm tra quy hoạch vững chắc cực tiểu hóa hối tiếc cực đại."""
    x_regret, max_regret = solve_minimax_regret_pulp()
    assert len(x_regret) == 3
    assert max_regret >= 0

# ==============================================================================
# 6. KIỂM THỬ MODULE M6 (RL / GYMNASIUM ECONOMIC ENVIRONMENT)
# ==============================================================================
def test_vietnam_economy_env():
    """Kiểm tra sự tích hợp của môi trường Gymnasium VietnamEconomyEnv."""
    env = VietnamEconomyEnv()
    state, info = env.reset()
    assert env.observation_space.contains(state)
    assert state == 40  # Trạng thái cơ sở ban đầu (1, 1, 1, 1) -> 1*27 + 1*9 + 1*3 + 1 = 40
    
    # Thực hiện 1 bước hành động chính trị (Action 3 - Phát triển cân bằng)
    next_state, reward, terminated, truncated, info = env.step(3)
    assert env.observation_space.contains(next_state)
    assert isinstance(reward, float)
    assert not terminated
    assert info['step_index'] == 1
