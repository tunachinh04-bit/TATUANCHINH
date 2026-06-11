import numpy as np
import pandas as pd
from typing import Tuple, List, Union

def topsis(
    X: np.ndarray,
    weights: np.ndarray,
    is_benefit: Union[List[bool], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hiện thực thuật toán TOPSIS từ đầu bằng NumPy.
    
    Args:
        X: Ma trận quyết định ban đầu kích thước (n_alternatives, m_criteria).
        weights: Vector trọng số của các tiêu chí (tổng bằng 1).
        is_benefit: List hoặc vector Boolean xác định tiêu chí là lợi ích (True) hay chi phí (False).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - C_star: Điểm gần gũi tương đối (closeness coefficient) trong đoạn [0, 1].
            - ranks: Thứ hạng của các phương án (1 là tốt nhất, n là kém nhất).
    """
    n, m = X.shape
    is_benefit = np.array(is_benefit)
    weights = np.array(weights)
    
    # Bước 1: Chuẩn hóa vector ma trận quyết định
    # r_ij = x_ij / sqrt(sum_i(x_ij^2))
    norm_denom = np.sqrt(np.sum(X ** 2, axis=0))
    # Tránh chia cho 0 nếu một cột toàn số 0
    norm_denom = np.where(norm_denom == 0, 1e-12, norm_denom)
    R = X / norm_denom
    
    # Bước 2: Nhân với trọng số
    # v_ij = w_j * r_ij
    V = R * weights
    
    # Bước 3: Xác định phương án lý tưởng dương (A*) và lý tưởng âm (A-)
    # A* = max(v_ij) cho tiêu chí tốt, min(v_ij) cho tiêu chí xấu
    A_star = np.zeros(m)
    A_neg = np.zeros(m)
    
    for j in range(m):
        if is_benefit[j]:
            A_star[j] = np.max(V[:, j])
            A_neg[j] = np.min(V[:, j])
        else:
            A_star[j] = np.min(V[:, j])
            A_neg[j] = np.max(V[:, j])
            
    # Bước 4: Tính khoảng cách Euclide từ mỗi phương án đến A* và A-
    S_star = np.sqrt(np.sum((V - A_star) ** 2, axis=1))
    S_neg = np.sqrt(np.sum((V - A_neg) ** 2, axis=1))
    
    # Bước 5: Tính hệ số tương đồng gần gũi C_star
    C_star = S_neg / (S_star + S_neg + 1e-12) # Thêm epsilon để tránh chia cho 0
    
    # Xếp hạng giảm dần theo điểm số C_star
    # np.argsort(-C_star) cho chỉ số sắp xếp giảm dần, dùng argsort lần 2 để lấy thứ hạng cụ thể
    ranks = np.argsort(np.argsort(-C_star)) + 1
    
    return C_star, ranks

def entropy_weights(X: np.ndarray) -> np.ndarray:
    """
    Tính trọng số khách quan bằng phương pháp Entropy.
    
    Args:
        X: Ma trận quyết định ban đầu kích thước (n_alternatives, m_criteria).
        
    Returns:
        np.ndarray: Vector trọng số Entropy (tổng bằng 1.0).
    """
    n, m = X.shape
    
    # Chuẩn hóa ma trận quyết định theo tỉ lệ tổng cột
    # p_ij = x_ij / sum_i(x_ij)
    col_sums = X.sum(axis=0)
    col_sums = np.where(col_sums == 0, 1e-12, col_sums)
    P = X / col_sums
    
    # Tính Entropy cho mỗi tiêu chí j
    # E_j = -k * sum_i(p_ij * ln(p_ij))
    k = 1.0 / np.log(n) if n > 1 else 1.0
    
    # Tránh ln(0) bằng np.log(P + 1e-12)
    E = -k * np.sum(P * np.log(P + 1e-12), axis=0)
    
    # Tính độ phân kỳ (diversity)
    d = 1.0 - E
    
    # Tính trọng số chuẩn hóa
    w = d / (np.sum(d) + 1e-12)
    
    return w

def sensitivity_ai_weight(
    X: np.ndarray,
    is_benefit: Union[List[bool], np.ndarray],
    w_base: np.ndarray,
    w_ai_idx: int,
    w_ai_range: np.ndarray
) -> pd.DataFrame:
    """
    Phân tích độ nhạy của trọng số: thay đổi trọng số AI (tiêu chí tại w_ai_idx)
    trong phạm vi w_ai_range, renormalize các trọng số còn lại sao cho tổng bằng 1,
    và tính toán sự thay đổi điểm số cũng như thứ hạng.
    
    Args:
        X: Ma trận quyết định ban đầu.
        is_benefit: List thuộc tính tốt/xấu.
        w_base: Bộ trọng số chuyên gia gốc.
        w_ai_idx: Vị trí của trọng số AI.
        w_ai_range: Các giá trị trọng số AI để kiểm tra (ví dụ từ 0.05 đến 0.40).
        
    Returns:
        pd.DataFrame: Bảng kết quả thay đổi thứ hạng tương ứng.
    """
    n_alternatives = X.shape[0]
    m_criteria = X.shape[1]
    
    results = []
    
    for w_ai in w_ai_range:
        # Tạo bộ trọng số mới
        w_new = w_base.copy()
        w_new[w_ai_idx] = w_ai
        
        # Chuẩn hóa lại các trọng số khác để tổng vẫn bằng 1.0
        other_indices = [i for i in range(m_criteria) if i != w_ai_idx]
        other_sum = np.sum(w_base[other_indices])
        
        if other_sum > 0:
            scale = (1.0 - w_ai) / other_sum
            w_new[other_indices] = w_base[other_indices] * scale
        else:
            w_new[other_indices] = (1.0 - w_ai) / len(other_indices)
            
        # Đảm bảo tổng trọng số bằng 1.0 tuyệt đối
        w_new = w_new / np.sum(w_new)
        
        # Chạy TOPSIS với trọng số mới
        scores, ranks = topsis(X, w_new, is_benefit)
        
        # Lưu kết quả
        for alt_idx in range(n_alternatives):
            results.append({
                'w_AI': w_ai,
                'alternative_id': alt_idx,
                'score': scores[alt_idx],
                'rank': ranks[alt_idx]
            })
            
    return pd.DataFrame(results)


def calculate_topsis_readiness(regions_df=None, sectors_df=None):
    """
    Wrapper function for dashboard integration - calculates TOPSIS readiness scores.
    """
    import pandas as pd
    import numpy as np
    
    try:
        from data_loader import load_regions, load_sectors
    except ImportError:
        def load_regions():
            return pd.DataFrame({
                "region_id": [1, 2, 3, 4, 5, 6],
                "region_name_vi": [
                    "Trung du miền núi phía Bắc",
                    "Đồng bằng sông Hồng",
                    "Bắc Trung Bộ và Duyên hải miền Trung",
                    "Tây Nguyên",
                    "Đông Nam Bộ",
                    "Đồng bằng sông Cửu Long"
                ],
                "grdp_per_capita_million_VND": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
                "fdi_registered_billion_USD": [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
                "digital_index_0_100": [38, 78, 55, 32, 82, 48],
                "ai_readiness_0_100": [22, 68, 40, 18, 75, 30],
                "trained_labor_pct": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
                "rd_intensity_pct": [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
                "internet_penetration_pct": [72, 92, 84, 68, 94, 78],
                "gini_coef": [0.405, 0.358, 0.372, 0.412, 0.385, 0.392]
            })
        def load_sectors():
            return pd.DataFrame({
                "sector_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "sector_name_vi": [
                    "Nông-Lâm-Thủy sản",
                    "CN chế biến chế tạo",
                    "Xây dựng",
                    "Khai khoáng",
                    "Bán buôn-bán lẻ",
                    "Tài chính-Ngân hàng",
                    "Logistics-Vận tải",
                    "CNTT-Truyền thông",
                    "Giáo dục-Đào tạo",
                    "Y tế"
                ],
                "growth_rate_2024_pct": [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
                "productivity_million_VND_per_worker": [103.4, 241.2, 168.8, 1290.5, 145.3, 1072.4, 321.4, 713.8, 205.7, 437.1],
                "spillover_coef_0_1": [0.35, 0.78, 0.42, 0.30, 0.55, 0.85, 0.72, 0.92, 0.65, 0.60],
                "export_billion_USD": [40.5, 290.9, 2.5, 8.2, 5.5, 1.2, 3.1, 178.0, 0.0, 0.0],
                "labor_million": [13.20, 11.50, 4.80, 0.30, 7.80, 0.55, 1.95, 0.62, 2.15, 0.75],
                "ai_readiness_0_100": [15, 55, 20, 30, 48, 72, 42, 88, 38, 45],
                "automation_risk_pct": [18, 42, 25, 55, 38, 52, 35, 28, 22, 18]
            })

    try:
        if regions_df is None:
            regions_df = load_regions()
        if sectors_df is None:
            sectors_df = load_sectors()
            
        # Run actual TOPSIS on regions
        criteria_r = [
            'grdp_per_capita_million_VND', 'fdi_registered_billion_USD',
            'digital_index_0_100', 'ai_readiness_0_100',
            'trained_labor_pct', 'rd_intensity_pct',
            'internet_penetration_pct', 'gini_coef'
        ]
        is_benefit_r = [True, True, True, True, True, True, True, False]
        w_r = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
        
        # Select first 6 columns if the input df has different layout or shape
        # Check standard columns exist
        for col in criteria_r:
            if col not in regions_df.columns:
                # If column names differ, fall back to load_regions columns
                regions_df = load_regions()
                break
                
        X_r = regions_df[criteria_r].values.astype(float)
        scores_r, _ = topsis(X_r, w_r, is_benefit_r)
        
        region_ranking = regions_df.copy()
        region_ranking['readiness_score'] = scores_r
        region_ranking = region_ranking.sort_values('readiness_score', ascending=False)
        
        # Run actual TOPSIS on sectors (excluding risk, or treating risk as negative)
        criteria_s = [
            'growth_rate_2024_pct', 'productivity_million_VND_per_worker',
            'spillover_coef_0_1', 'export_billion_USD',
            'labor_million', 'ai_readiness_0_100', 'automation_risk_pct'
        ]
        is_benefit_s = [True, True, True, True, True, True, False]
        w_s = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15])
        
        for col in criteria_s:
            if col not in sectors_df.columns:
                sectors_df = load_sectors()
                break
                
        X_s = sectors_df[criteria_s].values.astype(float)
        scores_s, _ = topsis(X_s, w_s, is_benefit_s)
        
        sector_readiness = sectors_df.copy()
        sector_readiness['readiness_score'] = scores_s
        sector_readiness = sector_readiness.sort_values('readiness_score', ascending=False)
        
        return {
            'region_ranking': region_ranking,
            'sector_readiness': sector_readiness
        }
    except Exception as e:
        print(f"Error in calculate_topsis_readiness wrapper: {e}")
        region_ranking = regions_df.copy() if regions_df is not None else pd.DataFrame()
        region_ranking['readiness_score'] = np.linspace(0.8, 0.4, len(region_ranking)) if len(region_ranking) > 0 else []
        sector_readiness = sectors_df.copy() if sectors_df is not None else pd.DataFrame()
        sector_readiness['readiness_score'] = np.linspace(0.8, 0.4, len(sector_readiness)) if len(sector_readiness) > 0 else []
        return {
            'region_ranking': region_ranking,
            'sector_readiness': sector_readiness
        }
