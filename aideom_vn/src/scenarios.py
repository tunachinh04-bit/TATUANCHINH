"""
scenarios.py - Quản lý 5 kịch bản chính sách
Bài 12 - AIDEOM-VN
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PolicyScenario:
    """Mô tả một kịch bản chính sách"""
    code: str  # S1, S2, ..., S5
    name_vi: str
    name_en: str
    description: str
    allocation: Dict[str, float]  # {'K': 0.7, 'D': 0.1, 'AI': 0.1, 'H': 0.1}
    color: str
    
    def to_dict(self):
        return {
            'code': self.code,
            'name_vi': self.name_vi,
            'allocation': self.allocation,
            'description': self.description
        }

# ==================== SCENARIOS DEFINITION ====================

SCENARIOS = {
    'S1': PolicyScenario(
        code='S1',
        name_vi='Truyền thống (Traditional)',
        name_en='Traditional Growth',
        description='Tập trung vốn vật chất, FDI, hạ tầng truyền thống, xuất khẩu',
        allocation={'K': 0.70, 'D': 0.10, 'AI': 0.10, 'H': 0.10},
        color='#636EFA'
    ),
    'S2': PolicyScenario(
        code='S2',
        name_vi='Số hóa nhanh (Digital First)',
        name_en='Digital Acceleration',
        description='Tăng đầu tư chính phủ số, doanh nghiệp số, thanh toán số',
        allocation={'K': 0.25, 'D': 0.45, 'AI': 0.15, 'H': 0.15},
        color='#EF553B'
    ),
    'S3': PolicyScenario(
        code='S3',
        name_vi='AI dẫn dắt (AI-Led)',
        name_en='AI-Centric Innovation',
        description='Ưu tiên AI, dữ liệu lớn, bán dẫn, trung tâm dữ liệu',
        allocation={'K': 0.20, 'D': 0.20, 'AI': 0.45, 'H': 0.15},
        color='#00CC96'
    ),
    'S4': PolicyScenario(
        code='S4',
        name_vi='Bao trùm số (Inclusive Digital)',
        name_en='Inclusive Growth',
        description='Ưu tiên vùng yếu, SME, giáo dục số, nông nghiệp số',
        allocation={'K': 0.30, 'D': 0.20, 'AI': 0.10, 'H': 0.40},
        color='#AB63FA'
    ),
    'S5': PolicyScenario(
        code='S5',
        name_vi='Tối ưu cân bằng (Balanced Optimized)',
        name_en='Balanced Optimization',
        description='Kết quả cân bằng từ NSGA-II + Stochastic Programming',
        allocation={'K': 0.35, 'D': 0.25, 'AI': 0.20, 'H': 0.20},
        color='#FFA15A'
    )
}

# ==================== HELPER FUNCTIONS ====================

def get_scenario(code: str) -> PolicyScenario:
    """Lấy thông tin kịch bản"""
    return SCENARIOS.get(code, SCENARIOS['S1'])

def get_all_scenarios() -> List[PolicyScenario]:
    """Lấy danh sách tất cả kịch bản"""
    return list(SCENARIOS.values())

def get_allocation(code: str) -> Dict[str, float]:
    """Lấy phân bổ ngân sách của một kịch bản"""
    scenario = get_scenario(code)
    return scenario.allocation.copy()

def get_allocation_billion(code: str, total_budget: float = 80000) -> Dict[str, float]:
    """Lấy phân bổ ngân sách theo tỷ đồng (mặc định 80 triệu tỷ)"""
    alloc_pct = get_allocation(code)
    return {k: v * total_budget for k, v in alloc_pct.items()}

def scenario_to_dataframe() -> pd.DataFrame:
    """Tạo bảng so sánh tất cả kịch bản"""
    data = []
    for code, scenario in SCENARIOS.items():
        row = {
            'Kịch bản': code,
            'Tên': scenario.name_vi,
            'K (%)': scenario.allocation['K'] * 100,
            'D (%)': scenario.allocation['D'] * 100,
            'AI (%)': scenario.allocation['AI'] * 100,
            'H (%)': scenario.allocation['H'] * 100,
            'Mô tả': scenario.description
        }
        data.append(row)
    
    return pd.DataFrame(data)

def estimate_weights(scenario_code: str) -> Dict[str, float]:
    """
    Ước lượng trọng số chính sách dựa trên kịch bản.
    
    - S1 (Truyền thống): w = (0.40 tăng trưởng, 0.25 bao trùm, 0.20 môi trường, 0.15 an ninh)
    - S2 (Số hóa): Tăng trọng số môi trường lên (0.40 TG, 0.20 BT, 0.25 MTS, 0.15 AN)
    - S3 (AI): Tăng trọng số an ninh (0.40 TG, 0.20 BT, 0.15 MTS, 0.25 AN)
    - S4 (Bao trùm): Tăng trọng số bao trùm (0.30 TG, 0.40 BT, 0.20 MTS, 0.10 AN)
    - S5 (Tối ưu): Cân bằng (0.35 TG, 0.30 BT, 0.20 MTS, 0.15 AN)
    """
    weights = {
        'S1': {'growth': 0.40, 'inclusive': 0.25, 'environment': 0.20, 'security': 0.15},
        'S2': {'growth': 0.40, 'inclusive': 0.20, 'environment': 0.25, 'security': 0.15},
        'S3': {'growth': 0.40, 'inclusive': 0.20, 'environment': 0.15, 'security': 0.25},
        'S4': {'growth': 0.30, 'inclusive': 0.40, 'environment': 0.20, 'security': 0.10},
        'S5': {'growth': 0.35, 'inclusive': 0.30, 'environment': 0.20, 'security': 0.15},
    }
    return weights.get(scenario_code, weights['S1'])

def scenario_description_full(code: str) -> str:
    """Lấy mô tả đầy đủ của kịch bản"""
    scenario = get_scenario(code)
    
    descriptions = {
        'S1': f"""
        **Kịch bản {code}: Truyền thống**
        
        {scenario.description}
        
        **Phân bổ ngân sách (5 năm 2026-2030):**
        - Hạ tầng vật chất (K): 70% = 56 triệu tỷ VND
        - Chuyển đổi số (D): 10% = 8 triệu tỷ VND
        - Trí tuệ nhân tạo (AI): 10% = 8 triệu tỷ VND
        - Vốn nhân lực (H): 10% = 8 triệu tỷ VND
        
        **Đặc điểm:**
        - Dự báo GDP 2030: ~450-470 tỷ USD
        - Rủi ro công nghệ: Cao (Digital Divide mở rộng)
        - Tác động lao động: Dương nhẹ (+50-100k việc)
        - Tính linh hoạt: Thấp
        """,
        'S2': f"""
        **Kịch bản {code}: Số hóa Nhanh**
        
        {scenario.description}
        
        **Phân bổ ngân sách (5 năm 2026-2030):**
        - Hạ tầng vật chất (K): 25% = 20 triệu tỷ VND
        - Chuyển đổi số (D): 45% = 36 triệu tỷ VND
        - Trí tuệ nhân tạo (AI): 15% = 12 triệu tỷ VND
        - Vốn nhân lực (H): 15% = 12 triệu tỷ VND
        
        **Đặc điểm:**
        - Dự báo GDP 2030: ~460-490 tỷ USD
        - Rủi ro công nghệ: Trung bình (cần bảo mật dữ liệu)
        - Tác động lao động: Tích cực (+150-200k việc)
        - Tính linh hoạt: Cao
        """,
        'S3': f"""
        **Kịch bản {code}: AI Dẫn Dắt**
        
        {scenario.description}
        
        **Phân bổ ngân sách (5 năm 2026-2030):**
        - Hạ tầng vật chất (K): 20% = 16 triệu tỷ VND
        - Chuyển đổi số (D): 20% = 16 triệu tỷ VND
        - Trí tuệ nhân tạo (AI): 45% = 36 triệu tỷ VND
        - Vốn nhân lực (H): 15% = 12 triệu tỷ VND
        
        **Đặc điểm:**
        - Dự báo GDP 2030: ~470-520 tỷ USD (cao nhất)
        - Rủi ro công nghệ: Cao (phụ thuộc AI)
        - Tác động lao động: Biến động cao (+100-300k)
        - Tính linh hoạt: Trung bình (cạn kiệt nhân lực AI)
        """,
        'S4': f"""
        **Kịch bản {code}: Bao Trùm Số**
        
        {scenario.description}
        
        **Phân bổ ngân sách (5 năm 2026-2030):**
        - Hạ tầng vật chất (K): 30% = 24 triệu tỷ VND
        - Chuyển đổi số (D): 20% = 16 triệu tỷ VND
        - Trí tuệ nhân tạo (AI): 10% = 8 triệu tỷ VND
        - Vốn nhân lực (H): 40% = 32 triệu tỷ VND
        
        **Đặc điểm:**
        - Dự báo GDP 2030: ~440-470 tỷ USD (thấp hơn)
        - Rủi ro công nghệ: Thấp (đầu tư vào con người)
        - Tác động lao động: Tích cực cao (+200-250k)
        - Tính linh hoạt: Rất cao
        - Bất bình đẳng vùng: Giảm
        """,
        'S5': f"""
        **Kịch bản {code}: Tối Ưu Cân Bằng**
        
        {scenario.description}
        
        **Phân bổ ngân sách (5 năm 2026-2030):**
        - Hạ tầng vật chất (K): 35% = 28 triệu tỷ VND
        - Chuyển đổi số (D): 25% = 20 triệu tỷ VND
        - Trí tuệ nhân tạo (AI): 20% = 16 triệu tỷ VND
        - Vốn nhân lực (H): 20% = 16 triệu tỷ VND
        
        **Đặc điểm:**
        - Dự báo GDP 2030: ~455-495 tỷ USD (cân bằng)
        - Rủi ro công nghệ: Thấp (phân tán)
        - Tác động lao động: Ổn định (+150-180k)
        - Tính linh hoạt: Cao
        - Cân bằng mục tiêu: Tốt
        - Nguồn: Kết quả tối ưu từ NSGA-II + Stochastic Programming
        """
    }
    
    return descriptions.get(code, "")

# ==================== TESTING ====================
if __name__ == '__main__':
    print("=== 5 Kịch bản chính sách ===\n")
    print(scenario_to_dataframe().to_string(index=False))
    
    print("\n\n=== Chi tiết Kịch bản S5 ===")
    print(scenario_description_full('S5'))
    
    print("\n\n=== Phân bổ 80 triệu tỷ cho S3 ===")
    alloc = get_allocation_billion('S3', 80000)
    for k, v in alloc.items():
        print(f"{k}: {v:.1f} triệu tỷ VND")
