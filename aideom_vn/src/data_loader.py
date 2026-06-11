import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Xác định đường dẫn tương đối đến thư mục data
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

def load_macro() -> pd.DataFrame:
    """
    Nạp dữ liệu vĩ mô Việt Nam giai đoạn 2020-2025.
    
    Returns:
        pd.DataFrame: DataFrame chứa số liệu vĩ mô được sắp xếp theo năm.
    """
    file_path = DATA_DIR / 'vietnam_macro_2020_2025.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu vĩ mô tại: {file_path}")
    df = pd.read_csv(file_path)
    # Đảm bảo dữ liệu được sắp xếp tăng dần theo năm
    df = df.sort_values('year').reset_index(drop=True)
    return df

def load_sectors() -> pd.DataFrame:
    """
    Nạp dữ liệu 10 ngành kinh tế chính của Việt Nam năm 2024.
    
    Returns:
        pd.DataFrame: DataFrame chứa số liệu cấu trúc ngành.
    """
    file_path = DATA_DIR / 'vietnam_sectors_2024.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu ngành tại: {file_path}")
    return pd.read_csv(file_path)

def load_regions() -> pd.DataFrame:
    """
    Nạp dữ liệu 6 vùng kinh tế - xã hội của Việt Nam năm 2024.
    
    Returns:
        pd.DataFrame: DataFrame chứa số liệu vùng miền.
    """
    file_path = DATA_DIR / 'vietnam_regions_2024.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu vùng tại: {file_path}")
    return pd.read_csv(file_path)

def load_priorities() -> pd.DataFrame:
    """
    Nạp dữ liệu trọng số ưu tiên chính sách phát triển.
    
    Returns:
        pd.DataFrame: DataFrame chứa 3 bộ trọng số ưu tiên.
    """
    file_path = DATA_DIR / 'vietnam_priorities.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu trọng số tại: {file_path}")
    return pd.read_csv(file_path)

def validate_data() -> None:
    """
    Thực hiện kiểm tra và in ra hình dạng cũng như dòng dữ liệu đầu tiên của mỗi tệp.
    """
    print("--- BẮT ĐẦU XÁC THỰC DỮ LIỆU ---")
    
    # Kiểm tra tệp vĩ mô
    macro = load_macro()
    print(f"\nTệp vĩ mô (vietnam_macro_2020_2025.csv) - Kích thước: {macro.shape}")
    print(macro.head(2))
    
    # Kiểm tra tệp ngành
    sectors = load_sectors()
    print(f"\nTệp ngành (vietnam_sectors_2024.csv) - Kích thước: {sectors.shape}")
    print(sectors.head(2))
    
    # Kiểm tra tệp vùng miền
    regions = load_regions()
    print(f"\nTệp vùng (vietnam_regions_2024.csv) - Kích thước: {regions.shape}")
    print(regions.head(2))
    
    # Kiểm tra tệp trọng số
    priorities = load_priorities()
    print(f"\nTệp trọng số (vietnam_priorities.csv) - Kích thước: {priorities.shape}")
    print(priorities.head(2))
    
    print("\n--- XÁC THỰC DỮ LIỆU THÀNH CÔNG ---")

if __name__ == '__main__':
    # Chạy kiểm tra dữ liệu trực tiếp khi gọi module
    validate_data()
