import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'aideom_vn/src'))

modules = [
    'pages.bai1_cobb_douglas',
    'pages.bai2_lp_budget',
    'pages.bai3_priority_sectors',
    'pages.bai4_lp_region_budget',
    'pages.bai5_mip_project_selection',
    'pages.bai6_topsis_ai_regions',
    'pages.bai7_multi_objective_nsga2',
    'pages.bai8_dynamic_forecast',
    'pages.bai9_labor_simulation',
    'pages.bai10_stochastic_budget',
    'pages.bai11_rl_policy',
    'pages.bai12_integrated'
]

print("Starting validation of AIDEOM-VN modules...")

for mod_name in modules:
    try:
        mod = __import__(mod_name, fromlist=['render'])
        if hasattr(mod, 'render'):
            print(f"[OK] {mod_name} has render() function.")
        else:
            print(f"[FAIL] {mod_name} missing render() function.")
    except Exception as e:
        print(f"[ERROR] {mod_name} failed to import: {e}")

print("Validation complete.")
