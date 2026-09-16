import os
os.environ['SKLEARN_N_JOBS']='1'; os.environ['OMP_NUM_THREADS']='4'
import sys, json, collections
sys.path[:0] = ['experiment','preprocessing']
import pandas as pd
from config import OUTPUT_PATH
from feature_engineering import (create_spatial_split_indices, prepare_xgboost_inputs,
                                 train_xgboost_and_select_features)
df = pd.read_csv(OUTPUT_PATH) if str(OUTPUT_PATH).endswith('.csv') else pd.read_excel(OUTPUT_PATH)
print("rows", len(df))
regions = sorted(df['Region'].dropna().unique())
sel = {}
for r in regions:
    sp = create_spatial_split_indices(df, test_region=r, val_ratio=0.2, random_seed=42)
    _, X_full, y_full = prepare_xgboost_inputs(df, fit_idx=sp['train_idx'])
    _, _, X_selected = train_xgboost_and_select_features(X_full, y_full, fit_idx=sp['train_idx'])
    sel[r] = list(X_selected.columns)
    print(f"{r:24s} {sel[r]}")
cnt = collections.Counter(f for v in sel.values() for f in v)
print("\n--- stability over %d folds ---" % len(regions))
for f,c in cnt.most_common(): print(f"  {f:32s} {c}/{len(regions)}")
print("\nunion size:", len(cnt), " always-selected:", sum(1 for c in cnt.values() if c==len(regions)))
json.dump(sel, open(os.path.join(os.path.dirname(__file__),'feat_stability.json'),'w'), indent=1)
