# q1_numeric_auc.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

CSV = "course_lead_scoring.csv"  # если имя другое — поменяй

df = pd.read_csv(CSV)

# target expected to be 'converted' (0/1). If it's 'yes'/'no', map it.
if df['converted'].dtype == object:
    df['converted'] = df['converted'].map({'yes':1, 'no':0}).fillna(df['converted'])

# Fill missing values: categorical -> 'NA', numeric -> 0.0
for col in df.columns:
    if df[col].dtype == 'O':
        df[col] = df[col].fillna('NA')
    else:
        df[col] = df[col].fillna(0.0)

# split 60/20/20 with random_state=1
df_full = df.copy()
df_trainval, df_test = train_test_split(df_full, test_size=0.2, random_state=1, stratify=df_full['converted'])
df_train, df_val = train_test_split(df_trainval, test_size=0.25, random_state=1, stratify=df_trainval['converted'])
print("Sizes: train={}, val={}, test={}".format(len(df_train), len(df_val), len(df_test)))

# Numeric candidates (as per homework)
candidates = ['lead_score', 'number_of_courses_viewed', 'interaction_count', 'annual_income']

# Check presence
present = [c for c in candidates if c in df_train.columns]
print("Present numeric candidates:", present)
if len(present) < len(candidates):
    print("WARNING: some candidate columns not found in CSV. Proceeding with those present.\n")

y = df_train['converted'].values
aucs = {}
for c in present:
    x = df_train[c].astype(float).values
    # if constant or invalid, roc_auc_score will fail -> handle
    try:
        auc = roc_auc_score(y, x)
    except Exception as e:
        auc = np.nan
    inverted = False
    if not np.isnan(auc) and auc < 0.5:
        try:
            auc = roc_auc_score(y, -x)
            inverted = True
        except Exception:
            pass
    aucs[c] = (auc, inverted)

print("\nAUCs (on TRAIN):")
for k,(v,inv) in aucs.items():
    print(f"  {k}: {np.round(v,3) if not np.isnan(v) else 'nan'}", "(inverted)" if inv else "")

# choose best (ignore nan)
valid = [(k,v) for k,(v,_) in aucs.items() if not np.isnan(v)]
if valid:
    best = max(valid, key=lambda kv: kv[1])
    print("\nQUESTION 1 — best numeric by AUC:", best[0], "with AUC =", round(best[1],3))
else:
    print("\nQUESTION 1 — no valid numeric AUCs computed.")

