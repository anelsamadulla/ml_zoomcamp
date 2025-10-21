# 0. Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
# adjust path if needed (here assumes the file is in the current working directory)
df = pd.read_csv("course_lead_scoring.csv")

# 2. Data preparation: fill missing values
# identify categorical and numerical columns explicitly from dataset description
categorical_cols = ['lead_source', 'industry', 'employment_status', 'location']
numeric_cols = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']

# Fill categorical missing with 'NA' and numeric missing with 0.0
df[categorical_cols] = df[categorical_cols].fillna('NA')
df[numeric_cols] = df[numeric_cols].fillna(0.0)

# Make sure numeric columns are numeric
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

# Show a quick check
print("Missing values after filling:")
print(df.isnull().sum())
print("\nData sample:")
print(df.head())

# ----------------------------
# Question 1: Mode for industry
# ----------------------------
mode_industry = df['industry'].mode().iloc[0]
print("\nQ1: Mode of industry ->", mode_industry)  # choose from provided options

# ----------------------------
# Question 2: Biggest correlation
# ----------------------------
corr = df[numeric_cols].corr()
print("\nCorrelation matrix (numeric features):")
print(corr)

# Pairs to compare:
pairs = [
    ('interaction_count','lead_score'),
    ('number_of_courses_viewed','lead_score'),
    ('number_of_courses_viewed','interaction_count'),
    ('annual_income','interaction_count'),
]

print("\nQ2: correlations for the listed pairs:")
for a,b in pairs:
    print(f"{a} & {b}: {corr.loc[a,b]:.4f}")

# compute the pair with largest absolute correlation among the listed ones
best_pair = max(pairs, key=lambda p: abs(corr.loc[p[0], p[1]]))
print("Q2: Biggest correlation pair ->", best_pair[0], "and", best_pair[1])

# ----------------------------
# Split data: 60% train / 20% val / 20% test (seed=42)
# ----------------------------
# First split out test (20%)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Then split df_full_train into train (60% total) and val (20% total)
# df_full_train currently is 80% of data; we want train to be 60% total => train is 75% of df_full_train
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42, shuffle=True)

# Extract target and drop from features
y_train = df_train['converted'].astype(int).reset_index(drop=True)
y_val = df_val['converted'].astype(int).reset_index(drop=True)
y_test = df_test['converted'].astype(int).reset_index(drop=True)

X_train = df_train.drop(columns=['converted']).reset_index(drop=True)
X_val = df_val.drop(columns=['converted']).reset_index(drop=True)
X_test = df_test.drop(columns=['converted']).reset_index(drop=True)

print("\nSizes: train, val, test ->", X_train.shape[0], X_val.shape[0], X_test.shape[0])

# ----------------------------
# Question 3: Mutual Information (categorical features)
# Use the training set only
# ----------------------------
# Encode categorical features to integers using OrdinalEncoder for MI calculation
ord_enc = OrdinalEncoder()
X_train_cat = X_train[categorical_cols].copy()
X_train_cat_enc = ord_enc.fit_transform(X_train_cat)

mi_scores = mutual_info_classif(X_train_cat_enc, y_train, discrete_features=True, random_state=42)
mi_dict = {col: round(score, 2) for col, score in zip(categorical_cols, mi_scores)}

print("\nQ3: Mutual information scores (train set), rounded to 2 decimals:")
for k,v in mi_dict.items():
    print(k, "->", v)

# find the categorical variable with the largest MI
mi_best = max(mi_dict.items(), key=lambda kv: kv[1])
print("Q3: Biggest MI ->", mi_best[0], "with MI =", mi_best[1])

# ----------------------------
# Prepare function to build feature matrices with OneHotEncoder fitted on training set
# ----------------------------
def build_feature_matrix(X_train_df, X_val_df, categorical_feats, numeric_feats):
    """
    Fits OneHotEncoder on X_train_df[categorical_feats] and transforms both train and val.
    Returns X_train_enc (pd.DataFrame), X_val_enc (pd.DataFrame), and the fitted encoder.
    Robust to scikit-learn versions (sparse vs sparse_output, get_feature_names vs get_feature_names_out).
    """
    # Try to construct the encoder with either parameter name that exists in the installed sklearn
    try:
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        # newer sklearn uses sparse_output
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    enc.fit(X_train_df[categorical_feats])

    X_train_cat = enc.transform(X_train_df[categorical_feats])
    X_val_cat = enc.transform(X_val_df[categorical_feats])

    # create column names for dummies in a version-robust way
    try:
        cat_names = enc.get_feature_names_out(categorical_feats)
    except AttributeError:
        # older sklearn
        cat_names = enc.get_feature_names(categorical_feats)

    X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_names, index=X_train_df.index)
    X_val_cat_df = pd.DataFrame(X_val_cat, columns=cat_names, index=X_val_df.index)

    X_train_num_df = X_train_df[numeric_feats].reset_index(drop=True)
    X_val_num_df = X_val_df[numeric_feats].reset_index(drop=True)

    X_train_final = pd.concat([X_train_cat_df.reset_index(drop=True), X_train_num_df.reset_index(drop=True)], axis=1)
    X_val_final = pd.concat([X_val_cat_df.reset_index(drop=True), X_val_num_df.reset_index(drop=True)], axis=1)

    return X_train_final, X_val_final, enc


# ----------------------------
# Question 4: Logistic Regression accuracy
# ----------------------------
Xtr, Xv, fitted_enc = build_feature_matrix(X_train, X_val, categorical_cols, numeric_cols)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(Xtr, y_train)
y_val_pred = model.predict(Xv)
acc_val = accuracy_score(y_val, y_val_pred)
print("\nQ4: Validation accuracy ->", round(acc_val, 2))

# ----------------------------
# Question 5: Feature elimination (test exclusion of specific features)
# Features to test: 'industry', 'employment_status', 'lead_score'
# We will remove each and measure the change in validation accuracy
# ----------------------------
original_acc = acc_val
features_to_test = ['industry', 'employment_status', 'lead_score']
diffs = {}

for feat in features_to_test:
    # determine which categorical and numeric features to use when removing feat
    cat_feats = [c for c in categorical_cols if not (feat == c)]
    num_feats = [n for n in numeric_cols if not (feat == n)]
    
    # Build matrices (fit encoder on training subset after dropping feature)
    Xtr_tmp, Xv_tmp, _ = build_feature_matrix(X_train[cat_feats + num_feats], X_val[cat_feats + num_feats], cat_feats, num_feats)
    
    # fit model
    mdl = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    mdl.fit(Xtr_tmp, y_train)
    yv_pred_tmp = mdl.predict(Xv_tmp)
    acc_tmp = accuracy_score(y_val, yv_pred_tmp)
    
    diff = original_acc - acc_tmp
    diffs[feat] = diff
    print(f"\nRemoved feature: {feat}")
    print("Validation accuracy without this feature:", acc_tmp)
    print("Difference (original - without):", diff)

# find smallest difference (least useful feature)
least_useful_feature = min(diffs.items(), key=lambda kv: kv[1])[0]
print("\nQ5: Feature with smallest difference (least useful) ->", least_useful_feature)

# ----------------------------
# Question 6: Parameter tuning for C in [0.01, 0.1, 1, 10, 100]
# ----------------------------
C_values = [0.01, 0.1, 1, 10, 100]
results = {}
for c in C_values:
    mdl = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
    mdl.fit(Xtr, y_train)
    preds = mdl.predict(Xv)
    acc = accuracy_score(y_val, preds)
    results[c] = round(acc, 3)
    print(f"C={c} -> val accuracy = {results[c]}")

# choose best accuracy and if tie choose smallest C
best_acc = max(results.values())
best_cs = [c for c, acc in results.items() if acc == best_acc]
best_c = min(best_cs)
print("\nQ6: Best C (smallest in tie) ->", best_c, "with accuracy", best_acc)

# ----------------------------
# Final summary print (answers)
# ----------------------------
print("\n=== FINAL ANSWERS (from code run) ===")
print("Q1 (mode industry):", mode_industry)
print("Q2 (biggest correlation pair):", best_pair[0], "and", best_pair[1])
print("Q3 (biggest MI):", mi_best[0], "->", mi_best[1])
print("Q4 (val accuracy rounded 2 decimals):", round(acc_val, 2))
print("Q5 (least useful feature):", least_useful_feature)
print("Q6 (best C):", best_c, "-> accuracy (3 decimals):", best_acc)

