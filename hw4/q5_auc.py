import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. Load data
df = pd.read_csv("course_lead_scoring.csv")

# 2. Fill missing values
for c in df.columns:
    if df[c].dtype == 'object':
        df[c] = df[c].fillna('NA')
    else:
        df[c] = df[c].fillna(0)

# 3. Separate target
y = df['converted']
del df['converted']

# 4. Prepare KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

# 5. Collect AUCs
scores = []

for train_idx, val_idx in kfold.split(df):
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # DictVectorizer for encoding
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train.to_dict(orient='records'))
    X_val = dv.transform(df_val.to_dict(orient='records'))

    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print("AUCs:", [round(s, 3) for s in scores])
print("Mean:", round(sum(scores)/len(scores), 3))
print("Std:", round(pd.Series(scores).std(), 3))

