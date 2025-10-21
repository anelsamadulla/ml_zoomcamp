import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. Load dataset
df = pd.read_csv("course_lead_scoring.csv")

# 2. Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('NA')
    else:
        df[col] = df[col].fillna(0.0)

# 3. Split data into train/val/test (60/20/20)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)  # 0.25 * 0.8 = 0.2

y_train = df_train.converted.values
y_val = df_val.converted.values

# 4. Prepare dictionaries for one-hot encoding
dicts_train = df_train.drop('converted', axis=1).to_dict(orient='records')
dicts_val = df_val.drop('converted', axis=1).to_dict(orient='records')

# 5. Vectorize categorical data
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(dicts_train)
X_val = dv.transform(dicts_val)

# 6. Train logistic regression
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# 7. Evaluate on validation
y_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)

print(f"Validation AUC: {auc:.3f}")

