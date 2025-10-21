import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

# 1. Load dataset
df = pd.read_csv("course_lead_scoring.csv")

# 2. Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('NA')
    else:
        df[col] = df[col].fillna(0.0)

# 3. Split data 60/20/20
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_train = df_train.converted.values
y_val = df_val.converted.values

# 4. Vectorize
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_train.drop('converted', axis=1).to_dict(orient='records'))
X_val = dv.transform(df_val.drop('converted', axis=1).to_dict(orient='records'))

# 5. Train logistic regression
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# 6. Predict probabilities
y_pred = model.predict_proba(X_val)[:, 1]

# 7. Compute precision and recall for each threshold
thresholds = np.arange(0.0, 1.01, 0.01)
precisions = []
recalls = []

for t in thresholds:
    y_pred_bin = (y_pred >= t)
    precisions.append(precision_score(y_val, y_pred_bin))
    recalls.append(recall_score(y_val, y_pred_bin))

# 8. Find where they intersect
diff = np.abs(np.array(precisions) - np.array(recalls))
best_t = thresholds[np.argmin(diff)]

print(f"Precision ≈ Recall at threshold: {best_t:.3f}")

