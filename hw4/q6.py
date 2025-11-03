import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_csv('course_lead_scoring.csv')
df = df.dropna()

# Target
y = df.converted.values

# Convert categorical features into numeric
X = df.drop('converted', axis=1)
X = pd.get_dummies(X, drop_first=True)  # one-hot encoding

# Define C values
C_values = [0.000001, 0.001, 1]

# Set up KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

results = {}

for C in C_values:
    scores = []
    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)
    
    results[C] = (np.mean(scores), np.std(scores))
    print(f"C={C}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")

# Find best C
best_C = max(results, key=lambda c: (results[c][0], -results[c][1], -c))
print(f"Best C: {best_C}")

