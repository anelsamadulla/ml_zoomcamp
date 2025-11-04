import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv('car_fuel_efficiency.csv')

# Заполняем пропуски нулями
df = df.fillna(0)

# Целевая переменная
y = df['fuel_efficiency_mpg']

# Объясняющие переменные
X = df.drop('fuel_efficiency_mpg', axis=1)

# train/val/test 60/20/20
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1) # 0.25 * 0.8 = 0.2

# Separate target
y_train = df_train['fuel_efficiency_mpg']
y_val = df_val['fuel_efficiency_mpg']

# Remove target from df
X_train = df_train.drop('fuel_efficiency_mpg', axis=1)
X_val = df_val.drop('fuel_efficiency_mpg', axis=1)

dv = DictVectorizer(sparse=True)

train_dict = X_train.to_dict(orient='records')
val_dict = X_val.to_dict(orient='records')

X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)

dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)

# Чтобы увидеть, по какому признаку сплит
print(dv.feature_names_[dt.tree_.feature[0]])

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

n_list = range(10, 201, 10)
rmse_scores = []

for n in n_list:
    rf = RandomForestRegressor(
        n_estimators=n,
        random_state=1,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred)**0.5
    
    rmse_scores.append((n, rmse))
    print(f"n_estimators={n}, RMSE={rmse:.6f}")

# find the best
best_n, best_rmse = min(rmse_scores, key=lambda x: x[1])
print("\nBest n_estimators:", best_n, "RMSE:", best_rmse)

from xgboost import XGBRegressor

etas = [0.01, 0.05, 0.1, 0.3]
best_rmse = float("inf")
best_eta = None

for eta in etas:
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=eta,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mean_squared_error(y_val, y_pred)**0.5
    print(f"eta={eta}, RMSE={rmse:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_eta = eta

print(f"\nBest eta: {best_eta}, RMSE: {best_rmse}")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

max_depth_values = [10, 15, 20, 25]
n_estimators_values = range(10, 201, 10)

results = {}

for max_depth in max_depth_values:
    rmses = []
    for n in n_estimators_values:
        rf = RandomForestRegressor(
            n_estimators=n,
            max_depth=max_depth,
            random_state=1,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmses.append(rmse)
        print(f"max_depth={max_depth}, n_estimators={n}, RMSE={rmse:.6f}")
    
    mean_rmse = np.mean(rmses)
    results[max_depth] = mean_rmse
    print(f"Mean RMSE for max_depth={max_depth}: {mean_rmse:.6f}\n")

best_max_depth = min(results, key=results.get)
print("RESULTS:", results)
print(f"Best max_depth: {best_max_depth}, Mean RMSE: {results[best_max_depth]:.6f}")

