import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


df = pd.read_csv('car_fuel_efficiency.csv')

# оставляем только нужные колонки
cols = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']
df = df[cols]

df.head()

df['fuel_efficiency_mpg'].hist(bins=50)
#plt.show()

df['fuel_efficiency_mpg'].describe()

#Question 1

df.isnull().sum()

#Question 2

df['horsepower'].median()

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

#Question 3

#split 60/20/20
n = len(df_shuffled)
n_train = int(0.6 * n)
n_val = int(0.2 * n)
n_test = n - n_train - n_val

df_train = df_shuffled.iloc[:n_train]
df_val = df_shuffled.iloc[n_train:n_train+n_val]
df_test = df_shuffled.iloc[n_train+n_val:]

def rmse(y_true, y_pred):
	return np.sqrt(mean_squared_error(y_true, y_pred))

def train_and_evaluate(df_train, df_val, fill_value):
	X_train = df_train.drop('fuel_efficiency_mpg', axis=1)
	y_train = df_train['fuel_efficiency_mpg']
	X_val = df_val.drop('fuel_efficiency_mpg', axis=1)
	y_val = df_val['fuel_efficiency_mpg']

	X_train = X_train.fillna(fill_value)
	X_val = X_val.fillna(fill_value)

	model = LinearRegression()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_val)
	return rmse(y_val, y_pred)

rmse_zero = train_and_evaluate(df_train, df_val, 0)
rmse_mean = train_and_evaluate(df_train, df_val, df_train.median())

#print(round(rmse_zero, 2), round(rmse_mean, 2))

#Question 4

scores = {}
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
	model = Ridge(alpha=r)
	X_train = df_train.drop('fuel_efficiency_mpg', axis=1).fillna(0)
	y_train = df_train['fuel_efficiency_mpg']
	X_val = df_val.drop('fuel_efficiency_mpg', axis=1).fillna(0)
	y_val = df_val['fuel_efficiency_mpg']

	model.fit(X_train, y_train)
	y_pred = model.predict(X_val)
	scores[r] = round(rmse(y_val, y_pred), 2)

scores
#print(scores)

#Question 5

rmses = []
for s in range(10):
    df_shuffled = df.sample(frac=1, random_state=s).reset_index(drop=True)
    n = len(df_shuffled)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    n_test = n - n_train - n_val
    
    df_train = df_shuffled.iloc[:n_train]
    df_val = df_shuffled.iloc[n_train:n_train+n_val]
    
    model = LinearRegression()
    X_train = df_train.drop('fuel_efficiency_mpg', axis=1).fillna(0)
    y_train = df_train['fuel_efficiency_mpg']
    X_val = df_val.drop('fuel_efficiency_mpg', axis=1).fillna(0)
    y_val = df_val['fuel_efficiency_mpg']
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmses.append(rmse(y_val, y_pred))

std = np.std(rmses)
print(round(std, 3))

#Question 6
# seed = 9
df_shuffled = df.sample(frac=1, random_state=9).reset_index(drop=True)
n = len(df_shuffled)
n_train = int(0.6 * n)
n_val = int(0.2 * n)
n_test = n - n_train - n_val

df_train = df_shuffled.iloc[:n_train]
df_val = df_shuffled.iloc[n_train:n_train+n_val]
df_test = df_shuffled.iloc[n_train+n_val:]

# объединяем train и val
df_full_train = pd.concat([df_train, df_val])

# готовим данные
X_train = df_full_train.drop('fuel_efficiency_mpg', axis=1).fillna(0)
y_train = df_full_train['fuel_efficiency_mpg']
X_test = df_test.drop('fuel_efficiency_mpg', axis=1).fillna(0)
y_test = df_test['fuel_efficiency_mpg']

model = Ridge(alpha=0.001)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(round(rmse(y_test, y_pred), 3))

print("std (full):", std)            # печатает неокруглённое значение
print(f"std (6 decimals): {std:.6f}")
print("RMSEs:")
for i, v in enumerate(rmses):
    print(i, v)

