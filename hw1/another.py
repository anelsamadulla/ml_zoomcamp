import pandas as pd

df = pd.read_csv("car_fuel_efficiency.csv")

# 1. Median horsepower before filling
median_before = df["horsepower"].median()
print("Median before:", median_before)

# 2. Most frequent value (mode)
most_freq = df["horsepower"].mode()[0]
print("Most frequent value:", most_freq)

# 3. Fill missing values with most frequent
df["horsepower"].fillna(most_freq, inplace=True)

# 4. Median horsepower after filling
median_after = df["horsepower"].median()
print("Median after:", median_after)

# 5. Compare
if median_after > median_before:
    print("Yes, it increased")
elif median_after < median_before:
    print("Yes, it decreased")
else:
    print("No")

