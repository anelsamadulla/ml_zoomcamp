import pandas as pd 
df = pd.read_csv("car_fuel_efficiency.csv")
#print(df.head())
#print(df["fuel_type"].unique())
#missing_per_column = df.isna().sum()
#print(missing_per_column[missing_per_column > 0])
#missing = df.isnull().sum()
#print(missing)
# Filter Asia cars and find max fuel efficiency
max_efficiency_asia = df[df["origin"] == "Asia"]["fuel_efficiency_mpg"].max()

print("Maximum fuel efficiency of cars from Asia:", max_efficiency_asia)
