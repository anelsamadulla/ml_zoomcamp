import pickle

with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

x = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
}

pred = model.predict_proba([x])[0][1]
print(pred)

