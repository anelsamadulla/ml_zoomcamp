import pickle
from fastapi import FastAPI

with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.post("/predict")
def predict(client: dict):
    prob = model.predict_proba([client])[0][1]
    return {"conversion_probability": prob}

