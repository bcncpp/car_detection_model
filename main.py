from fastapi import FastAPI
import torch
from models import CarDetection
from pydantic import BaseModel

app = FastAPI()
@app.on_event("startup")
async def load_model():
    app.state.model = CarDetection()
    app.state.model.load_state_dict(torch.load("cardection_model.pth"))
    app.state.model.eval()


@app.post("/predict")
async def predict():
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = app.state.model(dummy_input)
    return {"output": output.tolist()}
