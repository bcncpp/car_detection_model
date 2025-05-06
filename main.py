from fastapi import FastAPI
import torch
from models import CarDetection
from pydantic import BaseModel
import numpy as np
from typing import Optional

class ArrayModel(BaseModel):
    array: list[list[float]]
    def to_numpy(self):
        return np.array(self.array)

class InferenceRequest(BaseModel):
    mode: ArrayModel
    bounded_boxes: list[tuple[float, float, float]]
    confidence_scores: list[float]
    labels: list[str]
    label_colors: Optional[dict[str, str]]

class InferenceResponse(BaseModel):
    status: bool
    output: list[float]

app = FastAPI()

@app.on_event("startup")
async def load_model():
    app.state.model = CarDetection()
    app.state.model.load_state_dict(torch.load("cardection_model.pth", map_location=torch.device('cpu')))
    app.state.model.eval()

@app.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest):
    # Convert input to tensor
    np_input = req.mode.to_numpy()
    tensor_input = torch.tensor(np_input, dtype=torch.float32).unsqueeze(0)  # Add batch dim if needed

    with torch.no_grad():
        output = app.state.model(tensor_input)

    return output
