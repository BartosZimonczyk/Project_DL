import torch
from src.Project_DL.pipelines.train_model_pipeline.model import ErCaNet


model_2 = ErCaNet("Playground_2")
model_2.load_state_dict(torch.load('models/CaptionEraseBZ-GPU-TheSecond.pt'))
model_2.eval()