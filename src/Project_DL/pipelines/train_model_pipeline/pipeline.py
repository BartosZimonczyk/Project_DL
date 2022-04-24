"""
This is a boilerplate pipeline 'train_model_pipeline'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_dataset, get_model, get_trainer, get_logger, train, save_model_to_file


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(load_dataset, inputs=None, outputs=["train_loader", "test_loader", "val_loader"]),
        node(get_logger, inputs=None, outputs="wandb_logger"),
        node(get_model, inputs="wandb_logger", outputs="model"),
        node(get_trainer, inputs="wandb_logger", outputs="trainer"),
        node(train, inputs=["trainer", "model", "train_loader", "test_loader"], outputs=None),
        node(save_model_to_file, inputs=["model", "wandb_logger"], outputs=None)
    ])
