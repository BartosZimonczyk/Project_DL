"""
This is a boilerplate pipeline 'train_model_pipeline'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_dataset, get_model, get_trainer, get_logger, train, save_model_to_file


def create_pipeline(**kwargs) -> Pipeline:
    """Function that creates the model training pipeline

    Returns:
        Pipeline: Pipeline for training the model
    """
    return pipeline([
        node(load_dataset, inputs="params:dataset_params", outputs=["train_loader", "test_loader", "val_loader"]),
        node(get_logger, inputs="params:model_name", outputs="wandb_logger"),
        node(get_model, inputs="wandb_logger", outputs="model"),
        node(get_trainer, inputs=["wandb_logger", "params:trainer_params", "params:checkpoint_path"], outputs="trainer"),
        node(train, inputs=["trainer", "model", "train_loader", "test_loader"], outputs=None),
        node(save_model_to_file, inputs=["model", "wandb_logger", "params:model_save_path"], outputs=None)
    ])
