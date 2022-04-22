"""
This is a boilerplate pipeline 'train_model_pipeline'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_dataset_node, get_model_node, get_trainer_node, train_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        load_dataset_node,
        get_model_node,
        get_trainer_node,
        train_node
    ])
