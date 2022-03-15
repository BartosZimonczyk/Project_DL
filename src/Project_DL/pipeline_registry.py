"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from Project_DL.pipelines import train_model_pipeline as tmp


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    train_model_pipeline = tmp.create_pipeline()
    return {
        "__default__": train_model_pipeline,
        "train_model_pipeline": train_model_pipeline
    }
