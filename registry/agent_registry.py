from agents.cleaning_agent import CleaningAgent
from agents.dataset_agent import DatasetAgent
from agents.evaluation_agent import EvaluationAgent
from agents.feature_agent import FeatureAgent
from agents.report_agent import ReportAgent
from agents.training_agent import TrainingAgent


def get_agent_registry():
    return {
        "dataset": DatasetAgent(),
        "cleaning": CleaningAgent(),
        "feature": FeatureAgent(),
        "training": TrainingAgent(),
        "evaluation": EvaluationAgent(),
        "report": ReportAgent(),
    }

