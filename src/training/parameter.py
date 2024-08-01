from dataclasses import dataclass, field
from typing import List

from classify.LSTMTransformerClassify.config import config_lstm_transformer_model_classify
from classify.SimpleLSTMClassify.config import config_lstm_classify
from models.CNNLSTM.config import config_CNNLSTM
from models.LSTMTransformer.config import config_lstm_transformer_modelV1
import torch.nn as nn

from models.SimpleLSTM_v2_1.config import config_simple_lstm_v2_1
from models.SimpleLSTM_v2_2.config import config_simple_lstm_v2_2


@dataclass
class ModelParams:
    input_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int


@dataclass
class TrainingParams:
    batch_size: int
    learning_rate: int
    num_epochs: int
    model_save_path: str


@dataclass
class DataParams:
    feature_columns: List[int] = field(default_factory=list)
    target_column: str = "stock_close"


@dataclass
class ModelConfig:
    model_params: ModelParams
    training_params: TrainingParams
    data_params: DataParams
    Model: nn.Module
    data: str # data version


MODEL_CONFIGS = {
    "example": ModelConfig(
        model_params=ModelParams(input_dim=48, hidden_dim=512, num_layers=3, num_heads=16),
        training_params=TrainingParams(batch_size=64, learning_rate=1e-3, num_epochs=10000,
                                       model_save_path="model_files/lstm_transformer_model_a.pth"),
        data_params=DataParams(feature_columns=[i for i in range(48)], target_column="stock_close"),
        Model = nn.Module,
        data = "v1"
    ),
}


def load_config(cfg, name: str):
    MODEL_CONFIGS[name] = ModelConfig(
        model_params=ModelParams(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
                                 num_heads=cfg["num_heads"]),
        training_params=TrainingParams(batch_size=cfg["batch_size"], learning_rate=cfg["learning_rate"],
                                       num_epochs=cfg["num_epochs"],
                                       model_save_path=cfg["model_save_path"]),
        data_params=DataParams(feature_columns=cfg["feature_columns"], target_column=cfg["target_column"]),
        Model = cfg["model"],
        data = cfg["data"]
    )


load_config(config_lstm_transformer_modelV1, "v1")
load_config(config_CNNLSTM, "cnn_lstm")
load_config(config_simple_lstm_v2_1, "simple_lstm_v2_1")
load_config(config_simple_lstm_v2_2, "simple_lstm_v2_2")
load_config(config_lstm_transformer_model_classify, "v1_classify")
load_config(config_lstm_classify, "simple_lstm_classify")





def get_config(model_name: str) -> ModelConfig:
    """
    根据模型名称获取配置参数。

    Args:
        model_name (str): 模型名称。

    Returns:
        ModelConfig: 模型配置对象。
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model name: {model_name}")
    return MODEL_CONFIGS[model_name]
