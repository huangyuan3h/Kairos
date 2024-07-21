import torch
import pandas as pd

from models.LSTMTransformer import load_model
from models.LSTMTransformer.LSTMTransformerModel import LSTMAttentionTransformer
from models.standardize.FeatureStandardScaler import FeatureStandardScaler
from models.standardize.TargetStandardScaler import TargetStandardScaler
from src.training.parameter import get_config


class ModelPredictor:
    def __init__(self, version="v1"):
        config = get_config(version)
        # 获取模型参数
        mp = config.model_params
        tp = config.training_params
        Model = config.Model

        # 初始化特征和目标标准化器
        feature_scaler = FeatureStandardScaler()
        feature_scaler.load_scaler()
        target_scaler = TargetStandardScaler()
        target_scaler.load_scaler()

        self.model = Model(mp.input_dim, mp.hidden_dim, mp.num_layers, mp.num_heads)
        load_model(self.model, tp.model_save_path)
        self.model.eval()
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

    def preprocess_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        对输入数据进行特征标准化。

        Args:
            df (pd.DataFrame): 输入数据的 DataFrame。

        Returns:
            torch.Tensor: 标准化后的特征张量。
        """
        scaled_df = self.feature_scaler.transform(df)
        return torch.tensor(scaled_df).float().to(self.device)

    def postprocess_predictions(self, predictions: torch.Tensor) -> list:
        """
        对模型的预测结果进行反向标准化。

        Args:
            predictions (torch.Tensor): 模型的预测结果。

        Returns:
            pd.DataFrame: 反向标准化后的预测结果。
        """
        predictions_np = predictions.cpu().detach().numpy()
        return self.target_scaler.inverse_transform(predictions_np)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用训练好的模型对输入数据进行预测。

        Args:
            df (pd.DataFrame): 输入数据的 DataFrame。

        Returns:
            pd.DataFrame: 模型的预测结果。
        """
        x = self.preprocess_features(df)
        x = x.unsqueeze(0)  # 增加 batch 维度
        with torch.no_grad():
            predictions = self.model(x)
        return self.postprocess_predictions(predictions)