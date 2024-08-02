import torch
import pandas as pd
import torch.nn.functional as F
from pandas import DataFrame

from models.LSTMTransformer import load_model
from models.standardize.FeatureStandardScaler import FeatureStandardScaler
from src.training.parameter import get_config


class ModelPredictorClassify:
    def __init__(self, version="simple_lstm_v1_2"):
        config = get_config(version)
        # 获取模型参数
        mp = config.model_params
        tp = config.training_params
        Model = config.Model
        data_version = config.data
        # 初始化特征和目标标准化器
        feature_scaler = FeatureStandardScaler(data_version=data_version)
        feature_scaler.load_scaler()

        self.model = Model(mp.input_dim, mp.hidden_dim, mp.num_layers, mp.num_heads)
        load_model(self.model, tp.model_save_path)
        self.model.eval()
        self.feature_scaler = feature_scaler
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

    def predict(self, df: pd.DataFrame) -> DataFrame:
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
            logits = self.model(x)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()
            predict_class = probabilities.argmax(axis=1)
        result = pd.DataFrame(probabilities, columns=['flat', 'wave'])
        result['predict_class'] = predict_class
        return result
