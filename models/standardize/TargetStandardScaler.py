import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

class TargetStandardScaler:
    """
    对目标变量进行标准化，并保存/加载标准化器。
    """

    def __init__(self, data_version="v1"):
        """
        初始化 TargetStandardScaler。

        Args:
            scaler_path (str): 标准化器保存路径。默认为 'scaler_y.pkl'。
        """
        self.scaler_path = "model_files/scaler_y_v2.pkl"
        self.scaler = self.load_scaler() or StandardScaler()

    def fit(self, change_percentage_list: list):
        """
        使用给定的涨幅百分比列表拟合标准化器。

        Args:
            change_percentage_list (list): 涨幅百分比的列表。
        """
        # 将列表转换为二维数组以进行拟合
        change_percentage_array = pd.DataFrame(change_percentage_list).values
        self.scaler.fit(change_percentage_array)

    def transform(self, change_percentage_list: list) -> list:
        """
        对涨幅百分比列表进行标准化。

        Args:
            change_percentage_list (list): 涨幅百分比的列表。

        Returns:
            list: 标准化后的涨幅百分比。
        """
        change_percentage_array = pd.DataFrame(change_percentage_list).values
        scaled_array = self.scaler.transform(change_percentage_array)
        return scaled_array.tolist()

    def inverse_transform(self, scaled_change_percentage_list: list) -> list:
        """
        对标准化后的涨幅百分比列表进行反向缩放。

        Args:
            scaled_change_percentage_list (list): 标准化后的涨幅百分比列表。

        Returns:
            list: 反向缩放后的涨幅百分比列表。
        """
        scaled_array = pd.DataFrame(scaled_change_percentage_list).values
        original_array = self.scaler.inverse_transform(scaled_array)
        return original_array.tolist()

    def save_scaler(self):
        """
        保存标准化器到文件。
        """
        joblib.dump(self.scaler, self.scaler_path)

    def load_scaler(self):
        """
        从文件加载标准化器。
        """
        try:
            return joblib.load(self.scaler_path)
        except FileNotFoundError:
            return None