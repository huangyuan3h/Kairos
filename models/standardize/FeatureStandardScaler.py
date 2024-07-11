import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd


class FeatureStandardScaler:
    """
    对特征进行标准化，并保存/加载标准化器。
    """

    def __init__(self, scaler_path="\model_files\scaler_x.pkl", feature_columns=None):
        """
        初始化 FeatureStandardScaler。

        Args:
            scaler_path (str): 标准化器保存路径。默认为 'scaler_x.pkl'。
            feature_columns (list): 需要进行标准化的特征列名。默认为 None，表示对所有列进行标准化。
        """
        self.scaler_path = scaler_path
        self.feature_columns = feature_columns
        self.scaler = self.load_scaler() or StandardScaler()

    def fit(self, df: pd.DataFrame):
        """
        使用给定的 DataFrame 拟合标准化器。

        Args:
            df (pd.DataFrame): 用于拟合标准化器的 DataFrame。
        """
        data_to_fit = df if self.feature_columns is None else df[self.feature_columns]
        self.scaler.fit(data_to_fit)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对 DataFrame 中的特征进行标准化。

        Args:
            df (pd.DataFrame): 要进行标准化的 DataFrame。

        Returns:
            pd.DataFrame: 标准化后的 DataFrame。
        """
        if self.feature_columns is None:
            df_scaled = self.scaler.transform(df)
            return pd.DataFrame(df_scaled, columns=df.columns)
        else:
            df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
            return df

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
