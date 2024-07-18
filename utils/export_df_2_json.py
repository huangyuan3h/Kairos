import pandas as pd
import json

def df_to_json_with_key_value_pairs(df: pd.DataFrame, key_column: str, value_column: str, filename: str) -> None:
  """
  从 DataFrame 中选择两列，并将它们转换为 JSON 对象，其中一列作为键，另一列作为值。

  Args:
    df: 输入的 DataFrame。
    key_column: 用作键的列的名称。
    value_column: 用作值的列的名称。
    filename: 要保存的 JSON 文件名。
  """
  json_object = dict(zip(df[key_column], df[value_column]))
  with open(filename, 'w', encoding='utf-8') as f:
    json.dump(json_object, f, indent=4, ensure_ascii=False)

