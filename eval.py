from src.evaluation.evaluate import compare_models, get_my_data

# 比较两个模型的性能
comparison_df = compare_models("time_series_transformer", "simple_lstm_v1_2", get_my_data)
print(comparison_df)

# 根据不同指标选择最佳模型
best_model_rmse = comparison_df['RMSE'].idxmin()
best_model_mae = comparison_df['MAE'].idxmin()
best_model_mse = comparison_df['MSE'].idxmin()
best_model_r2 = comparison_df['R2'].idxmax()  # 注意 R2 是越大越好

print(f"Best model based on RMSE: {best_model_rmse}")
print(f"Best model based on MAE: {best_model_mae}")
print(f"Best model based on MSE: {best_model_mse}")
print(f"Best model based on R2: {best_model_r2}")