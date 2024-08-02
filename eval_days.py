from days.evaluate_days import get_days_data, compare_days_models

# 比较两个模型的性能
comparison_df = compare_days_models("LSTMTransformerV2", "lstmTransformer", get_days_data, days=1)
df = comparison_df[["RMSE", "MAE", "MSE", "R2"]]
print(df)

# 根据不同指标选择最佳模型
best_model_wrmse = comparison_df['RMSE'].idxmin()
best_model_wmae = comparison_df['MAE'].idxmin()
best_model_wmse = comparison_df['MSE'].idxmin()
best_model_wr2 = comparison_df['R2'].idxmax()  # 注意 R2 是越大越好

print(f"Best model based on RMSE: {best_model_wrmse}")
print(f"Best model based on MAE: {best_model_wmae}")
print(f"Best model based on MSE: {best_model_wmse}")
print(f"Best model based on R2: {best_model_wr2}")

