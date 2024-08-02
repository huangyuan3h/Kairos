from days.evaluate_days import get_days_data, compare_days_models

# 比较两个模型的性能
comparison_df = compare_days_models("simpleLSTM", "lstmTransformer", get_days_data, days = 1)
df = comparison_df[["Weighted RMSE", "Weighted MAE", "Weighted MSE", "Weighted R2"]]
print(df)

# 根据不同指标选择最佳模型
best_model_wrmse = comparison_df['Weighted RMSE'].idxmin()
best_model_wmae = comparison_df['Weighted MAE'].idxmin()
best_model_wmse = comparison_df['Weighted MSE'].idxmin()
best_model_wr2 = comparison_df['Weighted R2'].idxmax()  # 注意 R2 是越大越好

print(f"Best model based on Weighted RMSE: {best_model_wrmse}")
print(f"Best model based on Weighted MAE: {best_model_wmae}")
print(f"Best model based on Weighted MSE: {best_model_wmse}")
print(f"Best model based on Weighted R2: {best_model_wr2}")

# 查看每个时间步的指标表现
for metric in ["MAE per Step", "MSE per Step", "RMSE per Step"]:
    print(f"\n{metric}:")
    for model in comparison_df.index:
        print(f"{model}: {comparison_df.loc[model, metric]}")