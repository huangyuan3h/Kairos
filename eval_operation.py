
from operation.evaluate_operation import compare_operation_models, get_operation_data

# 比较两个模型的性能
comparison_df = compare_operation_models("lstmTransformerV2", "lstmTransformer", get_operation_data, days=2)
df = comparison_df[["RMSE", "MAE", "MSE", "R2", "Accuracy"]]
print(df)

# 根据不同指标选择最佳模型
best_model_wrmse = comparison_df['RMSE'].idxmin()
best_model_wmae = comparison_df['MAE'].idxmin()
best_model_wmse = comparison_df['MSE'].idxmin()
best_model_wr2 = comparison_df['R2'].idxmax()
best_model_accuracy = comparison_df['Accuracy'].idxmax()  # 获取准确率最高的模型

print(f"Best model based on RMSE: {best_model_wrmse}")
print(f"Best model based on MAE: {best_model_wmae}")
print(f"Best model based on MSE: {best_model_wmse}")
print(f"Best model based on R2: {best_model_wr2}")
print(f"Best model based on Accuracy: {best_model_accuracy}")  # 打印准确率最高的模型

