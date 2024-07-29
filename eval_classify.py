# 比较两个模型的性能
from src.evaluation.evaluate_classify import compare_classify_models, get_my_data_classify

comparison_df = compare_classify_models("simple_lstm_classify", "simple_lstm_classify_v1_2", get_my_data_classify)
df = comparison_df[["Accuracy", "Precision", "Recall", "F1 Score"]]
print(df)

# 根据不同指标选择最佳模型
best_model_accuracy = comparison_df['Accuracy'].idxmax()
best_model_precision = comparison_df['Precision'].idxmax()
best_model_recall = comparison_df['Recall'].idxmax()
best_model_f1 = comparison_df['F1 Score'].idxmax()

print(f"Best model based on Accuracy: {best_model_accuracy}")
print(f"Best model based on Precision: {best_model_precision}")
print(f"Best model based on Recall: {best_model_recall}")
print(f"Best model based on F1 Score: {best_model_f1}")

# 打印混淆矩阵
for model in comparison_df.index:
    print(f"\nConfusion Matrix for {model}:")
    print(comparison_df.loc[model, "Confusion Matrix"])