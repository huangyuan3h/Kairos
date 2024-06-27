# 决策函数
import numpy as np


def make_decision(predicted_returns: list, risk_tolerance: str = 'moderate') -> str:
    """
    根据预测的预期收益，返回操作建议

    Args:
        predicted_returns (list): 预测的 4 个预期收益值，分别对应 1 天、3 天、5 天和 10 天的涨跌幅
        risk_tolerance (str, optional): 用户的风险偏好，可选 'aggressive', 'moderate', 'conservative'。
            默认为 'moderate'。

    Returns:
        str: 操作建议，包含 '买入（Buy）', '卖出（Sell）', '持有（Hold）', '观望（Wait）'。
    """

    short_term_return = predicted_returns[1]  # 3 天的预测值
    medium_term_return = predicted_returns[2] # 5 天的预测值
    long_term_return = predicted_returns[-1] # 10 天的预测值

    # 设定不同的决策阈值，考虑风险偏好
    if risk_tolerance == 'aggressive':
        buy_threshold = 1.5
        sell_threshold = -0.5
    elif risk_tolerance == 'moderate':
        buy_threshold = 1
        sell_threshold = -0.3
    else:  # conservative
        buy_threshold = 0.5
        sell_threshold = 0

    # 综合考虑短期、中期和长期趋势
    if short_term_return > buy_threshold and medium_term_return > buy_threshold and long_term_return > buy_threshold:
        return "买入（Buy）"
    elif short_term_return < sell_threshold and medium_term_return < sell_threshold and long_term_return < sell_threshold:
        return "卖出（Sell）"
    elif short_term_return > 0 and medium_term_return > 0 and long_term_return > 0:
        return "持有（Hold）"
    else:
        return "观望（Wait）"


# 平均法
def average_decision(predicted_returns: list) -> str:
    avg_return = np.mean(predicted_returns)
    return make_decision(avg_return)


# 加权平均法
def weighted_average_decision(predicted_returns: list, weights: list) -> str:
    weighted_avg_return = np.average(predicted_returns, weights=weights)
    return make_decision(weighted_avg_return)


# 多数投票法
def majority_vote_decision(predicted_returns: list) -> str:
    decisions = [make_decision(r) for r in predicted_returns]
    return max(set(decisions), key=decisions.count)







# 假设预测结果为：
predicted_returns =[0.9197320938110352, 0.8447604179382324, 0.6181734800338745, 0.05447008088231087]  # 1天，3天，5天，10天

# 用户的风险偏好
risk_tolerance = 'moderate'

# 调用决策函数
decision = make_decision(predicted_returns, risk_tolerance)

# 输出决策结果
print(decision)  # 输出结果: 买入（Buy）

# predicted_returns = [predicted_return_1, predicted_return_3, predicted_return_5]

# 获取综合决策
# average_dec = average_decision(predicted_returns)
# weighted_average_dec = weighted_average_decision(predicted_returns, weights=[0.5, 0.3, 0.2])
# majority_vote_dec = majority_vote_decision(predicted_returns)