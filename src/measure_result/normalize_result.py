# 决策函数
import numpy as np


def make_decision(predicted_return: float) -> str:
    """
    根据预测的预期收益，返回操作建议
    """
    if predicted_return > 2:
        return "买入（Buy）"
    elif predicted_return < -2:
        return "卖出（Sell）"
    else:
        return "持有（Hold）"


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




# predicted_returns = [predicted_return_1, predicted_return_3, predicted_return_5]

# 获取综合决策
# average_dec = average_decision(predicted_returns)
# weighted_average_dec = weighted_average_decision(predicted_returns, weights=[0.5, 0.3, 0.2])
# majority_vote_dec = majority_vote_decision(predicted_returns)