import math


def calculate_decision_score(predicted_returns: list,
                             volatilities=None,
                             risk_tolerance: str = 'moderate') -> float:
    """
    根据预测的预期收益、波动率和风险偏好，计算决策分数。

    Args:
        predicted_returns (list): 预测的 4 个预期收益值，分别对应 1 天、3 天、5 天和 10 天的涨跌幅
        volatilities (list):  对应预测时间段的股票价格波动率
        risk_tolerance (str, optional): 用户的风险偏好，可选 'aggressive', 'moderate', 'conservative'。
            默认为 'moderate'。

    Returns:
        float: 决策分数，分数越高代表越倾向于买入。
    """

    if volatilities is None:
        volatilities = [0.02, 0.03, 0.04, 0.05]
    short_term_return = predicted_returns[1]  # 3 天的预测值
    medium_term_return = predicted_returns[2]  # 5 天的预测值
    long_term_return = predicted_returns[3]  # 10 天的预测值

    short_term_volatility = volatilities[1]
    medium_term_volatility = volatilities[2]
    long_term_volatility = volatilities[3]

    # 设定不同的风险偏好系数
    if risk_tolerance == 'aggressive':
        risk_factor = 1.2
    elif risk_tolerance == 'moderate':
        risk_factor = 1.0
    else:  # conservative
        risk_factor = 0.8

    # 使用夏普比率调整收益率，考虑波动率和风险偏好
    short_sharpe_ratio = short_term_return / short_term_volatility * risk_factor
    medium_sharpe_ratio = medium_term_return / medium_term_volatility * risk_factor
    long_sharpe_ratio = long_term_return / long_term_volatility * risk_factor

    # 动态调整权重，短期收益率权重更高
    total_sharpe = short_sharpe_ratio + medium_sharpe_ratio + long_sharpe_ratio
    short_weight = short_sharpe_ratio / total_sharpe
    medium_weight = medium_sharpe_ratio / total_sharpe
    long_weight = long_sharpe_ratio / total_sharpe

    # 计算加权平均得分
    decision_score = short_term_return * short_weight + \
                     medium_term_return * medium_weight + \
                     long_term_return * long_weight

    return decision_score


def make_decision(decision_score: float, risk_tolerance: str = 'moderate') -> str:
    """
    根据决策分数和风险偏好，返回操作建议。

    Args:
        decision_score (float): 决策分数，由 `calculate_decision_score` 函数计算得到。
        risk_tolerance (str, optional): 用户的风险偏好，可选 'aggressive', 'moderate', 'conservative'。
            默认为 'moderate'。

    Returns:
        str: 操作建议，包含 '买入（Buy）', '卖出（Sell）', '持有（Hold）', '观望（Wait）'。
    """

    # 根据风险偏好设置不同的阈值
    if risk_tolerance == 'aggressive':
        buy_threshold = 0.8
        hold_threshold = 0.3
        wait_threshold = -0.2
    elif risk_tolerance == 'moderate':
        buy_threshold = 0.5
        hold_threshold = 0.2
        wait_threshold = -0.1
    else:  # conservative
        buy_threshold = 0.3
        hold_threshold = 0.1
        wait_threshold = 0

    # 根据决策分数确定操作建议
    if decision_score >= buy_threshold:
        return "买入（Buy）"
    elif decision_score >= hold_threshold:
        return "持有（Hold）"
    elif decision_score >= wait_threshold:
        return "观望（Wait）"
    else:
        return "卖出（Sell）"
