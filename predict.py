from src.crawl.sync_daily_all import sync_daily_all
from src.measure_result.normalize_result import make_decision
from src.training.predict import predict_stock_list


# from src.measure_result.normalize_result import make_decision
# from src.training.predict import predict_stock_list


def main():
    sync_daily_all()
    risk_tolerance = 'moderate'
    result = predict_stock_list(['603259','600009','600763','601689'])
    text_result = []
    for idx in range(len(result)):
        predicted_returns = result[idx][0]
        decision = make_decision(predicted_returns, risk_tolerance)
        text_result.append(decision)
        print(predicted_returns)
    print(text_result)


if __name__ == "__main__":
    main()
