from days.training_days_model import training_days_model
from operation.training_operation_model import training_operation_model
from trend.training_trend_model import training_trend_model


def main():
    # training("cnn_lstm")
    training_trend_model("lstmTransformerV2")


if __name__ == "__main__":
    main()
