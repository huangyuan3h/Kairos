from operation.training_operation_model import training_operation_model
from trend.training_trend_model import training_trend_model


def main():
    # training("cnn_lstm")
    training_operation_model("lstmTransformer", 1)


if __name__ == "__main__":
    main()
