from days.training_days_model import training_days_model
from src.classify.training import training_classify
from src.training import training


def main():
    # training("cnn_lstm")
    # training_classify("v1_classify")
    training_days_model("lstmTransformer", 1)


if __name__ == "__main__":
    main()
