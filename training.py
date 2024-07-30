from src.classify.training import training_classify
from src.training import training


def main():
    # training("cnn_lstm")
    training_classify("simple_lstm_classify_v1_2")


if __name__ == "__main__":
    main()
