from src.training.predict import predict_stock_list


def main():
    result = predict_stock_list(['603259'])
    print(result)


if __name__ == "__main__":
    main()
