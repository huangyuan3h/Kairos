from src.training.fix_standardlize import build_data, fit_feature_scaler, fit_target_scaler


def main():
    df, y = build_data()
    fit_feature_scaler(df)
    fit_target_scaler(y)


if __name__ == "__main__":
    main()