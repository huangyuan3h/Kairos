from src.training.fix_standardlize import build_data, fit_feature_scaler, fit_target_scaler


def main():
    df, y = build_data("v1")
    fit_feature_scaler(df, "v2")
    fit_target_scaler(y, "v2")


if __name__ == "__main__":
    main()
