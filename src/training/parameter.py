input_dim = 48


def get_model_params():
    """
    获取模型参数
    """
    hidden_dim = 1024
    num_layers = 4
    num_heads = 32
    return input_dim, hidden_dim, num_layers, num_heads


def get_training_params():
    """
    获取训练参数
    """

    batch_size = 2000
    learning_rate = 1e-5

    num_epochs = 5000
    model_save_path = "model_files/lstm_transformer_model.pth"
    return batch_size, learning_rate, num_epochs, model_save_path


def get_data_params():
    """
    获取数据参数
    """
    feature_columns = [i for i in range(input_dim)]
    target_column = "stock_close"
    return feature_columns, target_column
