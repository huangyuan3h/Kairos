input_dim = 47


def get_model_params():
    """
    获取模型参数
    """
    hidden_dim = 512
    num_layers = 5
    num_heads = 16
    target_days = 10
    return input_dim, hidden_dim, num_layers, num_heads, target_days


def get_training_params():
    """
    获取训练参数
    """
    batch_size = 32
    learning_rate = 0.00001
    num_epochs = 30
    model_save_path = "model_files/lstm_transformer_model.pth"
    return batch_size, learning_rate, num_epochs, model_save_path


def get_data_params():
    """
    获取数据参数
    """
    feature_columns = [i for i in range(input_dim)]
    target_column = 7
    return feature_columns, target_column
