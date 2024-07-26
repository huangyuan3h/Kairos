import torch
from torch import optim
from torch.nn import CrossEntropyLoss

from classify.StockDatasetClassify import StockDatasetClassify
from models.LSTMTransformer import load_model, train_model
from models.LSTMTransformer.StockDataLoader import create_dataloader
from src.training.parameter import get_config


def training_classify(version="v1"):
    config = get_config(version)
    # 获取模型参数
    mp = config.model_params
    tp = config.training_params
    dp = config.data_params
    Model = config.Model
    data_version = config.data

    # 检查设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Model(mp.input_dim, mp.hidden_dim, mp.num_layers, mp.num_heads).to(device)

    model = load_model(model, tp.model_save_path)

    # 优化器

    criterion = CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=tp.learning_rate)

    dataset = StockDatasetClassify(feature_columns=dp.feature_columns, target_column=dp.target_column, batch_size=tp.batch_size,
                           num_epochs=tp.num_epochs, data_version=data_version)
    dataloader = create_dataloader(dataset, tp.batch_size)

    # 使用训练数据训练模型
    train_model(model, dataloader, criterion, optimizer, tp.num_epochs, tp.model_save_path)