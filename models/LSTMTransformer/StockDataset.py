import torch
from torch.utils.data import IterableDataset

from models.LSTMTransformer.RandomStockData import RandomStockData
import random
import threading
import queue

length_of_stock = 1


class StockDataset(IterableDataset):
    def __init__(self, feature_columns: list, target_column: str, batch_size: int, num_epochs: int):
        super(StockDataset).__init__()
        self.generate_pool = []
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.data_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.batch_counter = 0  # 添加一个计数器，记录当前 batch 中已返回的数据量
        for i in range(length_of_stock):
            self.generate_pool.append(RandomStockData(feature_columns, target_column))
        self.start_data_thread()

    def start_data_thread(self):
        # 创建并启动数据加载线程
        self.data_thread = threading.Thread(target=self._generate_data)
        self.data_thread.daemon = True  # 设置为守护线程
        self.data_thread.start()

    def _generate_data(self):
        while not self.stop_event.is_set():
            current_generator = random.choice(self.generate_pool)
            x, y = current_generator.get_data()
            self.data_queue.put((x, y))
        print("Data generation finished.")

    def __iter__(self):
        self.batch_counter = 0  # 每次迭代开始时，重置计数器
        while not self.stop_event.is_set():
            if self.batch_counter < self.batch_size:  # 检查是否达到 batch_size
                try:
                    x, y = self.data_queue.get(timeout=1)
                    self.batch_counter += 1  # 更新计数器
                    yield x, y
                except queue.Empty:
                    pass
            else:
                break

        print("Data loading finished.")

    def __next__(self):
        if self.stop_event.is_set() and self.data_queue.empty():
            print("Data loading finished.")
            raise StopIteration
        x, y = self.data_queue.get(timeout=1)  # 设置超时时间
        return x, y

    def stop_data_loading(self):
        self.stop_event.set()
        self.data_thread.join()

    def __del__(self):
        self.stop_data_loading()