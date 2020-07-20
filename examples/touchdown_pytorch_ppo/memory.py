import torch
from collections import defaultdict


class Memory:
    def __init__(self):
        self.data = defaultdict(list)

    def store(self, **kwargs):
        for key, value in kwargs.items():
            self.data[key].append(value)

    def get(self, *args):
        return [self.torchify(key) for key in args]

    def torchify(self, key) -> torch.Tensor:
        ty = type(self.data[key][0])
        if ty is bool:
            return torch.tensor(self.data[key], dtype=torch.bool).unsqueeze(1)
        elif ty is float:
            return torch.tensor(self.data[key], dtype=torch.float).unsqueeze(1)
        elif ty is torch.Tensor:
            return torch.stack(self.data[key])

    def clear(self):
        self.data.clear()

    def numpy(self) -> dict:
        data = {}
        for key, value in self.data.items():
            data[key] = self.torchify(key).detach().numpy()
        return data

    def store_numpy(self, data: dict):
        for key, values in data.items():
            for i, value in enumerate(values):
                self.data[key].append(torch.from_numpy(value))
