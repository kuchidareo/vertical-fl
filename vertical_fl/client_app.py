from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vertical_fl.task import load_trashnet


class ClientTrashNet(nn.Module):
    """Model from vasantvohra TrashNet: CNN 80% ipynb."""
    def __init__(self):
        super(ClientTrashNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU()
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.max_pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.max_pool1(x)

        x = self.leaky_relu(self.conv2(x))
        x = self.max_pool2(x)

        x = self.leaky_relu(self.conv3(x))
        x = self.max_pool3(x)

        x = self.flatten(x)
        return x

class FlowerClient(NumPyClient):
    def __init__(self, v_split_id, trainsets, testsets, lr):
        self.v_split_id = v_split_id
        self.trainsets = trainsets
        self.testsets = testsets
        self.model = ClientTrashNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0005)

    def get_parameters(self, config):
        pass

    def fit(self, parameters, config):
        trainloader = DataLoader(self.trainsets, batch_size=8, shuffle=True)

        embedding_aggregated = torch.empty(0)
        for batch in trainloader:
            data, label = batch["image"], batch["label"]
            embedding = self.model(data)
            embedding_aggregated = torch.cat((embedding_aggregated, embedding.unsqueeze(0)), dim=0)

        return [embedding_aggregated.detach().numpy()], 1, {}

    def evaluate(self, parameters, config):
        self.model.zero_grad()
        testloader = DataLoader(self.testsets, batch_size=8, shuffle=False)

        for batch in testloader:
            data, label = batch["image"], batch["label"]
            embedding = self.model(data)
            embedding.backward(torch.from_numpy(parameters[int(self.v_split_id)]))   
            self.optimizer.step()

        return 0.0, 1, {}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    v_split_id = np.mod(partition_id, 3)
    num_partitions = context.node_config["num-partitions"]
    trainsets, testsets = load_trashnet()
    lr = context.run_config["learning-rate"]
    return FlowerClient(v_split_id, trainsets, testsets, lr).to_client()


app = ClientApp(
    client_fn=client_fn,
)
