import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


class ServerTrashNet(nn.Module):
    """Model from vasantvohra TrashNet: CNN 80% ipynb."""
    def __init__(self, num_devices):
        super(ServerTrashNet, self).__init__()
        self.fc1 = nn.Linear(32 * 37 * 37 * num_devices, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 6)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout1(x)

        x = self.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


class Strategy(fl.server.strategy.FedAvg):
    def __init__(self, train_label, test_label, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = ServerTrashNet(num_devices=3)
        self.initial_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0005)
        self.criterion = nn.BCELoss()
        self.train_label = train_label
        self.test_label = test_label
        self.num_devices = 3
        self.last_layer_client_model = 32 * 37 * 37

    def aggregate_fit(
        self,
        rnd,
        results,
        failures,
    ):
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        embedding_results = [
            torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0])
            for _, fit_res in results
        ]
        embeddings_aggregated = torch.cat(embedding_results, dim=1)
        embedding_server = embeddings_aggregated.detach().requires_grad_()
        output = self.model(embedding_server)
        loss = self.criterion(output, self.train_label)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        grads = embedding_server.grad.split([self.last_layer_client_model for _ in range(self.num_devices)], dim=1)
        np_grads = [grad.numpy() for grad in grads]
        parameters_aggregated = ndarrays_to_parameters(np_grads)

        return parameters_aggregated, {}


    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures,
    ):
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        embedding_results = [
            torch.from_numpy(parameters_to_ndarrays(eval_res.parameters)[0])
            for _, eval_res in results
        ]
        embeddings_aggregated = torch.cat(embedding_results, dim=1)
        embedding_server = embeddings_aggregated.detach()

        with torch.no_grad():
            output = self.model(embedding_server)
            loss = self.criterion(output, self.test_label)
            accuracy = (output.argmax(1) == self.test_label.argmax(1)).float().mean().item()

        print(loss, accuracy)
        return loss, {"accuracy": accuracy}
