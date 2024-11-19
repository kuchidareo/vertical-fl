from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from vertical_fl.strategy import Strategy
from vertical_fl.task import load_trashnet


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""

    trainsets, valsets = load_trashnet()
    train_label, test_label = trainsets["label"], valsets["label"]
    # Define the strategy
    strategy = Strategy(train_label, test_label)

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Start Flower server
app = ServerApp(server_fn=server_fn)
