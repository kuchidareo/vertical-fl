from pathlib import Path
from logging import WARN
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn as nn
from flwr.common.logger import log

from datasets import Dataset, load_dataset
from flwr_datasets.partitioner import IidPartitioner
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine


NUM_VERTICAL_SPLITS = 3


def _bin_age(age_series):
    bins = [-np.inf, 10, 40, np.inf]
    labels = ["Child", "Adult", "Elderly"]
    return (
        pd.cut(age_series, bins=bins, labels=labels, right=True)
        .astype(str)
        .replace("nan", "Unknown")
    )


def _extract_title(name_series):
    titles = name_series.str.extract(" ([A-Za-z]+)\.", expand=False)
    rare_titles = {
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    }
    titles = titles.replace(list(rare_titles), "Rare")
    titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return titles


def _create_features(df):
    # Convert 'Age' to numeric, coercing errors to NaN
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = _bin_age(df["Age"])
    df["Cabin"] = df["Cabin"].str[0].fillna("Unknown")
    df["Title"] = _extract_title(df["Name"])
    df.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)
    all_keywords = set(df.columns)
    df = pd.get_dummies(
        df, columns=["Sex", "Pclass", "Embarked", "Title", "Cabin", "Age"]
    )
    return df, all_keywords


def process_dataset():

    df = pd.read_csv(Path(__file__).parents[1] / "data/train.csv")
    processed_df = df.dropna(subset=["Embarked", "Fare"]).copy()
    return _create_features(processed_df)


def load_data(partition_id: int, num_partitions: int):
    """Partition the data vertically and then horizontally.

    We create three sets of features representing three types of nodes participating in
    the federation.

    [{'Cabin', 'Parch', 'Pclass'}, {'Sex', 'Title'}, {'Age', 'Embarked', 'Fare',
    'SibSp', 'Survived'}]

    Once the whole dataset is split vertically and a set of features is selected based
    on mod(partition_id, 3), it is split horizontally into `ceil(num_partitions/3)`
    partitions. This function returns the partition with index `partition_id % 3`.
    """

    if num_partitions != NUM_VERTICAL_SPLITS:
        log(
            WARN,
            "To run this example with num_partitions other than 3, you need to update how "
            "the Vertical FL training is performed. This is because the shapes of the "
            "gradients migh not be the same along the first dimension.",
        )

    # Read whole dataset and process
    processed_df, features_set = process_dataset()

    # Vertical Split and select
    v_partitions = _partition_data_vertically(processed_df, features_set)
    v_split_id = np.mod(partition_id, NUM_VERTICAL_SPLITS)
    v_partition = v_partitions[v_split_id]

    # Comvert to HuggingFace dataset
    dataset = Dataset.from_pandas(v_partition)

    # Split horizontally with Flower Dataset partitioner
    num_h_partitions = int(np.ceil(num_partitions / NUM_VERTICAL_SPLITS))
    partitioner = IidPartitioner(num_partitions=num_h_partitions)
    partitioner.dataset = dataset

    # Extract partition of the `ClientApp` calling this function
    partition = partitioner.load_partition(partition_id % num_h_partitions)
    partition.remove_columns(["Survived"])

    return partition.to_pandas(), v_split_id

def load_trashnet():
    dataset = load_dataset("kuchidareo/small_trashnet", split="train[:5%]")
    dataset = dataset.with_format("torch")

    if isinstance(dataset, Dataset):
        dataset = dataset.train_test_split(test_size=0.2)
    else:
        if "train" in dataset.keys():
            dataset = dataset["train"].train_test_split(test_size=0.2)
            
    image_key = "image"
    label_key = "label"

    transform = Compose([
        ToTensor(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomAffine(degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=10),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Resize((300, 300))
    ])
    def apply_transform(data):
        if image_key not in data:
            return data
        data[image_key] = [transform(img) for img in data[image_key]]
        return data

    trainsets = dataset["train"].with_transform(apply_transform)
    testsets = dataset["test"].with_transform(apply_transform)

    return trainsets, testsets


def _partition_data_vertically(df, all_keywords):
    partitions = []
    keywords_sets = [{"Parch", "Cabin", "Pclass"}, {"Sex", "Title"}]
    keywords_sets.append(all_keywords - keywords_sets[0] - keywords_sets[1])

    for keywords in keywords_sets:
        partitions.append(
            df[
                list(
                    {
                        col
                        for col in df.columns
                        for kw in keywords
                        if kw in col or "Survived" in col
                    }
                )
            ]
        )

    return partitions
