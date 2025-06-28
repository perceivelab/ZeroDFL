import os
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

import numpy as np

@DATASET_REGISTRY.register()
class SkinCancer(DatasetBase):
    dataset_dir = "skin_cancer"

    def __init__(self, cfg, nclients=3, iid=False):
        """
        Args:
            cfg: Configurazione del dataset.
            nclients (int): Numero di client.
            iid (bool): Se True, i dati vengono suddivisi in modo IID; altrimenti, non-IID.
        """
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, "Train")
        self.test_dir = os.path.join(self.dataset_dir, "Test")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed_split.pkl")

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            classnames = self.read_classnames()
            train = self.read_data(self.train_dir, classnames)
            test = self.read_data(self.test_dir, classnames)
            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.clients_data = self.split_data_among_clients(train, nclients, iid)
        self.print_class_distribution_per_client()
        self.test_set = test

        super().__init__(
            train_x=train,
            val=test,  # Può essere ignorato, ma lo lasciamo per continuità
            test=test,
        )

    def read_classnames(self):
        """Restituisce un dizionario che associa il nome della cartella alla classe."""
        folders = sorted(f.name for f in os.scandir(self.train_dir) if f.is_dir())
        classnames = {folder: folder.replace("_", " ") for folder in folders}
        return classnames

    def split_data_among_clients(self, train, nclients, iid, niid_type="practical", alpha=0.6, seed=42):
        """
        Suddivide il dataset di training tra i client in modo IID o non-IID.

        Args:
            train (list): Dati di training.
            nclients (int): Numero di client.
            iid (bool): Se True, suddivide i dati in modo IID; altrimenti, non-IID.
            niid_type (str): Tipo di suddivisione non-IID ("pathological" o "practical").
            alpha (float): Parametro della distribuzione di Dirichlet (solo per practical non-IID).
            seed (int, optional): Seed per la riproducibilità. Default: None.

        Returns:
            dict: Un dizionario in cui ogni client ha il proprio train set.
        """
        if seed is not None:
            np.random.seed(seed)

        clients_data = defaultdict(list)

        if iid:
            np.random.shuffle(train)
            train_chunks = np.array_split(train, nclients)
            for i in range(nclients):
                clients_data[i] = list(train_chunks[i])
        else:
            train_by_class = defaultdict(list)
            for item in train:
                train_by_class[item.label].append(item)

            if niid_type == "pathological":
                class_ids = list(train_by_class.keys())
                np.random.shuffle(class_ids)
                num_classes_per_client = max(1, len(class_ids) // nclients)
                for i in range(nclients):
                    client_classes = class_ids[i * num_classes_per_client: (i + 1) * num_classes_per_client]
                    clients_data[i] = [
                        item for cls in client_classes for item in train_by_class[cls]
                    ]
            elif niid_type == "practical":
                class_ids = list(train_by_class.keys())
                num_classes = len(class_ids)
                class_proportions = np.random.dirichlet([alpha] * nclients, num_classes)
                for cls_idx, cls in enumerate(class_ids):
                    cls_items = train_by_class[cls]
                    np.random.shuffle(cls_items)
                    proportions = class_proportions[cls_idx]
                    split_points = (proportions * len(cls_items)).astype(int)

                    discrepancy = len(cls_items) - sum(split_points)
                    split_points[:discrepancy] += 1

                    split_chunks = np.split(cls_items, np.cumsum(split_points)[:-1])

                    for client_id, chunk in enumerate(split_chunks):
                        clients_data[client_id].extend(chunk)
            else:
                raise ValueError(f"Tipo di non-IID sconosciuto: {niid_type}")

        all_data = [item for client_data in clients_data.values() for item in client_data]
        assert len(all_data) == len(set(all_data)), "Duplicati trovati nei dati tra i client!"

        return clients_data

    def print_class_distribution_per_client(self):
        """Stampa il numero di classi per ciascun client."""
        class_distribution = self.get_class_distribution_per_client()
        for client_id, classes in class_distribution.items():
            print(f"Client {client_id} ha {len(classes)} classi: {classes}")

    def get_class_distribution_per_client(self):
        """Calcola la distribuzione delle classi per ogni client.

        Returns:
            dict: Un dizionario in cui ogni chiave è un client, e il valore è
                un insieme contenente le classi presenti nei dati del client.
        """
        class_distribution = {}
        for client_id, data in self.clients_data.items():
            classes = set(item.label for item in data)
            class_distribution[client_id] = classes
        return class_distribution

    def read_data(self, directory, classnames):
        """Legge i dati dal dataset e restituisce una lista di oggetti `Datum`."""
        folders = sorted(f.name for f in os.scandir(directory) if f.is_dir())
        items = []
        label_map = {folder: idx for idx, folder in enumerate(folders)}

        for folder in folders:
            imnames = listdir_nohidden(os.path.join(directory, folder))
            classname = classnames.get(folder, folder)
            label = label_map[folder]
            for imname in imnames:
                impath = os.path.join(directory, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
