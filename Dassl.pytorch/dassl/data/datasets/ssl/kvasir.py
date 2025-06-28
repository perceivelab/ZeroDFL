import os
import pickle
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase
from dassl.utils import listdir_nohidden

import torch

@DATASET_REGISTRY.register()
class KvasirDataset(DatasetBase):
    dataset_dir = "kvasir-dataset-v2"

    def __init__(self, cfg, nclients=3, iid=False, zero_shot=True, seed=43, relabel=True):
        """
        Args:
            cfg: Configurazione del dataset.
            nclients (int): Numero di client.
            iid (bool): Se True, i dati vengono suddivisi in modo IID; altrimenti, non-IID.
            zero_shot (bool): Se True, metà delle classi sono separate nel set Novel.
            seed (int): Seed per la riproducibilità.
            relabel (bool): Se True, rimappa le etichette del train e del novel set.
        """
        np.random.seed(seed)
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed_split.pkl")

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                data = pickle.load(f)
                train, test, novel = data["train"], data["test"], data["novel"]

            self._num_classes = max(datum.label for datum in train) + 1
        else:
            classnames = self.read_classnames()
            all_classes = list(classnames.keys())

            # Suddivisione classi viste e novel
            if zero_shot:
                np.random.shuffle(all_classes)
                seen_classes = all_classes[:len(all_classes) // 2]
                novel_classes = all_classes[len(all_classes) // 2:]
            else:
                seen_classes = all_classes
                novel_classes = []

            self._num_classes = len(seen_classes)

            # Mappiamo le etichette in range separati se relabel=True
            if relabel:
                train_relabeler = {cls: idx for idx, cls in enumerate(seen_classes)}
                novel_relabeler = {cls: idx for idx, cls in enumerate(novel_classes)}
            else:
                train_relabeler = {cls: cls for cls in seen_classes}
                novel_relabeler = {cls: cls for cls in novel_classes}

            train, test, novel = self.load_data(
                classnames, seen_classes, novel_classes, train_relabeler, novel_relabeler
            )

            with open(self.preprocessed, "wb") as f:
                pickle.dump({"train": train, "test": test, "novel": novel}, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.clients_data = self.split_data_among_clients(train, nclients, iid)
        self.print_class_distribution_per_client()

        self.test_set = test
        self.novel_set = novel

        super().__init__(train_x=train, val=test, test=novel)

    def read_classnames(self):
        """Restituisce un dizionario che associa il nome della cartella alla classe."""
        folders = sorted(f.name for f in os.scandir(self.dataset_dir) if f.is_dir())
        classnames = {folder: folder.replace("-", " ") for folder in folders}
        return classnames

    def load_data(self, classnames, seen_classes, novel_classes, train_relabeler, novel_relabeler):
        """Legge i dati e divide in Train, Test e Novel (se necessario)."""
        train, test, novel = [], [], []

        for folder in os.listdir(self.dataset_dir):
            class_path = os.path.join(self.dataset_dir, folder)
            if not os.path.isdir(class_path):
                continue

            impaths = [os.path.join(class_path, f) for f in listdir_nohidden(class_path)]
            classname = classnames[folder]

            if folder in seen_classes:
                label = train_relabeler[folder]  # Usa il mapping numerico del train
                train_paths, test_paths = train_test_split(impaths, test_size=0.2, random_state=42)
                train.extend([Datum(impath=img, label=label, classname=classname) for img in train_paths])
                test.extend([Datum(impath=img, label=label, classname=classname) for img in test_paths])
            elif folder in novel_classes:
                label = novel_relabeler[folder]  # Usa il mapping numerico del novel
                novel.extend([Datum(impath=img, label=label, classname=classname) for img in impaths])

        return train, test, novel

    def split_data_among_clients(self, train, nclients, iid, niid_type="practical", alpha=0.9, seed=42):
        """Suddivide il dataset di training tra i client in modo IID o non-IID."""
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

        return clients_data

    def get_class_counts_per_client(self):
        """
        Calcola per ogni client il numero di immagini per ciascuna classe,
        utilizzando la label numerica (item.label).
        """
        class_counts = {}
        for client_id, data in self.clients_data.items():
            counts = defaultdict(int)
            for item in data:
                counts[item.label] += 1
            class_counts[client_id] = dict(counts)
        return class_counts

    def print_class_distribution_per_client(self):
        """
        Stampa per ogni client le classi (con il numero della label) e il numero di immagini per classe;
        inoltre, calcola per ogni client un vettore di pesi (da usare in weighted loss) di dimensione self.num_classes.
        Il peso per una classe i è calcolato come:
            w_i = (total immagini nel client) / (self.num_classes * count_i)
        Se il client non ha immagini di una determinata classe, viene usato un peso di default pari a 1.0.
        """
        gamma = 0.5
        class_counts = self.get_class_counts_per_client()
        self.class_counts = class_counts
        self.class_weights = {}  # per ogni client, un tensore di dimensione (self.num_classes,)

        for client_id, counts in class_counts.items():
            total = sum(counts.values())
            weight_vector = [1.0] * self.num_classes  # inizializza con default
            for i in range(self.num_classes):
                cnt = counts.get(i, 0)
                if cnt > 0:
                    weight_vector[i] = (total / (self.num_classes * cnt)) ** gamma
                else:
                    weight_vector[i] = 1.0
            self.class_weights[client_id] = torch.tensor(weight_vector, dtype=torch.float)
            # Per la stampa, mostra anche il mapping label -> count
            classes_info = ", ".join([f"{i}: {counts.get(i, 0)} immagini (peso: {weight_vector[i]:.2f})"
                                       for i in range(self.num_classes)])
            print(f"Client {client_id} -> {classes_info}")

    def get_class_distribution_per_client(self):
        """
        Calcola la distribuzione delle classi per ogni client (in termini di label numeriche).
        """
        class_distribution = {}
        for client_id, data in self.clients_data.items():
            classes = set(item.label for item in data)
            class_distribution[client_id] = classes
        return class_distribution
