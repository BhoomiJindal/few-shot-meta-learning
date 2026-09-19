import numpy as np
import torch


class EpisodicSampler:
    """
    Generates N-way, K-shot episodes from a MiniImageNet dataset split.

    Each call to sample_episode() returns one episode:
        support_images, support_labels, query_images, query_labels

    Labels are remapped to a fresh 0..N-1 range PER EPISODE, since each
    episode involves a different random set of classes and the model
    only ever needs to distinguish "the N classes in front of it right now",
    not the dataset's global class indices.
    """

    def __init__(self, dataset, n_way=5, k_shot=5, q_query=15, seed=None):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        # Build a lookup: class label -> list of dataset indices belonging to it.
        # We do this once upfront so sampling is fast, instead of scanning
        # the whole dataset every time we need images from a given class.
        self.class_to_indices = {}
        for idx, label in enumerate(self.dataset.labels):
            label = int(label)
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

        self.all_classes = list(self.class_to_indices.keys())

        if len(self.all_classes) < n_way:
            raise ValueError(
                f"n_way={n_way} but dataset only has {len(self.all_classes)} classes."
            )

        self.rng = np.random.RandomState(seed)

    def sample_episode(self):
        # 1. Randomly choose N classes for this episode
        episode_classes = self.rng.choice(self.all_classes, size=self.n_way, replace=False)

        support_images, support_labels = [], []
        query_images, query_labels = [], []

        for episode_label, original_class in enumerate(episode_classes):
            indices = self.class_to_indices[original_class]

            if len(indices) < self.k_shot + self.q_query:
                raise ValueError(
                    f"Class {original_class} has only {len(indices)} images, "
                    f"need at least {self.k_shot + self.q_query} (k_shot + q_query)."
                )

            # 2. Randomly pick K+Q images from this class, no overlap between
            #    the support and query portions
            chosen = self.rng.choice(indices, size=self.k_shot + self.q_query, replace=False)
            support_idx = chosen[:self.k_shot]
            query_idx = chosen[self.k_shot:]

            for idx in support_idx:
                img, _ = self.dataset[idx]
                support_images.append(img)
                support_labels.append(episode_label)  # note: episode-local label, not original_class

            for idx in query_idx:
                img, _ = self.dataset[idx]
                query_images.append(img)
                query_labels.append(episode_label)

        return (
            torch.stack(support_images),
            torch.tensor(support_labels, dtype=torch.long),
            torch.stack(query_images),
            torch.tensor(query_labels, dtype=torch.long),
        )