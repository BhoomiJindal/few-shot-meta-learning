import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks (Snell et al., 2017).

    Wraps an embedding backbone. Classification works by computing one
    prototype per class (the mean embedding of that class's support
    examples), then classifying each query example by nearest prototype
    under squared Euclidean distance.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, support_images, support_labels, query_images, n_way):
        """
        support_images: (n_way * k_shot, 3, 84, 84)
        support_labels: (n_way * k_shot,)  -- episode-local labels, 0..n_way-1
        query_images:   (n_way * q_query, 3, 84, 84)
        n_way: number of classes in this episode

        Returns: logits of shape (n_query_total, n_way) -- negative distances
                 to each prototype. "Logits" here just means unnormalized
                 scores; higher score = model thinks the query is more
                 likely to belong to that class.
        """
        support_embeddings = self.backbone(support_images)  # (n_way*k_shot, emb_dim)
        query_embeddings = self.backbone(query_images)        # (n_query_total, emb_dim)

        # Compute one prototype per class: the mean embedding of that
        # class's support examples.
        prototypes = []
        for class_idx in range(n_way):
            class_mask = (support_labels == class_idx)
            class_embeddings = support_embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)  # (n_way, emb_dim)

        # Compute squared Euclidean distance from every query embedding to
        # every prototype. Using broadcasting: query_embeddings gets a new
        # middle dimension, prototypes gets a new first dimension, so
        # subtracting them compares every query against every prototype
        # in one vectorized operation instead of a slow nested loop.
        query_embeddings = query_embeddings.unsqueeze(1)   # (n_query_total, 1, emb_dim)
        prototypes = prototypes.unsqueeze(0)                # (1, n_way, emb_dim)
        distances = ((query_embeddings - prototypes) ** 2).sum(dim=2) / support_embeddings.shape[1]  # normalized by embedding dim
        
        # Negative distance = logits. Smaller distance should mean higher
        # score, so we flip the sign before feeding this into cross-entropy
        # loss (which expects "higher score = more likely").
        logits = -distances
        return logits