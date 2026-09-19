import pickle
import numpy as np
from torch.utils.data import Dataset


class MiniImageNet(Dataset):
    """
    Loads one split (train/val/test) of the miniImageNet dataset from the
    cached .pkl format (Ren et al., 2018), and exposes it as a standard
    PyTorch Dataset of (image, label) pairs, where label is an integer
    class index local to this split (0 to num_classes-1).
    """

    def __init__(self, pkl_path, transform=None):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.images = data["image_data"]  # shape: (N, 84, 84, 3), uint8
        class_dict = data["class_dict"]   # class_name -> list of indices

        # Assign each class name a clean integer label (0, 1, 2, ...)
        self.class_names = sorted(class_dict.keys())
        self.class_to_label = {name: i for i, name in enumerate(self.class_names)}

        # Build a flat label array aligned with self.images
        self.labels = np.zeros(len(self.images), dtype=np.int64)
        for class_name, indices in class_dict.items():
            label = self.class_to_label[class_name]
            for idx in indices:
                self.labels[idx] = label

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (84, 84, 3), uint8
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def num_classes(self):
        return len(self.class_names)