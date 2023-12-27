import torch
from torch.utils.data import Dataset

class DNADataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, label = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

    def get_labels(self):
        return [label for _, label in self.sequences]
