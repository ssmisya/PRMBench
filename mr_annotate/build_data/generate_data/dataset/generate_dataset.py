
from torch.utils.data import Dataset


class GenerateDataset(Dataset):
    def __init__(self, data_args):
        self.data_args = data_args  
        
    def load_data(self):
        pass
    
    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, idx):
        pass