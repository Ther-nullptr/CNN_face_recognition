from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.data[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)
