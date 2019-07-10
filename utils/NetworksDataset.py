from torch.utils.data import Dataset


class NetworksDataset(Dataset):
    def __init__(self, input_networks, output_networks):
        self.input_networks = input_networks
        self.output_networks = output_networks

    def __len__(self):
        if isinstance(self.input_networks, list):
            return self.input_networks[0].shape[0]
        else:
            return self.input_networks.shape[0]

    def __getitem__(self, indx):
        if isinstance(self.input_networks, list):
            x = []
            y = []
            for i in range(len(self.input_networks)):
                x.append(self.input_networks[i][indx])
                y.append(self.output_networks[i][indx])
        else:
            x = self.input_networks[indx]
            y = self.output_networks[indx]
        return x, y
