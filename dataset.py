import torch
import pandas as pd
import torch.utils.data as Data


# use 0 to padding

class TrainDataset(Data.Dataset):
    def __init__(self, args):
        
        self.data = pd.read_csv(args.data_path, header=None).replace(-1,0).values
        self.max_len = args.max_len
        self.num_item = args.num_item


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, index):

        seq = self.data[index, -self.max_len - 3:-3].tolist()
        pos = self.data[index, -self.max_len -2:-2].tolist()

        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        padding_len = self.max_len - len(pos)
        labels = [0] * padding_len + pos

        return torch.LongTensor(seq),torch.LongTensor(labels)#,torch.LongTensor(seqeval),torch.LongTensor(labelseval)#,torch.LongTensor(act)
       

class EvalDataset(Data.Dataset):
    def __init__(self,  args):
        self.data = pd.read_csv(args.data_path, header=None).replace(-1,0).values
        self.max_len = args.max_len
        self.num_item = args.num_item


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, index):

        seq = self.data[index, :-2]
        pos = self.data[index, -2]
        seq = list(seq)
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        answers = [pos]
        return torch.LongTensor(seq), torch.LongTensor(answers)


class TestDataset(Data.Dataset):
    def __init__(self,  args):
        self.data = pd.read_csv(args.data_path, header=None).replace(-1,0).values
        self.max_len = args.max_len
        self.num_item = args.num_item


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, index):

        seq = self.data[index, :-1]
        pos = self.data[index, -1]
        seq = list(seq)
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        answers = [pos]
        return torch.LongTensor(seq), torch.LongTensor(answers)
    
