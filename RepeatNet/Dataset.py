from torch.utils.data import Dataset
import csv
import codecs
import torch

class RepeatNetDataset(Dataset):
    def __init__(self, sample_file):
        super(RepeatNetDataset, self).__init__()

        self.sample_file=sample_file

        self.item_atts=dict()
        self.samples=[]
        self.load()

    def load(self):
        clean = lambda l: [int(x) for x in l.strip('[]').split(',')]

        id=0
        with codecs.open(self.sample_file, encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter='|')
            for row in csv_reader:
                id+=1
                self.samples.append([torch.tensor([id]), torch.tensor(clean(row[0])), torch.tensor(clean(row[1]))])

        self.len=len(self.samples)
        print('data size: ', self.len)

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return self.len

def collate_fn(data):
    id, item_seq, item_tgt = zip(*data)

    return {
            'id': torch.cat(id),
            'item_seq': torch.stack(item_seq),
            'item_tgt': torch.stack(item_tgt)
            }