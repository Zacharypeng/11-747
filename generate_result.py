from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from model import BaselineCNNModel
import time
import pdb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class BatchWrapper:
    def __init__(self, dataloader, x, y):
        self.dataloader = dataloader
        self.x = x
        self.y = y

    def __iter__(self):
        for b in self.dataloader:
            train_x = getattr(b, self.x)
            if self.y is not None:
                train_y = torch.cat([getattr(b, feat).unsqueeze(1) for feat in self.y], dim=1).long()
            else:
                train_y = torch.zeros((1))
            yield (train_x, train_y)

    def __len__(self):
        return len(self.dataloader)


# reference: https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/

tokenize = lambda x: x.split()
TEXT = Field(sequential=True, tokenize=tokenize, batch_first=True)
LABEL = Field(sequential=False, use_vocab=False, batch_first=True)
tv_data_fields = [("LABEL", LABEL), ("TEXT", TEXT)]

train_data = TabularDataset(path='topicclass/topicclass_train.csv',
                            format='csv',
                            skip_header=True,
                            fields=tv_data_fields)

val_data = TabularDataset(path='topicclass/topicclass_valid.csv',
                          format='csv',
                          skip_header=True,
                          fields=tv_data_fields)
# pdb.set_trace()
TEXT.build_vocab(train_data, val_data, vectors='fasttext.en.300d')
test_data = TabularDataset(path='topicclass/topicclass_test.csv',
                           format='csv',
                           skip_header=True,
                           fields=tv_data_fields)

test_iter = Iterator(test_data, batch_size=1, device=DEVICE, sort=False, sort_within_batch=False, repeat=False, shuffle=False)
val_iter = Iterator(val_data, batch_size=1, device=DEVICE, sort=False, sort_within_batch=False, repeat=False, shuffle=False)

test_loader = BatchWrapper(test_iter, "TEXT", ["LABEL"])
val_loader = BatchWrapper(val_iter, "TEXT", ["LABEL"])
model = torch.load('model_83dot5.pt')
model.to(DEVICE)

result = []
with torch.no_grad():
    model.eval()
    for x, y in val_loader:
        # print(x)
        preds = model(x)
        _, preds_labels = torch.max(preds.data, 1)
        label = preds_labels.cpu().detach().numpy()
        label = label.astype(int)
        # print(label)

        result.append(label[0])

print(result)
with open('val_label_result.txt', 'w') as f:
    for l in result:
        f.write(str(l))
        f.write('\n')

