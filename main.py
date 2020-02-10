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

batch_size = (256, 256)
train_iter, val_iter = BucketIterator.splits((train_data, val_data), batch_sizes=batch_size, device=DEVICE,
                                             sort_key=lambda d: len(d.TEXT), sort_within_batch=False, repeat=False)

train_loader = BatchWrapper(train_iter, "TEXT", ["LABEL"])
val_loader = BatchWrapper(val_iter, "TEXT", ["LABEL"])

vocab_size = len(TEXT.vocab)
embedding_size = 300
out_channel_size = 128  
filter_height = [3, 4, 5]
model = BaselineCNNModel(vocab=TEXT.vocab, vocab_size=len(TEXT.vocab), embedding_size=embedding_size,
                         out_channel_size=out_channel_size, filter_h=filter_height)
model = model.to(DEVICE)

optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad],
                       lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12], gamma=0.1)
criterion = nn.CrossEntropyLoss()
num_epoch = 20
cur_acc = 0.0
for epoch in range(1, num_epoch + 1):
    model.train()
    start_time = time.time()
    batch_num = 0
    running_loss = 0.0
    for x, y in train_loader:
        batch_num += 1
        y = y.squeeze()
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_num % 800 == 0:
            print('Batch: {}, Loss: {}'.format(batch_num, loss.item()))
    end_time = time.time()

    epoch_loss = running_loss / len(train_loader)
    print('*** Epoch: {}, Training Loss: {:.3f}, Time: {}min ***'.format(epoch, epoch_loss,
                                                                         (end_time - start_time) // 60))

    val_loss = 0.0
    batch_num = 0
    total_predictions = 0.0
    correct_predictions = 0.0
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            batch_num += 1
            y = y.squeeze()
            preds = model(x)
            _, preds_labels = torch.max(preds.data, 1)
            total_predictions += y.size(0)
            correct_predictions += (preds_labels == y).sum().item()
            loss = criterion(preds, y).detach()
            val_loss += loss.item()
        epoch_val_loss = val_loss / len(val_loader)
        acc = (correct_predictions / total_predictions) * 100
        print('*** Epoch: {}, Validation Loss: {:.3f}, Accuracy: {}% ***\n'.format(epoch, epoch_val_loss, acc))
        if acc > cur_acc:
            print('@@@ Saving Best Model with Accuracy: {}% @@@\n'.format(acc))
            torch.save(model, 'model.pt')
            cur_acc = acc
    scheduler.step()
