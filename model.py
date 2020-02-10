import torch.nn as nn
import torch.nn.functional as F
import torch

# reference: https://github.com/jojonki/cnn-for-sentence-classification/blob/master/pytorch-version.ipynb


class BaselineCNNModel(nn.Module):
    def __init__(self, vocab, vocab_size, embedding_size, out_channel_size, filter_h):
        super(BaselineCNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data.copy_(vocab.vectors)
        self.embedding.weight.requires_grad = False
        self.conv = nn.ModuleList([nn.Conv2d(1, out_channel_size, (fh, embedding_size)) for fh in filter_h])
        self.dropout = nn.Dropout(.5)
        self.fc = nn.Linear(out_channel_size * len(filter_h), 16)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
