import torch
from torch import nn
from transformers import AutoTokenizer, BertModel


class Bert_wrapper(nn.Module):
    def __init__(self, *args, hidden_dim=768, num_classes=4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.classifier_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        inp, attn_mask = x
        o = self.model(inp, attn_mask)
        final = self.classifier_layer(o.pooler_output)
        return final


class LSTMClassifier(nn.Module):
    def __init__(self, *args, hidden_dim=768, embedding_dim=768, vocab_size=50000, num_classes=4, **kwargs) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.out_projection = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, h, c):
        embeds = self.word_embeddings(x)
        lstm_out, (h,c) = self.lstm(embeds, (h,c))
        out = self.out_projection(lstm_out.view(len(x), -1))
        return out
