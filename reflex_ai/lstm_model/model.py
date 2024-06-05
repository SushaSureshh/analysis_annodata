import torch
from torch import nn
from transformers import BertModel

class LSTMClassifier(nn.Module):
    def __init__(self, *args, hidden_dim=1024, embedding_dim=1024, vocab_size=30522, num_classes=4, max_seq_len = 300, **kwargs) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=0.3, num_layers=2)
        self.drop = nn.Dropout(0.4)

        # The linear layer that maps from hidden state space to tag sqpace
        # TODO uncomment for previous exp of lstm
        # self.out_projection = nn.Linear(max_seq_len*hidden_dim, hidden_dim)
        self.out_projection = nn.Linear(hidden_dim, hidden_dim)
        self.final_out_projection = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape : 1 x Num_Utt x T
        # print(x.shape)
        B, N, T = x.shape 
        x = x.view(B, N * T)
        # x shape : 1 x (Num_Utt * T) 
        embeds = self.word_embeddings(x)
        # x shape : 1 x (Num_Utt * T) x emb_dim
        lstm_out, _ = self.lstm(embeds)
        # x shape : 1 x (Num_Utt * T) x hid_dim
        # TODO uncomment for original lstm exp
        # lstm_out_reshape = lstm_out.view(B, N, T* self.hidden_dim)
        lstm_out_reshape = lstm_out.view(B, N, T, self.hidden_dim)
        # print("pre projection",  lstm_out_reshape.shape)
        # x shape : 1 x Num_Utt x T x hid_dim : This is to take the last timestep of each sequence
        lstm_out_slice = lstm_out_reshape[:,:,-1,:]
        # x shape: 1 x Num_Utt x hid_dim
        
        out = self.out_projection(lstm_out_slice)
        out = self.drop(out)
        out = self.final_out_projection(out)
        # x shape : 1 x Num_Utt x num_classes
        # print("out", out.shape)
        return out
