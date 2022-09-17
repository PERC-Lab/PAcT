"""Module with all the models"""

from typing import Dict

import torch
from torch import nn
from torchtext import vocab

class AST2Class(nn.Module):

    def __init__(self,
                 n_vocab,
                 embed_dim,
                 n_hidden,
                 n_output,
                 n_layers,
                 batch_size,
                 drop_p=0.8):

        super().__init__()

        self.n_vocab = n_vocab
        self.embed_dim = embed_dim
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.fc_1 = nn.Linear(3 * embed_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim,
                            n_hidden,
                            n_layers,
                            batch_first=True,
                            dropout=drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc_2 = nn.Linear(n_hidden, n_output)
        self.softmax = nn.Softmax(dim=1)

    def load_embedding(self, vocab: vocab, embedding_dict: Dict):
        """Creates embedding layer from dataset vocab and em

        Parameters
        ----------
        vocab : vocab
            Vocabulary of the dataset (not from one of the sets)
        embedding_dict : Dict
            FastText's word2vector model where key is token and value
            is the embedding weight in a numpy array
        """
        vocab_size = len(vocab)
        embedding_dim = embedding_dict.vector_size

        embedding_matrix = torch.zeros(vocab_size, embedding_dim)

        for i, el in enumerate(vocab.itos):
            embedding_matrix[i] = torch.from_numpy(embedding_dict[el].copy())

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)

    def forward(self, input_words):
        num_paths = input_words.shape[1]  # Only accepts batch inputs

        embedded_words = self.embedding(input_words)
        embedded_words_reshaped = embedded_words.view(-1,
                                                      num_paths,
                                                      3 * self.embed_dim)
        fc_out_1 = self.fc_1(embedded_words_reshaped)
        lstm_out, h = self.lstm(fc_out_1)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1,
                                              self.n_hidden)
        fc_out = self.fc_2(lstm_out)
        softmax_out = self.softmax(fc_out)
        softmax_reshaped = softmax_out.view(self.batch_size,
                                            -1,
                                            self.n_output)
        softmax_last = softmax_reshaped[:, -1]

        return softmax_last, h

    def init_hidden(self):
        device = "cuda" if torch.cuda.is_available else "cpu"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, self.batch_size,
                         self.n_hidden).zero_().to(device),
             weights.new(self.n_layers, self.batch_size,
                         self.n_hidden).zero_().to(device))

        return h
