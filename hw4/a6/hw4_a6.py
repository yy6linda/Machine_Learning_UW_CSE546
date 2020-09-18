import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
def collate_fn(batch):
    """
    Create a batch of data given a list of N sequences and labels. Sequences are stacked into a single tensor
    of shape (N, max_sequence_length), where max_sequence_length is the maximum length of any sequence in the
    batch. Sequences shorter than this length should be filled up with 0's. Also returns a tensor of shape (N, 1)
    containing the label of each sequence.

    :param batch: A list of size N, where each element is a tuple containing a sequence tensor and a single item
    tensor containing the true label of the sequence.

    :return: A tuple containing two tensors. The first tensor has shape (N, max_sequence_length) and contains all
    sequences. Sequences shorter than max_sequence_length are padded with 0s at the end. The second tensor
    has shape (N, 1) and contains all labels.
    """
    x = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    x_len = [len(i) for i in x]
    max_sequence_length = max(x_len)
    sentences = pad_sequence(x, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels).reshape(len(labels),1)
    return sentences, labels
    #raise NotImplementedError


class RNNBinaryClassificationModel(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()

        vocab_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]

        # Construct embedding layer and initialize with given embedding matrix. Do not modify this code.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.embedding.weight.data = embedding_matrix
        #self.lstm = nn.LSTM(embedding_dim, HIDDEN_SIZE,batch_first = True)
        self.rnn = nn.RNN(embedding_dim, HIDDEN_SIZE,num_layers = 1,batch_first=True)
        #self.dropout = nn.Dropout(LSTM_DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

        #raise NotImplementedError

    def forward(self, inputs):
        """
        Takes in a batch of data of shape (N, max_sequence_length). Returns a tensor of shape (N, 1), where each
        element corresponds to the prediction for the corresponding sequence.
        :param inputs: Tensor of shape (N, max_sequence_length) containing N sequences to make predictions for.
        :return: Tensor of predictions for each sequence of shape (N, 1).
        """
        '''
        print(type(inputs[:3]))
        print(inputs[:3])
        print("***")
        print(inputs[:3][0])
        print("***")
        print(inputs[:3][1])
        '''
        embedded = self.embedding(inputs)
        output, hidden = self.rnn(embedded)
        output2 = self.fc(hidden.squeeze(0))
        #output2 = self.fc(hidden)
        output2 = torch.nn.functional.softmax(output2, dim = 1)
        #output2 = output2.reshape(output2.size(0),1)
        return output2

    def loss(self, logits, targets):
        """
        Computes the binary cross-entropy loss.
        :param logits: Raw predictions from the model of shape (N, 1)
        :param targets: True labels of shape (N, 1)
        :return: Binary cross entropy loss between logits and targets as a scalar tensor.
        """
        #logits = logits.squeeze()
        #targets = targets.squeeze()
        #loss = torch.nn.functional.cross_entropy(logits, targets)
        loss = torch.nn.functional.cross_entropy(logits, torch.max(targets, 1)[1])
        print(loss)
        return loss

    def accuracy(self, logits, targets):
        """
        Computes the accuracy, i.e number of correct predictions / N.
        :param logits: Raw predictions from the model of shape (N, 1)
        :param targets: True labels of shape (N, 1)
        :return: Accuracy as a scalar tensor.
        """

        raise NotImplementedError


# Training parameters
TRAINING_BATCH_SIZE = 10000
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

# Batch size for validation, this only affects performance.
VAL_BATCH_SIZE = 128
