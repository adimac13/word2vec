import numpy as np
import random
from dataset import Dataset

class Word2Vec:
    def __init__(self, path, window_size = 2, word_vec_dim = 100, epochs = 100):
        ds = Dataset(path)
        self.text2num, self.word2label, self.label2word = ds.setup()
        self.window_size = window_size
        self.word_vec_dim = word_vec_dim
        self.epochs = epochs

    def train(self):
        self.w1 = np.random.uniform(-0.8,0.8,(len(self.word2label), self.word_vec_dim))
        self.w2 = np.random.uniform(-0.8, 0.8, (self.word_vec_dim, len(self.word2label)))

        for epoch in range(self.epochs):
            for i, word_num in enumerate(self.text2num):
                y, h, u = self.forward_pass(word_num)

                min_idx = np.clip(i - self.window_size, 0, None)
                max_idx = np.clip(i + self.window_size, None, len(self.text2num)-1)

                idx_to_check = np.arange(min_idx, max_idx + 1)
                idx_to_check = np.delete(idx_to_check, np.where(idx_to_check == i))
                words_to_check = [self.text2num[idx] for idx in idx_to_check]

    def forward_pass(self, word_num):
        h = self.w1[word_num,:]
        u = np.dot(self.w2.T, h.T)
        y = self.softmax(u)
        return y, h, u

    def softmax(self, arr):
        e = np.exp(arr - np.max(arr))
        return e / e.sum(axis=0)

if __name__ == "__main__":
    pass
