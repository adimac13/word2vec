import numpy as np
from dataset import Dataset

class Word2Vec:
    """
    This class is an implementation of Word2Vec model, using skip-gram algorithm to produce word-embeddings.
    """
    def __init__(self, path, window_size = 2, word_vec_dim = 100, epochs = 400, learning_rate = 1e-1):
        ds = Dataset(path)
        self.text2num, self.word2label, self.label2word = ds.setup()
        self.window_size = window_size
        self.word_vec_dim = word_vec_dim
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self):
        self.w1 = np.random.uniform(-0.8,0.8,(len(self.word2label), self.word_vec_dim))
        self.w2 = np.random.uniform(-0.8, 0.8, (self.word_vec_dim, len(self.word2label)))

        for epoch in range(self.epochs):
            cce_loss = 0
            for i, word_num in enumerate(self.text2num):
                y, h, u = self.forward_pass(word_num)

                # Getting indexes of words in the neighbourhood of the current word
                min_idx = np.clip(i - self.window_size, 0, None)
                max_idx = np.clip(i + self.window_size, None, len(self.text2num)-1)
                idx_to_check = np.arange(min_idx, max_idx + 1)
                idx_to_check = np.delete(idx_to_check, np.where(idx_to_check == i))
                words_to_check = [self.text2num[idx] for idx in idx_to_check]

                # Categorical cross-entropy loss
                cce_loss += self.CCE_loss(y, words_to_check)

                # Getting loss for back propagation
                loss = self.loss(y, words_to_check)
                self.back_propagation(loss, h, word_num)

            # Slightly dividing learning rate after each 200 epochs, for better results
            if not epoch % 200 and not epoch == 0: self.learning_rate /= 10

            print(f"Epoch {epoch}: loss = {cce_loss / len(self.text2num)}")


    def back_propagation(self, loss, h, word_num):
        dE_dw2 = np.outer(h, loss)      # Dim x Num_of_words
        dE_dh = np.dot(self.w2, loss)   # Dim

        self.w1[word_num, :] = self.w1[word_num, :] - (self.learning_rate * dE_dh)
        self.w2 = self.w2 - (self.learning_rate * dE_dw2)

    def loss(self, predicted, words_to_check):
        predicted_multiplied = predicted * len(words_to_check)
        for word in words_to_check:
            predicted_multiplied[word] -= 1
        return predicted_multiplied

    def CCE_loss(self, predicted, words_to_check):
        prob = predicted[words_to_check]
        prob = np.log(prob + 1e-9)
        loss = -1 * np.sum(prob)
        return loss

    def forward_pass(self, word_num):
        h = self.w1[word_num,:]
        u = np.dot(self.w2.T, h)
        y = self.softmax(u)
        return y, h, u

    def softmax(self, arr):
        e = np.exp(arr - np.max(arr))
        return e / e.sum(axis=0)

    def cosine_similarity(self, word = "been"):
        """
        Extra function, to find most similar words to the word given
        """
        if word not in self.word2label:
            print("Did not find given word")
            return
        else:
            dot_product = self.w1 @ self.w1[self.word2label[word]]
            norm = np.linalg.norm(self.w1, axis = 1)

            # Omitting dividing by the length of embedding vector, because it will eventually give the same result
            final_arr = dot_product / norm
            idx_sorted = np.argsort(final_arr)

            print("\n5 most similar words:")
            for i in range(2,7):
                print(self.label2word[idx_sorted[-i]])

if __name__ == "__main__":
    word2vec = Word2Vec('text.txt')
    word2vec.train()
    word2vec.cosine_similarity()
