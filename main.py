import numpy as np
import time
from dataset import Dataset


def timer_wrapper(func):
    """
    Wrapper func to evaluate execution time of different methods
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        stop_time = time.time()
        print(f"Execution time for {func.__name__}: {stop_time - start_time}")
        return result
    return wrapper

class Word2Vec:
    """
    This class is an implementation of Word2Vec model, using skip-gram algorithm to produce word-embeddings.
    Two methods are implemented: with and without negative sampling
    """
    def __init__(self, path, window_size = 2, word_vec_dim = 100, epochs = 300, learning_rate = 1e-1, neg_samples = 5):
        ds = Dataset(path)
        self.text2num, self.word2label, self.label2word, self.noise_distribution = ds.setup()
        self.window_size = window_size
        self.word_vec_dim = word_vec_dim
        self.epochs = epochs
        self.initial_lr = learning_rate
        self.neg_samples = neg_samples
        self.negative_sampling = True
        self.learning_rate = None

    @timer_wrapper
    def train_without_negative_sampling(self):
        self.learning_rate = self.initial_lr
        self.negative_sampling = False
        self.w1 = np.random.uniform(-0.8,0.8,(len(self.word2label), self.word_vec_dim))
        self.w2 = np.random.uniform(-0.8, 0.8, (self.word_vec_dim, len(self.word2label)))

        for epoch in range(self.epochs):
            cce_loss = 0
            for i, word_num in enumerate(self.text2num):
                cce_loss_for_word = 0
                min_idx = max(i - self.window_size, 0)
                max_idx = min(i + self.window_size , len(self.text2num) - 1)

                for j in range(min_idx, max_idx + 1):
                    if i == j: continue

                    # Doing forward pass for each pair, because the values change after each loop
                    y, h, u = self.forward_pass(word_num)

                    # Categorical cross-entropy loss
                    cce_loss_for_word += self.CCE_loss(y.copy(), self.text2num[j])

                    # Getting loss for back propagation
                    loss = self.loss(y.copy(), self.text2num[j])
                    self.back_propagation(loss, h.copy(), word_num)

                cce_loss_for_word /= (max_idx - min_idx)
                cce_loss += cce_loss_for_word

            # Slightly dividing learning rate after each 200 epochs, for better results
            if not epoch % 100 and not epoch == 0: self.learning_rate /= 10

            print(f"Epoch {epoch}: loss = {cce_loss / len(self.text2num)}")

    @timer_wrapper
    def train_with_negative_sampling(self):
        self.learning_rate = self.initial_lr
        self.negative_sampling = True
        self.w1 = np.random.uniform(-0.8,0.8,(len(self.word2label), self.word_vec_dim))
        self.w2 = np.random.uniform(-0.8, 0.8, (self.word_vec_dim, len(self.word2label)))

        for epoch in range(self.epochs):
            bce_loss = 0
            for i, word_num in enumerate(self.text2num):
                bce_loss_for_word = 0
                min_idx = max(i - self.window_size, 0)
                max_idx = min(i + self.window_size , len(self.text2num) - 1)

                for j in range(min_idx, max_idx + 1):
                    if i == j: continue
                    # Taking random words with respect to noise distribution
                    false_values = np.random.choice(len(self.word2label), size = self.neg_samples, replace = False, p = self.noise_distribution)

                    # Doing forward pass for each pair
                    y, h, u = self.forward_pass(word_num, true_value = self.text2num[j], false_values = false_values)

                    # Binary cross entropy loss
                    bce_loss_for_word += self.BCE_loss(y.copy())

                    # Getting loss for back propagation
                    loss = self.loss(y.copy())
                    self.back_propagation(loss, h, word_num, true_value = self.text2num[j], false_values = false_values)

                bce_loss_for_word /= (max_idx - min_idx)
                bce_loss += bce_loss_for_word

            # Slightly dividing learning rate after each 200 epochs, for better results
            if not epoch % 50 and not epoch == 0: self.learning_rate /= 10

            print(f"Epoch {epoch}: loss = {bce_loss / len(self.text2num)}")

    def back_propagation(self, loss, h, word_num, true_value = None, false_values = None):
        if not self.negative_sampling:
            dE_dw2 = np.outer(h, loss)      # Dim x Num_of_words
            dE_dh = np.dot(self.w2, loss)   # Dim

            self.w1[word_num, :] = self.w1[word_num, :] - (self.learning_rate * dE_dh)
            self.w2 = self.w2 - (self.learning_rate * dE_dw2)

        else:
            values_stack = np.insert(false_values, 0, true_value)
            dE_dw2 = np.outer(h, loss)
            dE_dh = np.dot(self.w2[:, values_stack], loss)

            self.w1[word_num, :] = self.w1[word_num, :] - (self.learning_rate * dE_dh)
            self.w2[:, values_stack] = self.w2[:, values_stack] - (self.learning_rate * dE_dw2)

    def loss(self, predicted, word_num = None):
        if not self.negative_sampling:
            predicted[word_num] -= 1
            return predicted
        else:
            predicted[0] -= 1
            return predicted

    def CCE_loss(self, predicted, words_to_check):
        prob = predicted[words_to_check]
        prob = np.log(prob + 1e-9)
        loss = -1 * np.sum(prob)
        return loss

    def BCE_loss(self, predicted):
        predicted[1:] -= 1
        predicted[1:] *= -1
        predicted = np.log(predicted + 1e-9)
        arr_sum = np.sum(predicted)
        return -1 * arr_sum / len(predicted)


    def forward_pass(self, word_num, true_value = None, false_values = None):
        if not self.negative_sampling:
            h = self.w1[word_num,:]
            u = np.dot(self.w2.T, h)
            y = self.softmax(u)
            return y, h, u
        else:
            h = self.w1[word_num, :]
            w2_positive = self.w2[:, true_value]
            w2_negative = self.w2[:, false_values]
            w2_all = np.column_stack((w2_positive, w2_negative))
            u = np.dot(h, w2_all)
            y = self.sigmoid(u)
            return y, h, u

    def softmax(self, arr):
        e = np.exp(arr - np.max(arr))
        return e / e.sum(axis=0)

    def sigmoid(self, arr):
        return 1/(1 + np.exp(np.clip(-arr, -500, 500)))

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

    # word2vec.train_with_negative_sampling()
    word2vec.train_without_negative_sampling()

    word2vec.cosine_similarity()
