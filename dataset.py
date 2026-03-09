import string
import numpy as np

class Dataset:
    """
    This class takes in a text, find unique words and create suitable dictionaries
    """
    def __init__(self, path):
        # adding '\u2019' & '\u2018' to get rid of typographic encoding
        punc = string.punctuation + '\u2019' + '\u2018'
        translator = str.maketrans('','', punc)
        with open(path, 'r', encoding='utf-8') as f:
            file = f.read().lower()
            clean_text = file.translate(translator).split()
        self.text = clean_text
        self.word2label = {}
        self.label2word = {}
        # This list stores the text translated to numbers using word2label
        self.text2num = list()
        # This list stores number of certain words
        self.num_of_words = np.empty(0)
        self.noise_distribution = None

    def setup(self):
        """
        Setup function returns text translated into numbers based on constructed vocabulary, and two
        dictionaries: mappings from a word to idx and from idx to word
        """
        l = set()
        for el in self.text:
            if el not in l:
                l.add(el)
                self.word2label.update({el:len(l)-1})
                self.label2word.update({len(l)-1:el})
                self.num_of_words = np.append(self.num_of_words,0)
            self.num_of_words[self.word2label[el]] += 1
            self.text2num.append(self.word2label[el])

        # Word frequency table with softmax applied
        self.num_of_words **= 3/4
        sum_of_prob = np.sum(self.num_of_words)
        self.noise_distribution = self.num_of_words / sum_of_prob

        return self.text2num, self.word2label, self.label2word, self.noise_distribution

if __name__ == "__main__":
    pass
    ds = Dataset('text.txt')
    _ = ds.setup()
    false_values = np.random.choice(len(ds.word2label), size=5, replace=False,
                                    p=ds.noise_distribution )
    print(false_values)
    # ds.setup()
