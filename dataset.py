import string

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
        self.text2num = list()

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
            self.text2num.append(self.word2label[el])

        return self.text2num, self.word2label, self.label2word

if __name__ == "__main__":
    pass
    # ds = Dataset('text.txt')
    # ds.setup()
