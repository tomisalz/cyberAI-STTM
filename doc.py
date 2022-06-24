from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords


stopwordss = stopwords.words()
ps = PorterStemmer()

class Doc:
    """
    class to represent a single short text
    """

    def __init__(self, author, is_predator: bool, text: str):
        self.author = author
        self.text = text
        self.cluster = -1
        self.is_predator = is_predator
        self.confidence = 0.0
        self.list = []
        self.__parse()
        self.nwd = {w: self.list.count(w) for w in set(self.list)}

    def set_cluster(self, cluster: int, confidence: float):
        self.cluster = cluster
        self.confidence = confidence

    def __parse(self):
        """
        Preprocessing stage of text. includes:
        1. punctioation removal
        2. tokenization
        3. stopwords removal
        4. stemming of words
        :return:
        """
        new_string = self.text.translate(str.maketrans('', '', string.punctuation))

        text_tokens = word_tokenize(new_string)

        self.list = [word for word in text_tokens if word not in stopwordss]
        self.list = [ps.stem(word) for word in self.list]

    def to_list(self):
        return self.list
