import nltk
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')

stopwordss = stopwords.words()
ps = PorterStemmer()

class Doc:

    def __init__(self, author, is_predator: bool, text: str):
        self.author = author
        self.text = text
        self.cluster = -1
        self.is_predator = is_predator
        self.confidence = 0.0
        self.list = []
        self.__parse()

    def set_cluster(self, cluster: int, confidence: float):
        self.cluster = cluster
        self.confidence = confidence

    def __parse(self):
        new_string = self.text.translate(str.maketrans('', '', string.punctuation))

        text_tokens = word_tokenize(new_string)

        self.list = [word for word in text_tokens if word not in stopwordss]
        self.list = [ps.stem(word) for word in self.list]

    def to_list(self):
        return self.list
