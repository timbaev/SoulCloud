import nltk
import string
from nltk.corpus import stopwords


class Tokenization:

    def tokenize_me(self, file_text) -> list:

        tokens = nltk.word_tokenize(file_text)

        tokens = [i for i in tokens if (i not in string.punctuation)]

        stop_words = stopwords.words('russian')
        stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])
        tokens = [i for i in tokens if (i not in stop_words)]

        tokens = [i.replace("«", "").replace("»", "") for i in tokens]

        return tokens
