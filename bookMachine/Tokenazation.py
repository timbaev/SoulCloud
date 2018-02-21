import nltk
import string
import codecs
from nltk.corpus import stopwords


class Tokenazation:

    def tokenize_me(self, file_text):
        tokens = nltk.word_tokenize(file_text)

        tokens = [i for i in tokens if (i not in string.punctuation)]

        stop_words = stopwords.words('russian')
        stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])
        tokens = [i for i in tokens if (i not in stop_words)]

        tokens = [i.replace("«", "").replace("»", "") for i in tokens]

        return tokens


textFile = codecs.open('C:\\Test\\text.txt', 'r', 'utf-8-sig')
tokenazation = Tokenazation()
resultTokens = tokenazation.tokenize_me(textFile.read())

resultTextFile = open("C:\\Test\\textOut.txt", "w")
resultTextFile.write(' '.join(resultTokens))
resultTextFile.close()

print("Complete!")
