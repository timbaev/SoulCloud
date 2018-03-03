import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import to_unicode


def tokenize(doc_list):
    tokenizer = RegexpTokenizer(r'\w+')
    ru_stop = stopwords.words('russian')
    ru_stop.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])

    paragraph_stemmer = PorterStemmer()
    tagged_doc = []

    for index, i in enumerate(doc_list):
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

        stopped_tokens = [i for i in tokens if i not in ru_stop]

        number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
        number_tokens = ' '.join(number_tokens).split()

        stemmed_tokens = [paragraph_stemmer.stem(i) for i in number_tokens]

        td = TaggedDocument(to_unicode(str.encode(' '.join(stemmed_tokens))).split(), str(index))
        tagged_doc.append(td)

    return tagged_doc
