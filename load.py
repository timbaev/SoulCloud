import gensim
import os
import re
import codecs
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument


def get_doc_list(folder_name):
    doc_list = []
    file_list = [folder_name + '/' + name for name in os.listdir(folder_name) if name.endswith('txt')]
    for file in file_list:
        st = codecs.open(file, 'r', 'utf-8').read()
        doc_list.append(st)
    print('Found %s documents under the dir %s .....' % (len(file_list), folder_name))
    return doc_list


def get_doc_names(folder_name):
    doc_name_list = []
    file_list = [folder_name + '/' + name for name in os.listdir(folder_name) if name.endswith('txt')]
    for file in file_list:
        doc_name_list.append(file)
        print('Document with name %s founded' % file)
    return doc_name_list


def get_doc(folder_name):
    doc_list = get_doc_list(folder_name)
    doc_name_list = get_doc_names(folder_name)
    tokenizer = RegexpTokenizer(r'\w+')
    ru_stop = stopwords.words('russian')
    ru_stop.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])
    p_stemmer = PorterStemmer()

    taggeddoc = []

    texts = []
    for index, i in enumerate(doc_list):
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

        stopped_tokens = [i for i in tokens if not i in ru_stop]

        number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
        number_tokens = ' '.join(number_tokens).split()

        stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
        length_tokens = [i for i in stemmed_tokens if len(i) > 1]
        texts.append(length_tokens)

        td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(), doc_name_list[index])
        taggeddoc.append(td)

    return taggeddoc
