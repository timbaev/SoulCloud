import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def get_model_with_types() -> dict:
    type1 = ['любовь', 'семья', 'слёзы', 'возлюбленный', 'возлюбленная',
             'взаимность', 'нежность', 'привязанность', 'взаимопонимание', 'отношения']

    type2 = ['быт', 'страх', 'разговор', 'рассказ', 'интрига', 'речь', 'неотвратимсть']

    tag1 = 'ЛЮБОВЬ'
    tag2 = 'ДРАМА'

    doc1 = TaggedDocument(type1, [tag1])
    doc2 = TaggedDocument(type2, [tag2])

    docs = [doc1, doc2]

    model = Doc2Vec(size=100, window=10, min_count=1, workers=8)
    model.train(docs, total_examples=len(docs), epochs=10)

    return {'model': model, 'tags': [tag1, tag2], 'types': [type1, type2]}
