from gensim.models import Doc2Vec
from pathlib import Path
import load

book_path = Path('books\\train_books')
model_save_path = Path('models\\book_model.doc2vec')

documents = load.get_doc(str(book_path))
print('Data Loading finished')

print(len(documents), type(documents))

model = Doc2Vec(documents, dm=0, alpha=0.025, size=20, min_alpha=0.025, min_count=0)

for epoch in range(2):
    print('Now training epoch %s' % epoch)
    token_count = sum([len(document) for document in documents])
    model.train(documents, total_examples=token_count, epochs=model.iter)
    model.alpha -= 0.002
    model.min_alpha = model.alpha

print(model.most_similar('обман'))

print(model['обман'])

model.save(str(model_save_path))
print("model saved")
