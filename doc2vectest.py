import gensim
import load

documents = load.get_doc('books')
print('Data Loading finished')

print(len(documents), type(documents))

model = gensim.models.Doc2Vec(documents, dm=0, alpha=0.025, size=20, min_alpha=0.025, min_count=0)

for epoch in range(200):
    if epoch % 20 == 0:
        print('Now training epoch %s' % epoch)
    token_count = sum([len(document) for document in documents])
    model.train(documents, total_examples=token_count, epochs=model.iter)
    model.alpha -= 0.002
    model.min_alpha = model.alpha

print(model.most_similar('обман'))

print(model['обман'])

print(model.docvecs.most_similar(str('books/love.txt')))

