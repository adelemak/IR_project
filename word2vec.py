from gensim.models import Word2Vec
import numpy as np
from gensim.models import KeyedVectors
import pandas as pd

from preprocessing import TextPreprocessor


w2v_model = KeyedVectors.load_word2vec_format('models/model.bin', binary=True)

corpus = pd.read_csv('data/youtube_comments_sample_1500 copy.csv')
texts = corpus["text"].tolist()
preprocessor = TextPreprocessor()
result = preprocessor.process_corpus(texts)


def get_w2v_vector(text, model):
    # words = text.split()
    # Важно: RusVectōrēs часто требует теги частей речи (например, 'мама_NOUN')
    # Если ваша предобработка их не делает, ищите модели 'raw' или 'lemmas'
    vectors = [model[w] for w in text if w in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


doc_vectors = np.array([get_w2v_vector(doc, w2v_model) for doc in result])
np.save('w2v_index.npy', doc_vectors)