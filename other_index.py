from navec import Navec
import numpy as np
import pandas as pd

from preprocessing import TextPreprocessor


path = 'models/navec_hudlit_v1_12B_500K_300d_100q.tar'
navec_model = Navec.load(path)

corpus = pd.read_csv('data/youtube_comments_sample_1500 copy.csv')
texts = corpus["text"].tolist()
preprocessor = TextPreprocessor()
result = preprocessor.process_corpus(texts, upos=False)


def get_navec_vector(text, model):
    vectors = [model[w] if w in model else model['<unk>'] for w in text]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)


doc_vectors = np.array([get_navec_vector(doc, navec_model) for doc in result])
np.save('navec_index.npy', doc_vectors)
