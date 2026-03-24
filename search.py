import time
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from preprocessing import TextPreprocessor
from navec import Navec


class SearchEngine:


    def __init__(self, corpus, model_type='w2v',
                 w2v_model_path='models/model.bin',
                 w2v_index_path='w2v_index.npy',
                 navec_model_path='models/navec_hudlit_v1_12B_500K_300d_100q.tar',
                 navec_index_path='navec_index.npy',
                 bm25_index_path="bm25_index.pkl"):
        
        self.corpus = corpus
        self.preprocessor = TextPreprocessor()
        self.model_type = model_type

        # Загружаем модель и индекс в зависимости от типа
        if model_type == 'w2v':
            print("Загрузка модели Word2Vec...")
            self.model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
            print(f"Загрузка индекса Word2Vec из {w2v_index_path}...")
            self.doc_vectors = np.load(w2v_index_path)
            # self.vector_size = self.model.vector_size

        elif model_type == 'navec':
            print("Загрузка модели Navec...")
            self.model = Navec.load(navec_model_path)
            print(f"Загрузка индекса Navec из {navec_index_path}...")
            self.doc_vectors = np.load(navec_index_path)
            # self.vector_size = self.model.dim

        elif model_type == 'bm25':
            print(f"Загрузка индекса BM25 из {bm25_index_path}...")
            with open(bm25_index_path, 'rb') as f:
                self.model = pickle.load(f)

        else:
            raise ValueError("model_type должен быть 'w2v'/'navec'/'bm25'")


    def search(self, query_text, top_n=5):
        start_time = time.perf_counter()

        if self.model_type == 'bm25':
            query_tokens = self.preprocessor.process_text(query_text, upos=False)
            scores = self.model.get_scores(query_tokens)
            best_indices = np.argsort(scores)[::-1][:top_n]
            final_scores = scores

        else:
            if self.model_type == 'w2v':
                tokens = self.preprocessor.process_text(query_text, upos=True)
            else:  # navec
                tokens = self.preprocessor.process_text(query_text, upos=False)

            vectors = [self.model[token] for token in tokens if token in self.model]
            if vectors:
                query_vector = np.mean(vectors, axis=0).reshape(1, -1)
            else:
                query_vector = np.zeros((1, 300))

            final_scores = cosine_similarity(query_vector, self.doc_vectors)[0]
            best_indices = np.argsort(final_scores)[::-1][:top_n]

        search_results = []
        for idx in best_indices:
            if isinstance(self.corpus, pd.DataFrame):
                text = self.corpus.iloc[idx]['text']
            else:
                text = self.corpus[idx]

            search_results.append({
                'index': idx,
                'text': text,
                'score': float(final_scores[idx]),
                'similarity': f"{final_scores[idx]:.4f}"
            })

        elapsed_time = time.perf_counter() - start_time
        return search_results, elapsed_time

