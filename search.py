import time
from sklearn.metrics.pairwise import cosine_similarity
from index_BM25 import BM25Index
from word2vec import get_w2v_vector

from preprocessing import TextPreprocessor
preprocessor = TextPreprocessor()
BM25 = BM25Index()

corpus = pd.read_csv('data/youtube_comments_sample_1500 copy.csv')
texts = corpus["text"].tolist()
result_pos = preprocessor.process_corpus(texts, upos=True)
result_nopos = preprocessor.process_corpus(texts, upos=False)


def search(corpus, query_text, index_type, top_n=5):
    start_time = time.perf_counter()

    # 2. Выбор индекса и получение вектора запроса/результатов
    if index_type == 'bm25':
        clean_query = preprocessor.process_text(query_text, upos=False)
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_model = BM25.build(corpus)
        scores = bm25_model.get_scores(clean_query)
    elif index_type == 'w2v':
        clean_query = preprocessor.process_text(query_text, upos=True)

        # Для w2v или fasttext
        q_vec = get_w2v_vector(clean_query, w2v_model).reshape(1, -1)
        # Считаем косинусное сходство со всей матрицей сразу
        scores = cosine_similarity(q_vec, loaded_index)[0]

    # 3. Ранжирование
    best_indices = np.argsort(scores)[::-1][:top_n]

    end_time = time.perf_counter()
    duration = end_time - start_time

    return best_indices, duration