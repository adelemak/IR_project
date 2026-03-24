from typing import List, Tuple
from rank_bm25 import BM25Okapi


# реализация BM25 обратного индекса через библиотеку
class BM25Index:
    def __init__(self) -> None:
        self.bm25 = None
        self.documents = None

    def build(self, documents: List[List[str]]) -> None:
        self.documents = documents
        self.bm25 = BM25Okapi(documents)

    def search(
            self,
            query_tokens: List[str],
            top_k: int = 5
    ) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(query_tokens)

        ranked_doc_ids = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        results = []
        for doc_id in ranked_doc_ids[:top_k]:
            results.append((doc_id, float(scores[doc_id])))

        return results
