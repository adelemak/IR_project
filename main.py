import argparse
from search import SearchEngine
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Search Engine CLI")
    parser.add_argument("--query", type=str, required=True, help="Текст поискового запроса")
    parser.add_argument("--index", choices=['bm25', 'w2v', 'navec'], required=True, help="Тип индекса")
    parser.add_argument("--top", type=int, default=5, help="Количество результатов")

    args = parser.parse_args()

    corpus = pd.read_csv('data/youtube_comments_sample_1500 copy.csv')
    corpus_list = corpus['text'].tolist()
    search = SearchEngine(corpus_list, args.index)

    results, search_time = search.search(args.query, args.top)

    print(f"Результаты для индекса {args.index}:")
    for res in results:
        idx = res["index"]
        print(f"Doc ID: {idx} | Score: {res["score"]} | {corpus_list[idx][:100]}...")
    print(f"\nВремя поиска: {search_time:.4f} сек.")


if __name__ == "__main__":
    main()
