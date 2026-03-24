## Проект
Выполнили: Дарья Баранова, Аделина Макарова

Поисковая система для текстовых данных с использованием:
- BM25
- Word2Vec
- Navec

### Корпус
*  [датасет](https://huggingface.co/datasets/Maxstan/russian_youtube_comments_political_and_nonpolitical) политических комментариев на ютубе с HuggingFace; взяли первые 1500 записей.

### Модели
[Navec](https://github.com/ba-soft/natasha-navec)

[Word2Vec](https://vectors.nlpl.eu/repository/20/185.zip)

Выбирали исходя из веса моделей. 

### Структура репозитория
infopoisk_project/
│
├── data/ 
  ├── youtube_comments_sample_1500 copy.csv # Датасет
├── models/ # Модели (не включены в репозиторий)
│ ├── model.bin
│ └── navec_*.tar
│
├── main.py
├── preprocessing.py # Предобработка текста
├── index_BM25.py # BM25 индекс
├── search.py # Поисковик
├── word2vec.py # Word2Vec
├── other.py # Navec
│
├── w2v_index.npy # Индекс Word2Vec
├── navec_index.npy # Индекс Navec
│
└── README.md
### Команда запуска
Для запуска через командную строку код выглядит следующим образом:
python main.py --query "привет" --index bm25 --top 3
Параметры: запрос, тип индекса, число топ документов на выдачу.

##### Файлы с индексами
Получаются в результате кода в файлах index_BM25.py, other_index.py и word2vec.py
