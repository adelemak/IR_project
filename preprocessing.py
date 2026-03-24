import re
import string
from typing import List
import nltk
import pymorphy3
from nltk.corpus import stopwords


# Класс предобработки текста
class TextPreprocessor:
    def __init__(self) -> None:
        self.morph = pymorphy3.MorphAnalyzer()
        self.stop_words = set(stopwords.words("russian"))

        self.url_pattern = re.compile(r"http\S+|www\S+")
        self.punctuation_table = str.maketrans("", "", string.punctuation)

    # Чистим текст
    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(self.url_pattern, "", text)
        text = text.translate(self.punctuation_table)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Токенизируем
    def tokenize(self, text: str) -> List[str]:
        return text.split()


    def get_upos(self, parsed_word) -> str:
        # Словарь соответствия тегов pymorphy2 -> UPOS
        mapping = {
            'NOUN': 'NOUN',
            'VERB': 'VERB', 'INFN': 'VERB', 'PRTF': 'VERB', 'PRTS': 'VERB', 'GRND': 'VERB',
            'ADJF': 'ADJ', 'ADJS': 'ADJ', 'COMP': 'ADJ',
            'NUMR': 'NUM',
            'ADVB': 'ADV',
            'NPRO': 'PRON',
            'PREP': 'ADP',
            'CONJ': 'CCONJ',
            'PRCL': 'PART',
            'INTJ': 'INTJ'
        }
        pos = parsed_word.tag.POS
        return mapping.get(pos, 'X')  # 'X' для неизвестных частей речи

    # Лемматизация
    def lemmatize(self, tokens: List[str], upos=True) -> List[str]:
        lemmas = []
        for token in tokens:
            parsed = self.morph.parse(token)[0]
            lemma = parsed.normal_form
            pos = parsed.tag.POS  # часть речи
            if upos:
                lemmas.append(f"{lemma}_{self.get_upos(parsed)}")
            else:
                lemmas.append(lemma)
        return lemmas

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        result = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                result.append(token)
        return result

    def process_text(self, text: str, upos=True) -> List[str]:
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        lemmas = self.lemmatize(tokens, upos)
        return lemmas

    def process_corpus(self, texts: List[str], upos=True) -> List[List[str]]:
        processed_documents = []
        for text in texts:
            tokens = self.process_text(text, upos)
            processed_documents.append(tokens)
        return processed_documents


