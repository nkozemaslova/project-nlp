import re
from functools import lru_cache

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from tqdm import tqdm


def first_preprocess(df):
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M")
    df = df.dropna()
    df = df.astype({"grades": "int32"})

    def time_day(hour):
        if 0 <= hour < 6:
            return "night"
        elif 6 <= hour < 12:
            return "morning"
        if 12 <= hour < 18:
            return "afternoon"
        if 18 <= hour < 24:
            return "evening"

    df["year"] = df["date"].apply(lambda item: item.year)
    df["month"] = df["date"].apply(lambda item: item.month)
    df["day"] = df["date"].apply(lambda item: item.day)
    df["time_day"] = df["date"].apply(lambda item: time_day(item.hour))
    df["sym_len"] = df.feeds.apply(len)
    df["word_len"] = df.feeds.apply(lambda x: len(x.split()))
    df["sym_len"] = np.log(df["sym_len"])
    df["word_len"] = np.log(df["word_len"])

    return df


m = MorphAnalyzer()
regex = re.compile("[А-Яа-яA-z]+")


def words_only(text, regex=regex):
    try:
        return regex.findall(text.lower())
    except:  # noqa: E722
        return []


@lru_cache(maxsize=128)
def lemmatize_word(token, pymorphy=m):
    return pymorphy.parse(token)[0].normal_form


def lemmatize_text(text):
    return [lemmatize_word(w) for w in text]


mystopwords = stopwords.words("russian")


def remove_stopwords(lemmas, stopwords=mystopwords):
    return [w for w in lemmas if w not in stopwords and len(w) > 3]


def clean_text(text):
    tokens = words_only(text)
    lemmas = lemmatize_text(tokens)
    return " ".join(remove_stopwords(lemmas))


def preprocess_data(df):
    lemmas = list(tqdm(map(clean_text, df["feeds"]), total=len(df)))

    df["lemmas"] = lemmas
    return df


def preprocess_data_test(df):
    lemmas = list(tqdm(map(clean_text, df["feeds"]), total=len(df)))

    df["lemmas"] = lemmas
    return df


def preprocess_test(test):
    def time_day(hour):
        if 0 <= hour < 6:
            return "night"
        elif 6 <= hour < 12:
            return "morning"
        if 12 <= hour < 18:
            return "afternoon"
        if 18 <= hour < 24:
            return "evening"

    test["date"] = pd.to_datetime(test["date"], format="%d.%m.%Y %H:%M")
    test["year"] = test["date"].apply(lambda item: item.year)
    test["month"] = test["date"].apply(lambda item: item.month)
    test["day"] = test["date"].apply(lambda item: item.day)
    test["time_day"] = test["date"].apply(lambda item: time_day(item.hour))
    test["sym_len"] = np.log(test.feeds.apply(len))
    test["word_len"] = np.log(test.feeds.apply(lambda x: len(x.split())))
    return test
