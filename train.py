import pandas as pd
from catboost import CatBoostClassifier, Pool
from preprocessing import first_preprocess, preprocess_data
from sklearn.model_selection import train_test_split as tts

df = pd.read_csv("./data/train_df.csv")

df = first_preprocess(df)


if __name__ == "__main__":
    df = preprocess_data(df)


def fit_model(train_pool, validation_pool, **kwargs):
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.009,
        eval_metric="MultiClass",
        use_best_model=True,
        task_type="CPU",
        **kwargs
    )

    return model.fit(
        train_pool,
        eval_set=validation_pool,
        verbose=100,
    )


df.reset_index(drop=True, inplace=True)

df_train_val = df[
    [
        "bank",
        "feeds",
        "lemmas",
        "year",
        "month",
        "day",
        "time_day",
        "sym_len",
        "word_len",
    ]
]
y_train_val = df["grades"]
X_train, X_val, y_train, y_val = tts(
    df_train_val,
    y_train_val,
    shuffle=True,
    stratify=y_train_val,
    train_size=0.999,
)


train_pool = Pool(
    X_train,
    y_train,
    cat_features=["bank", "time_day", "year", "month", "day"],
    text_features=["lemmas", "feeds"],
)

validation_pool = Pool(
    X_val,
    y_val,
    cat_features=["bank", "time_day", "year", "month", "day"],
    text_features=["lemmas", "feeds"],
)

print("Train dataset shape: {}\n".format(train_pool.shape))

model = fit_model(train_pool, validation_pool)
model.save_model("model.bin")
