import hydra
import pandas as pd
from catboost import CatBoostClassifier, Pool
from omegaconf import DictConfig, OmegaConf
from preprocessing import first_preprocess, preprocess_data
from sklearn.model_selection import train_test_split as tts


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)

    def fit_model(train_pool, validation_pool, **kwargs):
        model = CatBoostClassifier(
            iterations=cfg.params.iterations,
            learning_rate=cfg.params.learning_rate,
            eval_metric=cfg.params.eval_metric,
            use_best_model=cfg.params.use_best_model,
            task_type=cfg.params.task_type,
            **kwargs
        )

        return model.fit(
            train_pool,
            eval_set=validation_pool,
            verbose=cfg.params.verbose,
        )

    df = pd.read_csv("./data/train_df.csv")
    df = first_preprocess(df)
    df = preprocess_data(df)

    df.reset_index(drop=True, inplace=True)

    df_train_val = df[cfg.dataset.columns]
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
        cat_features=cfg.features.cat,
        text_features=cfg.features.text,
    )

    validation_pool = Pool(
        X_val,
        y_val,
        cat_features=cfg.features.cat,
        text_features=cfg.features.text,
    )

    print("Train dataset shape: {}\n".format(train_pool.shape))

    model = fit_model(train_pool, validation_pool)
    return model.save_model("model.bin")


if __name__ == "__main__":
    main()
