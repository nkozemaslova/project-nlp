import hydra
import pandas as pd
from catboost import CatBoostClassifier, Pool
from omegaconf import DictConfig, OmegaConf
from preprocessing import preprocess_data_test, preprocess_test


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)

    model = CatBoostClassifier()
    model.load_model("model3.bin")

    test = pd.read_csv("./data/test_df.csv", index_col=0)

    test = preprocess_data_test(test)
    test = preprocess_test(test)

    test.drop("date", axis=1, inplace=True)

    test_pool = Pool(
        test,
        cat_features=cfg.features.cat,
        text_features=cfg.features.text,
    )
    pred = model.predict(test_pool)

    df_pred = pd.DataFrame(pred, columns=["predictions"])

    return df_pred.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
