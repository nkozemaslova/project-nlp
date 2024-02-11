import pandas as pd
from catboost import CatBoostClassifier, Pool
from preprocessing import preprocess_data_test, preprocess_test

model = CatBoostClassifier()
model.load_model("model.bin")

test = pd.read_csv("./data/test_df.csv", index_col=0)


if __name__ == "__main__":
    test = preprocess_data_test(test)

test = preprocess_test(test)

test.drop("date", axis=1, inplace=True)

test_pool = Pool(
    test,
    cat_features=["bank", "time_day", "year", "month", "day"],
    text_features=["feeds", "lemmas"],
)
pred = model.predict(test_pool)

# Create a DataFrame from the predictions
df_pred = pd.DataFrame(pred, columns=["predictions"])

# Save the DataFrame to a .csv file
df_pred.to_csv("predictions.csv", index=False)
