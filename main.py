import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, KFold


class CarsUtils:
    @staticmethod
    def read_cars_from_file(filename) -> pd.DataFrame:
        car_list = []
        with open(filename, "r") as file:
            for line in file:
                parts = line.split()
                try:
                    mpg, cylinders, displacement, horsepower, weight, accel, model_year, origin = map(float, parts[:8])
                except ValueError:
                    continue
                model_year, origin = int(model_year), int(origin)
                name = " ".join(parts[8:]).replace('"', '')
                car_dict = {
                    "mpg": mpg,
                    "cylinders": cylinders,
                    "displacement": displacement,
                    "horsepower": horsepower,
                    "weight": weight,
                    "accel": accel,
                    "model_year": model_year,
                    "origin": origin,
                    "name": name
                }
                car_list.append(car_dict)
        return pd.DataFrame(car_list)

    @staticmethod
    def normalise_data(df, columns) -> pd.DataFrame:
        scaler = preprocessing.MinMaxScaler()

        # Apply normalization to specified columns
        df[columns] = scaler.fit_transform(df[columns])
        return df

    @staticmethod
    def plot_features_vs_mpg(df):
        # TODO: Refactor into one-hot encoded origin compatible plotting
        features = ["cylinders", "displacement", "horsepower", "weight", "accel", "model_year"]

        # Create a figure with multiple subplots
        fig, axs = plt.subplots(len(features), figsize=(8, 20))

        for i, feature in enumerate(features):
            axs[i].scatter(df[feature], df["mpg"])
            axs[i].set_xlabel(feature)
            axs[i].set_ylabel("MPG")
            axs[i].set_title(f"{feature} over MPG")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame):
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        mask = np.triu(corr)
        heatmap = sns.heatmap(corr, annot=True, mask=mask, fmt='.2f')
        heatmap.set_xticklabels(corr.columns, fontsize=13)
        heatmap.set_yticklabels(corr.columns, fontsize=13)
        plt.title("Pearson correlation coefficient across attributes")
        plt.subplots_adjust(bottom=0.19)
        plt.subplots_adjust(left=0.19)
        plt.show()


def main():

    # Read the data
    file_path = "auto-mpg.data"
    df_cars = CarsUtils.read_cars_from_file(file_path)
    columns_to_normalise = ["mpg", "displacement", "horsepower", "weight", "accel"]

    # Perform one-hot encoding on the categorical origin variable
    origin = df_cars.pop("origin")
    df_cars["USA"] = (origin == 1) * 1.0
    df_cars["Europe"] = (origin == 2) * 1.0
    df_cars["Japan"] = (origin == 3) * 1.0

    df_cars.pop("name")

    # Split the data
    train_data, test_data = train_test_split(df_cars, test_size=0.20, random_state=42)

    # Inspect the data
    train_stats = train_data.describe()
    train_stats.pop("mpg")
    train_stats = train_stats.transpose()

    # Normalise the training dataset
    CarsUtils.normalise_data(train_data, columns_to_normalise)

    # Plot correlation heatmap
    CarsUtils.plot_correlation_heatmap(train_data)
    CarsUtils.plot_features_vs_mpg(train_data)

    return 0


if __name__ == "__main__":
    main()
