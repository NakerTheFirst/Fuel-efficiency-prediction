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
    def convert_mpg_to_litres_per_100km(df):
        # Conversion factor: 1 mpg = 235.214583 litres per 100km
        df['litres_per_100km'] = round(235.214583 / df['mpg'], 2)
        return df

    @staticmethod
    def normalise_data(df, columns) -> pd.DataFrame:
        scaler = preprocessing.MinMaxScaler()

        # Apply normalization to specified columns
        df[columns] = scaler.fit_transform(df[columns])
        return df

    @staticmethod
    def plot_features_vs_mpg(df, features, size=(6, 14)):
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(len(features), figsize=size)

        for i, feature in enumerate(features):
            axs[i].scatter(df[feature], df["mpg"], color="#0487c4")
            axs[i].set_xlabel(feature)
            axs[i].set_ylabel("MPG")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_vs_mpg(df, feature, label_name):
        plt.figure(figsize=(6, 3.5))
        sns.scatterplot(x=feature, y="litres_per_100km", data=df, color="#0487c4")
        plt.xlabel(label_name, fontsize=13)
        plt.ylabel("Litry na 100 km", fontsize=13)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame):
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        mask = np.triu(corr)
        heatmap = sns.heatmap(corr, annot=True, mask=mask, fmt=".2f")
        heatmap.set_xticklabels(corr.columns, fontsize=13)
        heatmap.set_yticklabels(corr.columns, fontsize=13)
        plt.subplots_adjust(bottom=0.25)
        plt.subplots_adjust(left=0.19)
        plt.show()

    @staticmethod
    def plot_histogram(df: pd.DataFrame, feature_name: str, label_name):
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(x=feature_name, data=df, color="#0487c4")
        ax.bar_label(ax.containers[0], label_type="edge")
        plt.xlabel(label_name, fontsize=13)
        plt.ylabel("Liczba samochodów", fontsize=13)
        plt.show()

    @staticmethod
    def plot_line(df: pd.DataFrame, x_feature_name: str, y_feature_name: str, x_label: str, y_label="Litry na 100 km"):
        plt.figure(figsize=(8, 5))
        plt.xlabel(x_label, fontsize=13)
        plt.ylabel(y_label, fontsize=13)
        sns.lineplot(x=x_feature_name, y=y_feature_name, data=df, color="#0487c4")
        plt.show()


def main():

    # General preprocessing
    file_path = "auto-mpg.data"
    df_cars_raw = CarsUtils.read_cars_from_file(file_path)
    df_cars = CarsUtils.convert_mpg_to_litres_per_100km(df_cars_raw)
    origin = df_cars.pop("origin")
    df_cars.pop("name")
    df_cars.pop("mpg")

    cols_order = ["litres_per_100km", "cylinders", "displacement", "horsepower", "weight", "accel", "model_year"]
    columns_to_normalise = ["litres_per_100km", "displacement", "horsepower", "weight", "accel"]

    df_cars = df_cars.reindex(columns=cols_order)

    # Perform one-hot encoding on the categorical origin variable
    df_cars["USA"] = (origin == 1) * 1.0
    df_cars["Europe"] = (origin == 2) * 1.0
    df_cars["Japan"] = (origin == 3) * 1.0

    # Split the data
    train_data, test_data = train_test_split(df_cars, test_size=0.20, random_state=42)

    # Inspect the data
    train_stats = train_data.describe()
    train_stats = train_stats.transpose()

    CarsUtils.normalise_data(train_data, columns_to_normalise)
    print(train_data.to_string())

    # Visualisations
    # CarsUtils.plot_correlation_heatmap(train_data)
    # CarsUtils.plot_feature_vs_mpg(train_data, "weight", "Waga pojazdu")
    # CarsUtils.plot_feature_vs_mpg(train_data, "model_year", "Rok produkcji")
    # CarsUtils.plot_histogram(train_data, "model_year", "Rok produkcji")
    CarsUtils.plot_line(train_data, "model_year", "litres_per_100km", "Rok produkcji")

    return 0


if __name__ == "__main__":
    main()
