import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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
    def plot_feature_vs_mpg(df, feature, label_name):
        plt.figure(figsize=(6, 3.5))
        sns.scatterplot(x=feature, y="litres_per_100km", data=df, color="#0487c4")
        plt.xlabel(label_name, fontsize=13)
        plt.ylabel("Litry na 100 km", fontsize=13)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, labels):
        x_labels = labels[:7]
        y_labels = list(x_labels)
        x_labels[-1] = ""
        y_labels[0] = ""

        plt.figure(figsize=(10, 8))
        corr = df.corr()
        mask = np.triu(corr)
        heatmap = sns.heatmap(corr, annot=True, mask=mask, fmt=".2f")
        heatmap.set_xticklabels(x_labels, fontsize=13)
        heatmap.set_yticklabels(y_labels, fontsize=13)
        plt.subplots_adjust(bottom=0.25)
        plt.subplots_adjust(left=0.19)
        plt.show()

    @staticmethod
    def plot_line(df: pd.DataFrame, x_feature_name: str, y_feature_name: str, x_label: str, y_label="Litry na 100 km"):
        plt.figure(figsize=(8, 5))
        plt.xlabel(x_label, fontsize=13)
        plt.ylabel(y_label, fontsize=13)
        sns.lineplot(x=x_feature_name, y=y_feature_name, data=df, color="#0487c4")
        plt.show()

    @staticmethod
    def plot_point(df: pd.DataFrame, x_feature_name: str, y_feature_name: str, x_label: str, y_label: str):
        plt.figure(figsize=(8, 5))
        sns.pointplot(x=x_feature_name, y=df[y_feature_name], data=df, color="#0487c4")
        plt.xlabel(x_label, fontsize=13)
        plt.ylabel(y_label, fontsize=13)
        plt.show()

    @staticmethod
    def plot_predicted_vs_true(test_predictions, test_true):
        sns.scatterplot(x=test_true, y=test_predictions, color="#0487c4")
        # plt.scatter(test_true, test_predictions)
        plt.xlabel('Prawdziwe wartości [Litry na 100 km]', fontsize=13)
        plt.ylabel('Predyktory [Litry na 100 km]', fontsize=13)
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0, plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]])
        plt.plot([-100, 100], [-100, 100])
        plt.show()

    @staticmethod
    def plot_regular_histogram(feature):
        plt.figure(figsize=(8, 5))
        sns.histplot(feature, bins=20, color='#0487c4')
        plt.xlabel("Wartości błędu predykcji [Litry na 100 km]", fontsize=13)
        plt.ylabel("Liczba występowań", fontsize=13)
        plt.show()


def main():
    # General preprocessing
    file_path = "auto-mpg.data"
    df_cars_raw = CarsUtils.read_cars_from_file(file_path)
    df_cars = CarsUtils.convert_mpg_to_litres_per_100km(df_cars_raw)
    origin = df_cars.pop("origin")
    df_cars.pop("name")
    df_cars.pop("mpg")

    labels = ["Litry na 100 km", "Liczba cylindrów", "Objętość cylindra", "Liczba KM", "Waga pojazdu",
              "Przyspieszenie", "Rok produkcji", "Stany Zjednoczone", "Europa", "Japonia"]
    cols_order = ["litres_per_100km", "cylinders", "displacement", "horsepower", "weight", "accel", "model_year"]
    columns_to_normalise = ["displacement", "horsepower", "weight", "accel"]
    df_cars = df_cars.reindex(columns=cols_order)

    # Perform one-hot encoding on the categorical origin variable
    df_cars["USA"] = (origin == 1) * 1.0
    df_cars["Europe"] = (origin == 2) * 1.0
    df_cars["Japan"] = (origin == 3) * 1.0

    train_data, test_data = train_test_split(df_cars, test_size=0.20, random_state=42)
    train_labels = train_data.pop("litres_per_100km")
    test_labels = test_data.pop("litres_per_100km")

    CarsUtils.normalise_data(train_data, columns_to_normalise)
    CarsUtils.normalise_data(test_data, columns_to_normalise)

    # Build & train the model
    linear_model = LinearRegression()
    linear_model.fit(train_data, train_labels)

    # Extracting coefficients
    coefficients = linear_model.coef_
    intercept = linear_model.intercept_
    print("Coefficients:", coefficients)
    print("Intercept:", intercept)

    # Evaluate the model on a test set
    test_predictions = linear_model.predict(test_data)
    mse = sklearn.metrics.mean_squared_error(test_labels, test_predictions)
    print("Mean Squared Error on Test Set: ", round(mse, 2))

    error = test_predictions - test_labels

    # Visualisations
    # CarsUtils.plot_line(train_data, "model_year", "litres_per_100km", "Rok produkcji")
    # CarsUtils.plot_point(train_data, origin, "litres_per_100km", "Miejsce pochodzenia", labels[0])
    # CarsUtils.plot_point(train_data, origin, "horsepower", "Miejsce pochodzenia", labels[3])
    # CarsUtils.plot_predicted_vs_true(test_predictions, test_labels)
    # CarsUtils.plot_regular_histogram(error)

    return 0


if __name__ == "__main__":
    main()
