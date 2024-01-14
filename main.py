import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, KFold


class CarsUtils:
    @staticmethod
    def read_cars_from_file(filename) -> pd.DataFrame:
        car_list = []
        with open(filename, 'r') as file:
            for line in file:
                parts = line.split()
                try:
                    mpg, cylinders, displacement, horsepower, weight, accel, model_year, origin = map(float, parts[:8])
                except ValueError:
                    continue
                model_year, origin = int(model_year), int(origin)
                name = ' '.join(parts[8:]).replace('"', '')
                car_dict = {
                    'mpg': mpg,
                    'cylinders': cylinders,
                    'displacement': displacement,
                    'horsepower': horsepower,
                    'weight': weight,
                    'accel': accel,
                    'model_year': model_year,
                    'origin': origin,
                    'name': name
                }
                car_list.append(car_dict)
        return pd.DataFrame(car_list)

    @staticmethod
    def standardise_data(df) -> pd.DataFrame:
        scaler = preprocessing.StandardScaler()

        # Select columns to standardise
        numerical_cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'accel', 'model_year', 'origin']

        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df


def main():

    # Read the data
    file_path = 'auto-mpg.data'
    df_cars = CarsUtils.read_cars_from_file(file_path)

    # Split the data
    train_data, test_data = train_test_split(df_cars, test_size=0.20, random_state=42)

    # Standardise the training dataset
    train_data_standardised = CarsUtils.standardise_data(train_data)

    print(df_cars.to_string())

    return 0


if __name__ == '__main__':
    main()
