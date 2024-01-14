import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CarUtils:
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


def main():

    # Read the data
    file_path = 'auto-mpg.data'
    df_cars = CarUtils.read_cars_from_file(file_path)

    print(df_cars.to_string())

    return 0


if __name__ == '__main__':
    main()
