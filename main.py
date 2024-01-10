import numpy as np
import matplotlib.pyplot as plt


class Car:
    def __init__(self, mpg: float, cylinders: int, displacement: float, horsepower: float, weight: float, accel: float,
                 model_year: int, origin: int, name: str):
        self.__mpg = mpg
        self.__cylinders = cylinders
        self.__displacement = displacement
        self.__horsepower = horsepower
        self.__weight = weight
        self.__accel = accel
        self.__model_year = model_year
        self.__origin = origin
        self.__name = name

    def __str__(self):
        return (f"mpg: {self.__mpg}, cylinders: {self.__cylinders}, displacement: {self.__displacement}, horsepower: "
                f"{self.__horsepower}, weight: {self.__weight}, accel: {self.__accel}, model year: {self.__model_year} "
                f"origin: {self.__origin}, name: {self.__name}")


def read_cars_from_file(filename):
    cars = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()

            # Extracting numeric values
            try:
                mpg, cylinders, displacement, horsepower, weight, accel, model_year, origin = map(float, parts[:8])
            except ValueError:
                continue

            model_year, origin = int(model_year), int(origin)

            # Extracting car name (last part)
            name = ' '.join(parts[8:]).replace('"', '')  # Removing quotes

            # Creating Car instance
            car = Car(mpg, cylinders, displacement, horsepower, weight, accel, model_year, origin, name)
            cars.append(car)
    return cars


def main():

    # Read the data
    file_path = 'auto-mpg.data'
    cars = read_cars_from_file(file_path)

    for x in cars:
        print(x)

    return 0


if __name__ == '__main__':
    main()
