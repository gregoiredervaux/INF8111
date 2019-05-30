import csv
import os
import re
import numpy as np
import datetime
from Stations import Stations
from sklearn.preprocessing import normalize


class PreTraitement:

    def __init__(self):

        self.geo_region = Stations("quartierssociologiques2014.json")
        self.data_matrix = np.array([])
        self.raw_header = []
        self.header_flat = []
        self.transformed_matrix_flat = []
        self.transformed_matrix = np.array([])
        self.data_structure = {
            "date_OH": None,
            "hour_OH": {
                "bornes": [6, 9, 11.30, 14, 16.30, 19, 21]
            },
            "wind_dir_OH": {
                "division": 4.5
            },
            "region_OH": {
                "stations": self.geo_region.stations,
                "regions": self.geo_region.region_ls
            },
            "pressure": {
                "bornes": [98.0, 100.0, 102.5, 104.5]
            },
            "weather": {
                "list": []
            },
            "Public Holiday": None,
            "Volume": None
        }

        self.column_to_normalize = ["Temperature (°C)", "Drew point (°C)", "Relativite humidity (%)",
                                    "Wind speed (km/h)", "Visibility (km)", "hmdx", "Withdrawals"]

        self.column_to_add_auto = ["Temperature (°C)", "Drew point (°C)", "Relativite humidity (%)",
                                   "Wind speed (km/h)", "Visibility (km)", "hmdx", "Withdrawals", "Public Holiday",
                                   "Volume"]

    def get_OH_date(self, date):
        year, month, day = (int(x) for x in date.split('-'))
        data_day = datetime.date(year, month, day).weekday()
        return [0 if i != data_day else 1 for i in range(7)]

    def get_OH_hour(self, hour):
        hour_bornes = self.data_structure["hour_OH"]["bornes"]
        hour_int = float(hour.replace(":", "."))
        hour_one_hot = [0 for _ in hour_bornes]
        if hour_int < hour_bornes[0] or hour_int > hour_bornes[-1]:
            hour_one_hot[0] = 1
        else:
            for i, _ in enumerate(hour_bornes):
                if hour_int > hour_bornes[i] and hour_int < hour_bornes[i + 1]:
                    hour_one_hot[i] = 1
                    break
        return hour_one_hot

    def get_OH_wind_dir(self, wind_dir):
        try:
            index = int(float(wind_dir) // self.data_structure["wind_dir_OH"]["division"])
        except:
            index = -1
        return [1 if i == index else 0 for i in range(8)]

    def get_OH_pressure(self, pressure):

        pressure_interval = self.data_structure["pressure"]["bornes"]
        pressure_one_hot = [0 for _ in range(len(pressure_interval))]
        if pressure == '':
            return  pressure_one_hot
        else:
            pressure = float(pressure.replace(',', '.'))
            if pressure < pressure_interval[0]:
                pressure_one_hot[0] = 1
            elif pressure > pressure_interval[-1]:
                pressure_one_hot[-1] = 1
            else:
                for i, _ in enumerate(pressure_interval):
                    if pressure > pressure_interval[i] and pressure < pressure_interval[i + 1]:
                        pressure_one_hot[i] = 1
                        break
            return pressure_one_hot

    def get_OH_region(self, station_code):
        nb_region = len(self.data_structure["region_OH"]["regions"])
        for station in self.data_structure["region_OH"]["stations"]:
            if station_code == station["id"]:
                return [1 if station["quartier"]["id"] == index else 0 for index in range(nb_region)]

    def get_OH_weather(self, weather):
        return [1 if str in weather else 0 for str in self.data_structure["weather"]["list"]]

    def get_caract(self, column):

        column_full = []
        for i, value in enumerate(column):
            if value != '':
                column_full.append(float(value.replace(",", ".")))
                column[i] = float(value.replace(",", "."))

        column_full = np.array(column_full)
        return np.median(column_full), np.mean(column_full), np.std(column_full)

    def fill_zero_by_median(self, column, mediane):

        for i, value in enumerate(column):
            if value == '' or value == 'nan':
                column[i] = str(mediane)
        return column

    def normalise(self, column, mean, std):

        normalized_column = []
        for item in column:
            normalized_column.append((float(item) - mean) / std)
        return np.array(normalized_column)

    def load_data(self, filepath):

        with open(filepath + ".csv", "r", encoding='UTF-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            self.raw_header = next(csv_reader)
            self.data_matrix = []
            i = 0
            for row in csv_reader:
                self.data_matrix.append(row)
                i += 1
                if i > 10000:
                    break

        self.data_matrix = np.array(self.data_matrix)


    def fit(self):

        # extraction des data du document et construction des OH vectors
        for row in self.data_matrix:
            index_weather = self.raw_header.index("Weather")
            for weather_name in row[index_weather].split(","):
                if weather_name not in self.data_structure["weather"]["list"]:
                    self.data_structure["weather"]["list"].append(weather_name)

            # if csv_reader.line_num > nb_lines/100: break

        # remplissages des valeures manquantes avant normalisation


        for column in self.column_to_add_auto:
            index = self.raw_header.index(column)
            mediane, mean, std = self.get_caract(self.data_matrix[:, index])
            self.data_structure[column] = {
                "mediane": mediane,
                "mean": mean,
                "std": std
            }

        print("\nend of setting")

    def transform(self):

        self.transformed_matrix = []
        index_to_del = []

        # extraction des data du document et construction des OH vectors
        for row in self.data_matrix:

            tranformed_row = []

            for index, column_name in enumerate(self.raw_header):
                if column_name == "Date/Hour":
                    date, hour = row[index].split(' ')
                    tranformed_row.append([self.get_OH_date(date), self.get_OH_hour(hour)])
                elif column_name == "wind direction (10s deg)":
                    tranformed_row.append(self.get_OH_wind_dir(row[index]))
                elif column_name == "Station Code":
                    tranformed_row.append(self.get_OH_region(row[index]))
                elif column_name == "Weather":
                    tranformed_row.append(self.get_OH_weather(row[index].split(",")))
                elif column_name == "Pressure at the station (kPa)":
                    tranformed_row.append(self.get_OH_pressure(row[index]))
                elif column_name in self.column_to_add_auto:
                    tranformed_row.append(row[index].replace(",", "."))
                else:
                    if index not in index_to_del:
                        index_to_del.append(index)

            self.transformed_matrix.append(tranformed_row)

        index_to_del.reverse()
        for index in index_to_del:
            del self.raw_header[index]

        self.transformed_matrix = np.array(self.transformed_matrix)
        # remplissages des valeures manquantes avant normalisation

        for column_name in self.column_to_add_auto:

            index_column = self.raw_header.index(column_name)
            column = self.transformed_matrix[:, index_column]
            mediane = self.data_structure[column_name]["mediane"]
            mean = self.data_structure[column_name]["mean"]
            std = self.data_structure[column_name]["std"]

            self.transformed_matrix[:, index_column] = self.fill_zero_by_median(column, mediane)
            normalised_column = self.normalise(self.transformed_matrix[:, index_column], mean, std)
            self.transformed_matrix[:, index_column] = [[result] for result in normalised_column]

        print("\nend of pre_traitement")

    def save_matrix(self, name):

        ls_flat_column = []
        for index in range(self.transformed_matrix.shape[1]):
            column = self.transformed_matrix[:, index].tolist()
            flat_column  = np.array(column).reshape(len(column), -1)
            for i in range(flat_column.shape[1]):
                self.header_flat.append(self.raw_header[index] + str(i) if flat_column.shape[1] != 1 else self.raw_header[index])
            ls_flat_column.append(flat_column)

        self.transformed_matrix_flat = np.column_stack(ls_flat_column)

        np.savetxt("data_" + name + ".csv", self.transformed_matrix_flat, header=str(self.header_flat), delimiter=",",
                   fmt="%.3e")

    def load_from_save(self, name):
        with open("data_" + name + ".csv", "r") as file:
            str_header = file.readline()
            self.raw_header = re.sub(r'[\[\]\s\'#]', '', str_header).split(",")
        self.transformed_matrix_flat = np.loadtxt("data_" + name + ".csv", delimiter=",")

if __name__ == "__main__":
    pretrait = PreTraitement()
    pretrait.load_data("training")
    pretrait.fit()
    pretrait.transform()
    pretrait.save_matrix("training_test")
    pretrait.load_from_save("training_test")


