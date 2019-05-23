import csv
import os
import json
import numpy as np
import datetime
from Stations import Stations
from sklearn.preprocessing import normalize

class PreTraitement:

    def __init__(self, filepath, save=False):

        self.index = {
            "date": 0,
            "temp": 1,
            "drew_pt": 2,
            "relat_hum": 3,
            "wind_dir": 4,
            "wind_speed": 5,
            "visibility": 6,
            "visib_indic": 7,
            "pressure": 8,
            "hmdx": 9,
            "wind_chill": 10,
            "weather": 11,
            "public_holy": 12,
            "station_code": 13,
            "withdrawals": 14,
            "volume": 15
        }

        self.geo_region = Stations("quartierssociologiques2014.json")
        if os.path.isfile("data_" + filepath + ".json") and save==False:
            with open("data_" + filepath + ".json", "r") as json_file:
                self.data = json.load(json_file)
        else:
            self.data = []
            self.load_data(filepath)
            with open("data_" + filepath + ".json", "w") as json_file:
                 json.dump(self.data, json_file)


    def get_OH_date(self, date):
        year, month, day = (int(x) for x in date.split('-'))
        data_day = datetime.date(year, month, day).weekday()
        return  [0 if i != data_day else 1 for i in range(7)]

    def get_OH_hour(self, hour):
        hour_bornes = [6, 9, 11.30, 14, 16.30, 19, 21]
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
        index = int(wind_dir//4.5)
        return [1 if i == index else 0 for i in range(8)]

    def get_OH_pressure(self, pressure):
        pressure_interval = [980, 1000, 1025, 1045]
        pressure_one_hot = [0 for _ in range(len(pressure_interval + 1))]
        if int(pressure) < pressure_interval[0]:
            pressure_one_hot[0] = 1
        elif int(pressure) > pressure_interval[-1]:
            pressure_one_hot[-1] = 1
        else:
            for i, _ in enumerate(pressure_interval):
                if pressure > pressure_interval[i] and pressure < pressure_interval[i + 1]:
                    pressure_one_hot[i] = 1
                    break
        return pressure_one_hot

    def get_OH_region(self, station_code, geo_region_obj):
        for station in geo_region_obj.stations:
            if station_code == station["id"]:
                return [1 if station["quartier"]["id"] == index else 0 for index in range(len(geo_region_obj.region_ls))]

    def get_OH_weather(self, str_ls, values):
        return [1 if str in values else 0 for str in str_ls]


    def count_total_line(self, filename):

        def _make_gen(reader):
            b = reader(1024 * 1024)
            while b:
                yield b
                b = reader(1024 * 1024)

        f = open(filename, 'rb')
        f_gen = _make_gen(f.raw.read)
        return sum(buf.count(b'\n') for buf in f_gen)

    def normalize(self, array):
        return array

    def load_data(self, filepath):

        nb_lines = self.count_total_line(filepath)

        with open(filepath, "r") as trainig_file:
            csv_reader = csv.reader(trainig_file)
            headers = next(csv_reader)
            weather_list = []

            self.data = {
                "date_OH": [],
                "hour_OH": [],
                "wind_dir_OH": [],
                "region_OH": [],
                # median
                "temp": [],
                # median
                "drew_pt": [],
                # median
                "relat_hum": [],
                # median
                "wind_speed": [],
                "visibility": [],
                "visib_indic": [],
                "hmdx": [],
                "wind_chill": [],
                "weather": [],
                "public_holy": [],
                "withdrawals": [],
                "volume": []
            }

            column_to_add_auto = ["temp", "drew_pt", "relat_hum", "relat_hum", "wind_speed", "visibility",
                                  "visib_indic", "hmdx", "wind_chill", "public_holy", "withdrawals", "volume"]

            for row in csv_reader:

                date, hour = row[self.index["date"]].split(' ')

                self.data["date_OH"].append(self.get_OH_date(date))
                self.data["hour_OH"].append(self.get_OH_hour(hour))
                self.data["wind_dir_OH"].append(row[self.index["wind_dir"]])
                self.data["region_OH"].append(self.get_OH_region(row[self.index["station_code"]], self.geo_region))
                self.data["weather"].append(row[self.index["weather"]].split(","))
                for weather_name in row[self.index["weather"]].split(","):
                    if weather_name not in weather_list:
                        weather_list.append(weather_name)

                for column in column_to_add_auto:
                    self.data[column].append(row[self.index[column]].replace(",", "."))

                ratio = float(csv_reader.line_num) / float(nb_lines)
                len_compt = 30
                print("\r data generation [{}]{}%".format("=" * (int(len_compt * ratio) - 1) + ">" + "-" * int(len_compt * (1-ratio)),
                                                   int(ratio * 100)), end="")

                if csv_reader.line_num > nb_lines/50:
                    break

            column_to_normalize = ["temp", "drew_pt", "relat_hum", "relat_hum", "wind_speed",
                                        "visibility", "visib_indic", "hmdx", "wind_chill", "withdrawals"]
            for i, column in enumerate(column_to_normalize):
                print("\ncolumn: " + str(column))
                print("test array: " + str(np.array(self.data[column])[:10]))
                self.data[column] = normalize(np.array(self.data[column]))

                nb_column = len(column_to_normalize)
                ratio = float(i) / float(nb_column)
                len_compt = 30
                print("\r normalisation [{}]{}%".format(
                    "=" * (int(len_compt * ratio) - 1) + ">" + "-" * int(len_compt * (1 - ratio)),
                    int(ratio * 100)), end="")

            self.data["weather"] = [self.get_OH_weather(weathers) for weathers in self.data["weather"]]

            print("end of pre_traitement")


if __name__ == "__main__":
    data = PreTraitement("training.csv", True)