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
        self.raw_header_transform =[]
        self.header = []
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
                "bornes": [980, 1000, 1025, 1045]
            },
            "weather": {
                "list": []
            },
            "public_holy": None,
            "volume": None
        }

        self.column_to_normalize = ["Temperature (°C)", "Drew point (°C)", "Relativite humidity (%)",
                                    "Wind speed (km/h)", "Visibility (km)", "hmdx", "Withdrawals"]
        self.column_to_add_auto = ["Temperature (°C)", "Drew point (°C)", "Relativite humidity (%)", "Wind speed (km/h)",
                              "Visibility (km)", "hmdx", "Withdrawals", "Public Holiday", "Volume"]

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

    def get_OH_region(self, station_code):
        #print("Entrée dans get_OH_region")
        #print(station_code)
        nb_region = len(self.data_structure["region_OH"]["regions"])
        for station in self.data_structure["region_OH"]["stations"]:
            if station_code == station["id"]:
                #print("Entrée dans if")
                #print([1 if station["quartier"]["id"] == index else 0 for index in range(nb_region)])
                return [1 if station["quartier"]["id"] == index else 0 for index in range(nb_region)]

    def get_OH_weather(self, weather):
        return [1 if str in weather else 0 for str in self.data_structure["weather"]["list"]]

    def get_caract(self, column):

        column_full = []
        for i, value in enumerate(column):
            if value != '':
                #Attention au type si on cast avant d'entrée dans cette fonction
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

        normalized_column = np.zeros((column.shape))
        for i in range(len(column)):
            normalized_column[i] = ((float(column[i]) - mean) / std)
        return normalized_column


    """def load_data(self, filepath):

        with open(filepath + ".csv", "r", encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            self.raw_header = next(csv_reader)
            print("header: " + str(self.raw_header))
            self.data_matrix = []
            i = 0
            for row in csv_reader:
                self.data_matrix.append(row)
                i += 1
                if i > 10000:
                    break

        self.data_matrix = np.array(self.data_matrix)"""
        
        
    def load_data(self, filepath):
        with open(filepath + '.csv', "r", encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            self.raw_header = next(csv_reader)
            print("header: " + str(self.raw_header))
            self.data_matrix = []
            line_count = 0
            volume_eq_1 = 0
            volume_eq_0 =0
            list_row = []
            row_vol_1 = []
            for row in csv_reader:
                #TODO : Est ce qu'on prends la premièer ligne???
                if line_count != 0:
                    #TODO : gestion des dates 
                    #if row[0]:
                    #    row[0] = np.datetime64(row[0])
                    for i in range (len(row)):
                        if re.match("^\d+?\,\d+?$", row[i]):
                            row[i] = row[i].replace(",",".")
                            row[i] = float(row[i])
                        elif re.match("^\d+?$", row[i]):
                            row[i] = row[i].replace(",",".")
                            row[i] = int(row[i])
                    list_row.append(row)
                    if row[self.raw_header.index("Volume")] == 1 :
                        volume_eq_1 += 1
                        row_vol_1.append(row)
                        #news_row = self.perturb_row(row)
                        #list_row = list_row + news_row
                    else : 
                        volume_eq_0 += 1
                        
                line_count += 1
                if line_count > 1000:
                    break
            print("LOAD DATA : " + str(list_row[0][1].__class__))
            #TODO: Type dans le numpy déjà correct
            self.data_matrix = np.asarray(list_row)
            print("LOAD DATA : " + str(self.data_matrix[0][1].__class__))
            return self.data_matrix
    
    def perturb_row(self, row):
        #print(row.__class__)
        #print(self.index["temp"].__class__)
        new_rows = []
        """if row[self.index["date"]]: Ne fonctionne pas
            row_cp = row.copy()
            heure = int(random.uniform(0,2))
            row_cp[self.index["date"]] += heure*60*60*1000000
            #row_cp[self.index["date"]] = np.datetime64(row_cp[self.index["date"]])
            new_rows.append(row_cp)"""
        if row[self.index["temp"]]:
            row_cp = row.copy()
            row_cp[self.index["temp"]] += (-1)**(random.randint(0,1))*random.uniform(0,1)
            new_rows.append(row_cp)
        if row[self.index["drew_pt"]]:
            row_cp = row.copy()
            row_cp[self.index["drew_pt"]] += random.uniform(0,1)
            new_rows.append(row_cp)
        if row[self.index["relat_hum"]]:
            row_cp = row.copy()
            row_cp[self.index["relat_hum"]] += random.uniform(0,1)
            new_rows.append(row_cp)
        if row[self.index["wind_dir"]]:#one_hot
            row_cp = row.copy()
            row_cp[self.index["wind_dir"]] += int(random.uniform(0,5))
            new_rows.append(row_cp)
        if row[self.index["wind_speed"]]:
            row_cp = row.copy()
            row_cp[self.index["wind_speed"]] += random.uniform(0,1)
            new_rows.append(row_cp)
        if row[self.index["visibility"]]:
            row_cp = row.copy()
            row_cp[self.index["visibility"]] += random.uniform(0,1)
            new_rows.append(row_cp)
        if row[self.index["visility_indicator"]]:
            row_cp = row.copy()
            row_cp[self.index["visility_indicator"]] += random.uniform(0,1)
            new_rows.append(row_cp)
        """if row[self.index["pressure"]]:#
            row_cp = row.copy()
            row[self.index["pressure"]] += random.uniform(0,10)
            new_rows.append(row_cp)"""
        if row[self.index["hmdx"]]:
            row_cp = row.copy()
            row_cp[self.index["hmdx"]] += random.uniform(0,1)
            new_rows.append(row_cp)
        if row[self.index["wind_chill"]]:
            row_cp = row.copy()
            row_cp[self.index["wind_chill"]] += random.uniform(0,1)
            new_rows.append(row_cp)
        """if row[self.index["weather"]]:#
            row[self.index["weather"]] += random.uniform(0,1)
            new_rows.append(row_cp)"""
        if row[self.index["public_holy"]]:
            row_cp = row.copy()
            row_cp[self.index["public_holy"]] += random.uniform(0,1)
            new_rows.append(row_cp)
        """if row[self.index["station_code"]]:#
            row[self.index["station_code"]] += random.uniform(0,1)
            new_rows.append(row_cp)"""
        if row[self.index["withdrawals"]]:
            row_cp = row.copy()
            row_cp[self.index["withdrawals"]] += random.uniform(0,1)
            new_rows.append(row_cp)
        return new_rows


    def fit(self):
        #Enregistre informations utiles pour les OH et les normalisations dans data_structure
        # extraction des data du document et construction des OH vectors
        for row in self.data_matrix:
            index_weather = self.raw_header.index("Weather")
            #Initialisation de data_structure pour Weather, avec tout les weather possible
            for weather_name in row[index_weather].split(","):
                if weather_name not in self.data_structure["weather"]["list"]:
                    self.data_structure["weather"]["list"].append(weather_name)

            # if csv_reader.line_num > nb_lines/100: break

        # remplissages des valeures manquantes avant normalisation


        for column in self.column_to_normalize:
            index = self.raw_header.index(column)
            #On enregistre les informations pour chaque features (colonne) dans data_structure
            mediane, mean, std = self.get_caract(self.data_matrix[:, index])
            self.data_structure[column] = {
                "mediane": mediane,
                "mean": mean,
                "std": std
            }
        #print(self.raw_header)
        print("\nend of setting")
        self.raw_header_transform = self.get_header(self.data_matrix[0])
        #print(self.raw_header_transform)

    
    
    def get_header(self, row):

        header = []
        #TODO : Separe date et heure
        #Old_header : ['Date/Hour', 'Temperature (°C)', 'Drew point (°C)', 'Relativite humidity (%)', 'wind direction (10s deg)', 'Wind speed (km/h)', 'Visibility (km)', 'Visility indicator', 'Pressure at the station (kPa)', 'hmdx', 'Wind Chill', 'Weather', 'Public Holiday', 'Station Code', 'Withdrawals', 'Volume']
        #header: ['Date/Hour', 'Temperature (°C)', 'Drew point (°C)', 'Relativite humidity (%)', 'wind direction (10s deg)', 'Wind speed (km/h)', 'Visibility (km)', 'Visility indicator', 'Pressure at the station (kPa)', 'hmdx', 'Wind Chill', 'Weather', 'Public Holiday', 'Station Code', 'Withdrawals', 'Volume']
        #print(self.raw_header.index("Date/Hour"))
        date, hour = row[self.raw_header.index("Date/Hour")].split(' ')
        for i in range(len(self.get_OH_date(date))):
            header.append("Date " + str(i))
        for i in range(len(self.get_OH_hour(hour))):
            header.append("Hour " + str(i))

        for i in range( len(self.get_OH_wind_dir(row[self.raw_header.index('wind direction (10s deg)')]))):
            header.append("Wind direction " + str(i))
            
        for i in range(len(self.get_OH_region(row[self.raw_header.index('Station Code')]))):
            header.append("Station Code " + str(i))
        for i in range(len(self.get_OH_weather(row[self.raw_header.index('Weather')].split(",")))):
            header.append("Weather " + str(i))
            
        for column in self.column_to_add_auto:
            header.append(column)

        return header
    
    def transform_rowbis(self, row):
        #Transform la row en entrée en row selon le format avec lequel on va travailler
        tranformed_row = []
        #TODO : Separe date et heure
        #header: ['Date/Hour', 'Temperature (°C)', 'Drew point (°C)', 'Relativite humidity (%)', 'wind direction (10s deg)', 'Wind speed (km/h)', 'Visibility (km)', 'Visility indicator', 'Pressure at the station (kPa)', 'hmdx', 'Wind Chill', 'Weather', 'Public Holiday', 'Station Code', 'Withdrawals', 'Volume']
        #print(self.raw_header.index("Date/Hour"))
        date, hour = row[self.raw_header.index("Date/Hour")].split(' ')
        
        tranformed_row =  np.append(tranformed_row, self.get_OH_date(date))
        tranformed_row =  np.append(tranformed_row, self.get_OH_hour(hour))
        tranformed_row =  np.append(tranformed_row, self.get_OH_wind_dir(row[self.raw_header.index('wind direction (10s deg)')]))
        tranformed_row =  np.append(tranformed_row, self.get_OH_region(row[self.raw_header.index('Station Code')]))
        #Rajoute liste ou concatenation?
        tranformed_row =  np.append(tranformed_row, self.get_OH_weather(row[self.raw_header.index('Weather')].split(",")))

        for column in self.column_to_add_auto:
            #replace , par . pour pouvoir les cast en float
            tranformed_row =  np.append(tranformed_row, row[self.raw_header.index(column)])

        return np.asarray([tranformed_row])

    def transform(self):
        #manque visibility indicator et pressure
        # initialisation du nouveau header

        #header = self.get_header(self.data_matrix[0])
        #print(header)
        #row = self.transform_rowbis(self.data_matrix[0])
        # extraction des data du document et construction des OH vectors
        c=0
        for row in self.data_matrix:
            #print(self.data_matrix.__class__)
            
            if self.transformed_matrix.shape[0] == 0:
                self.transformed_matrix = np.array(self.transform_rowbis(row))
            else :
                c+=1
                self.transformed_matrix = np.append(self.transformed_matrix,self.transform_rowbis(row), axis=0)
                #print(self.transformed_matrix.shape)


        
        #Transforme list de list en tableau
        self.transformed_matrix = np.array(self.transformed_matrix)
        # remplissages des valeures manquantes avant normalisation

        for i, column in enumerate(self.column_to_normalize):
            index = self.raw_header_transform.index(column)
            self.transformed_matrix[:, self.raw_header_transform.index(column)] = self.fill_zero_by_median(self.transformed_matrix[:, self.raw_header_transform.index(column)], self.data_structure[column]["mediane"])
            normalised_column = self.normalise(self.transformed_matrix[:, self.raw_header_transform.index(column)],
                                                                             self.data_structure[column]["mean"],
                                                                             self.data_structure[column]["std"])
            
            #Remplacer les colonnes par les colonnes normalisé
            #self.data[column] = [[result] for result in normalised_column]
            self.transformed_matrix[:, self.raw_header_transform.index(column)] = normalised_column

            nb_column = len(self.column_to_normalize)
            ratio = float(i) / float(nb_column)
            len_compt = 30
            print("\r normalisation [{}]{}%".format(
                "=" * (int(len_compt * ratio) - 1) + ">" + "-" * int(len_compt * (1 - ratio)),
                int(ratio * 100)), end="")

        #Changer type des colonnes!?!
        self.transformed_matrix[:, self.raw_header_transform.index("Public Holiday")] = np.asarray([[int(value)] for value in self.transformed_matrix[:, self.raw_header_transform.index("Public Holiday")]])[:,0]
        self.transformed_matrix[:, self.raw_header_transform.index("Volume")] = np.asarray([[int(value)] for value in self.transformed_matrix[:, self.raw_header_transform.index("Volume")]])[:,0]

        print("\nend of pre_traitement")

        #Verifier à quoi cela sert?
        #self.data_matrix = np.column_stack([np.array(column) for i, column in self.data.items()])

        def save_matrix(name):
            np.savetxt("data_" + name + ".csv", self.data_matrix, header=str(self.raw_header), delimiter=",",
                       fmt='%.3e')

        def load_from_save(name):
            with open("data_" + filepath + ".csv", "r") as file:
                str_header = file.readline()
                self.raw_header = re.sub(r'[\[\]\s\'#]', '', str_header).split(",")
                print("header: " + str(self.raw_header))
            self.data_matrix = np.loadtxt("data_" + filepath + ".csv", delimiter=",")

if __name__ == "__main__":
    pretrait = PreTraitement()
    pretrait.load_data("training")
    pretrait.fit()
    pretrait.transform()


