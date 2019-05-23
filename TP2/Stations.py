from matplotlib import patches
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import json
import random

class Stations:

    def __init__(self, quartier_json_path="./quartierssociologiques2014.json",
                 stations_json_path="./station_information.json"):

        self.region_ls = []
        self.stations = []
        self.load_stations_from_json(quartier_json_path, stations_json_path)

    def load_stations_from_json(self, quartier_json_path, stations_json_path):
        with open(quartier_json_path, "r") as json_file:
            data = json.load(json_file)
            for region in data["features"]:
                polygones = self.get_recur_polygon(region["geometry"]["coordinates"])
                self.region_ls.append({
                    "id": region["properties"]["id"],
                    "name": region["properties"]["Q_socio"],
                    "polygon": polygones})
            print("nb de polygones: " + str(len(self.region_ls)))

        fig, ax = plt.subplots()
        flatten_polygon = []
        for region in np.array(self.region_ls):
            for polygon in region["polygon"]:
                flatten_polygon.append(polygon)
        p = PatchCollection(flatten_polygon)
        ax.add_collection(p)

        with open(stations_json_path, "r") as json_file:
            data = json.load(json_file)
            for station in data["data"]["stations"]:
                region = self.region_ls[self.get_index_region((station["lon"], station["lat"]))]
                id_region, name_region = region["id"], region["name"]
                self.stations.append({
                    "id": station["short_name"],
                    "xy": (station["lat"], station["lon"]),
                    "quartier": {
                        "id":  None if id_region is None else id_region,
                        "name": None if id_region is None else name_region
                    }
                })

        plt.scatter([station["xy"][1] for station in self.stations], [station["xy"][0] for station in self.stations], c='green', marker=".")
        station_out=[[],[]]
        for station in self.stations:
            if station["quartier"]["id"] is None:
                station_out[0].append(station["xy"][1])
                station_out[1].append(station["xy"][0])

        plt.scatter(station_out[0], station_out[1], c='red', marker=".")
        plt.autoscale()
        #plt.show()


        with open("./stations.json", "w") as json_file:
            json.dump(self.stations, json_file)

    def get_index_region(self, xy):
        for region_index, region in enumerate(self.region_ls):
            for polygon in region["polygon"]:
                if polygon.contains_point(xy):
                    return region_index

    def get_recur_polygon(self, coords, polygones=None):

        if polygones is None:
            polygones = []

        if len(coords[0]) == 2:
            polygones.append(patches.Polygon(coords))

        else:
            for coords_next in coords:
                self.get_recur_polygon(coords_next, polygones)
        return polygones

    def test_station(self):
        station_chosen = random.choice(self.stations)
        region_chosen = station_chosen["quartier"]
        print("on choisi la région: " + region_chosen["name"] + " d'id: " + str(region_chosen["id"]))
        ls_stations = []
        for station in self.stations:
            if station["quartier"]["id"] == region_chosen["id"]:
                print("station: " + str(station["id"]) + " car appartient à: " + str(station["quartier"]["id"]) + " de nom " + str(station["quartier"]["name"]))
                print("coord: " + str(station["xy"]))
                ls_stations.append(station)

        fig, ax = plt.subplots()
        print("polygone selected: " + str([polygon.get_xy() for polygon in self.region_ls[region_chosen["id"]]["polygon"]]))
        p = PatchCollection([polygon for polygon in self.region_ls[region_chosen["id"]]["polygon"]])
        ax.add_collection(p)

        x = [station["xy"][1] for station in ls_stations]
        y = [station["xy"][0] for station in ls_stations]

        plt.scatter(x, y, c='red', marker=".")
        plt.autoscale()
        plt.show()


if __name__ == "__main__":
    station_obj = Stations("./quartierssociologiques2014.json","./station_information.json")
    station_obj.test_station()
