from matplotlib.patches import Patch
import json
import os.path

class Stations:

    def __init__(self, quartier_json_path=None, stations_json_path=None):

        if not os.path.isfile("./station.json") or (quartier_json_path is None or stations_json_path is None):
            with open("./station.json", "r") as json_file:
                self.stations = json.loads(json_file)
        else:
            self.load_stations_from_json(quartier_json_path, stations_json_path)

    def load_stations_from_json(self,quartier_json_path, stations_json_path):
        with open(quartier_json_path, "r") as json_file:
            data = json.loads(json_file)
            self.polygon_ls = []
            self.polygon_index = []
            for region in data["features"]:
                self.polygon_ls.append(Patch.Polygon(region["geometry"]["coordinates"]))
                self.polygon_index.append(region["properties"]["Q_socio"])

        with open(stations_json_path, "r") as json_file:
            data = json.loads(json_file)
            self.stations = []
            for station in data["data"]["stations"]:
                id_region = self.get_id_region((station["lat"], station["lon"]))
                self.stations.append({
                    "id": station["short_name"],
                    "xy": (station["lat"], station["lon"]),
                    "quartier": {
                        "id": id_region,
                        "name": self.polygon_index[id_region]
                    }
                })
        with open("./station.json", "w") as json_file:
            json.dump(self.stations, json_file)

    def get_id_region(self, xy):
        for polygon_id, polygon in enumerate(self.polygon_ls):
            if polygon.contains(xy):
                return polygon_id



