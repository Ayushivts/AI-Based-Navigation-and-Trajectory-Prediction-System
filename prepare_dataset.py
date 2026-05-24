import os
import pandas as pd

# Path to OXTS data folder
folder_path = r"E:.\dataset\oxts_02\data"

data = []

files = sorted(os.listdir(folder_path))

for file in files:

    file_path = os.path.join(folder_path, file)

    with open(file_path, 'r') as f:

        values = f.readline().split()

        lat = float(values[0])
        lon = float(values[1])
        yaw = float(values[5])

        vf = float(values[8])

        ax = float(values[11])
        ay = float(values[12])
        az = float(values[13])

        data.append([lat, lon, yaw, vf, ax, ay, az])

df = pd.DataFrame(data,
                  columns=[
                      'lat',
                      'lon',
                      'yaw',
                      'vf',
                      'ax',
                      'ay',
                      'az'
                  ])

df.to_csv("trajectory_data.csv", index=False)

print(df.head())