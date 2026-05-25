import os
import pandas as pd

# ====================================
# LOAD AIS DATASET
# ====================================

df = pd.read_csv(
    "dataset/nari_dynamic.csv"
)

# ====================================
# PRINT COLUMNS
# ====================================

print(df.columns)

# ====================================
# SELECT REQUIRED COLUMNS
# ====================================

df = df[
    [
        'sourcemmsi',
        'lon',
        'lat',
        'speedoverground',
        'courseoverground',
        'trueheading',
        't'
    ]
]

# ====================================
# REMOVE MISSING VALUES
# ====================================

df = df.dropna()

# ====================================
# SORT BY TIME
# ====================================

df = df.sort_values(by='t')

# ====================================
# RENAME COLUMNS
# ====================================

df.columns = [
    'mmsi',
    'lon',
    'lat',
    'speed',
    'course',
    'heading',
    'timestamp'
]

# ====================================
# SAVE CLEAN DATASET
# ====================================

df.to_csv(
    "ship_trajectory_data.csv",
    index=False
)

print("\nDataset Prepared Successfully!")

print(df.head())
# # Path to OXTS data folder
# folder_path = r"E:.\dataset\nari_dynamic.csv"

# data = []

# files = sorted(os.listdir(folder_path))

# for file in files:

#     file_path = os.path.join(folder_path, file)

#     with open(file_path, 'r') as f:

#         values = f.readline().split()

#         lat = float(values[0])
#         lon = float(values[1])
#         yaw = float(values[5])

#         vf = float(values[8])

#         ax = float(values[11])
#         ay = float(values[12])
#         az = float(values[13])

#         data.append([lat, lon, yaw, vf, ax, ay, az])

# df = pd.DataFrame(data,
#                   columns=[
#                       'lat',
#                       'lon',
#                       'yaw',
#                       'vf',
#                       'ax',
#                       'ay',
#                       'az'
#                   ])

# df.to_csv("trajectory_data.csv", index=False)

# print(df.head())