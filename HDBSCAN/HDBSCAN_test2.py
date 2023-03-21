import numpy as np
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import pi, cos, sin, atan2, sqrt


def get_centroid(cluster):
    x = y = z = 0
    coord_num = len(cluster)
    for coord in cluster:
        lat = coord[0] * pi / 180
        lon = coord[1] * pi / 180

        a = cos(lat) * cos(lon)
        b = cos(lat) * sin(lon)
        c = sin(lat)

        x += a
        y += b
        z += c
    x /= coord_num
    y /= coord_num
    z /= coord_num
    lon = atan2(y, x)
    hyp = sqrt(x * x + y * y)
    lat = atan2(z, hyp)
    return [lat * 180 / pi, lon * 180 / pi]


df = pd.read_excel('test.xlsx')

hotel_df = df[['latitude', 'longitude']]
hotel_df = hotel_df.dropna(axis=0, how='any')
hotel_coord = hotel_df.values

hotel_dbsc = hdbscan.HDBSCAN(metric='haversine', min_cluster_size=int(len(hotel_df) / 5)).fit(np.radians(hotel_coord))
hotel_df['labels'] = hotel_dbsc.labels_
hotel_df['probab'] = hotel_dbsc.probabilities_
hotel_df.loc[hotel_df['probab'] < 0.5, 'labels'] = -1  # remove noise
print(hotel_df)

cluster_list = hotel_df['labels'].value_counts(dropna=False)
center_coords = []
for index, item_count in cluster_list.iteritems():
    if index != -1:
        df_cluster = hotel_df[hotel_df['labels'] == index]
        center_coord = get_centroid(df_cluster[['latitude', 'longitude']].values)
        center_lat = center_coord[0]
        center_lon = center_coord[1]
        center_coords.append(center_coord)
center_coords = pd.DataFrame(center_coords, columns=['latitude', 'longitude'])
print(center_coords)

# plot
fig, ax = plt.subplots(figsize=[20, 12])
facility_scatter = ax.scatter(hotel_df['longitude'], hotel_df['latitude'], c=hotel_df['labels'], cmap=cm.Dark2,
                              edgecolor='None', alpha=0.7, s=120)
centroid_scatter = ax.scatter(center_coords['longitude'], center_coords['latitude'], marker='x', linewidths=2,
                              c='k', s=50)
ax.set_title('Facility Clusters & Facility Centroid', fontsize=30)
ax.set_xlabel('Longitude', fontsize=24)
ax.set_ylabel('Latitude', fontsize=24)
# ax.set_xlim(120, 122)
# ax.set_ylim(30, 33)
ax.legend([facility_scatter, centroid_scatter], ['Facilities', 'Facility Cluster Centroid'], loc='upper left',
          fontsize=20)
# plt.show()
