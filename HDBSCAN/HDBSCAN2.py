from data2 import origin_data as origin, test_data as test
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

sns.set_context('talk')
sns.set_style('white')
sns.set_color_codes()
mpl.rcParams['font.family'] = 'Times New Roman'
plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}


def normalize_param(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value) if max_value != min_value else 0.5


def distance(a, b):
    x, y = (a['input_params'], b['input_params']) if (type(a) == dict and type(b) == dict) else (a, b)
    return sum([(x[i] - y[i]) ** 2 for i in range(len(x))]) ** 0.5


def get_centroid(cluster):
    centroid = [0] * len(cluster[0]['input_params'])
    for x in cluster:
        for i in range(len(x['input_params'])):
            centroid[i] += x['input_params'][i]
    for i in range(len(centroid)):
        centroid[i] /= len(cluster)
    return centroid


data = []
for x in origin:
    data.append({
        'id': x['index'],
        'input_params': list(x.values())[1:8],
        'output_params': list(x.values())[-4:],
    })

test_case = {
    'input_params': list(test.values()),
}

# for i in range(len(data[0]['input_params'])):
#     max_value = max([x['input_params'][i] for x in data])
#     min_value = min([x['input_params'][i] for x in data])
#     for x in data:
#         x['input_params'][i] = normalize_param(x['input_params'][i], min_value, max_value)

np_data = np.vstack([x['input_params'] for x in data])
print(np_data)

clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, gen_min_span_tree=True)
clusterer.fit(np_data)

print(clusterer.labels_)
print(clusterer.probabilities_)


clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
plt.show()

clusterer.condensed_tree_.plot()
plt.show()

clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
plt.show()

for i in range(len(clusterer.labels_)):
    data[i]['label'] = clusterer.labels_[i]

print([x['label'] for x in data])

number_of_clusters = clusterer.labels_.max() + 1
if number_of_clusters == 0:
    print('no clusters found')
else:
    clusters = [[] for _ in range(number_of_clusters)]
    for x in data:
        if x['label'] != -1:
            clusters[x['label']].append(x)

    centroids = [get_centroid(cluster) for cluster in clusters]
    distance_with_centroids = [distance(test_case['input_params'], centroid) for centroid in centroids]
    closest_cluster_label = distance_with_centroids.index(min(distance_with_centroids))

    similar_cases = clusters[closest_cluster_label]
    upper_limit = [0] * 4
    lower_limit = [0] * 4
    for i in range(4):
        upper_limit[i] = max([x['output_params'][i] for x in similar_cases])
        lower_limit[i] = min([x['output_params'][i] for x in similar_cases])
    print([x['id'] for x in similar_cases])
    print(upper_limit)
    print(lower_limit)
