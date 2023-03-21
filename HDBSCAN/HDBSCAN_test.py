import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
import matplotlib as mpl
import hdbscan

# %matplotlib inline
sns.set_context('talk')
sns.set_style('white')
sns.set_color_codes()
mpl.rcParams['font.family'] = 'Times New Roman'
plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}

moons, _ = data.make_moons(n_samples=70, noise=0.1)
blobs, _ = data.make_blobs(n_samples=75, centers=[(-1.0, 3.0), (0.5, 2.15), (2.0, 3.15)], cluster_std=0.3)
noise1, _ = data.make_blobs(n_samples=1, centers=[(2.0, 1.5)], cluster_std=0.05)
noise2, _ = data.make_blobs(n_samples=1, centers=[(-1.0, 1.5)], cluster_std=0.05)
test_data = np.vstack([moons, blobs, noise1, noise2])
plt.scatter(test_data.T[0], test_data.T[1], c='b', **plot_kwds)
# plt.show()
print(moons)

# clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
# clusterer.fit(test_data)
#
# clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)
# plt.show()
#
# clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
# plt.show()
#
# clusterer.condensed_tree_.plot()
# plt.show()
#
# clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
# plt.show()
#
# palette = sns.color_palette()
# cluster_color = [sns.desaturate(palette[col], sat) if col >= 0 else (0, 0, 0) for col, sat in zip(clusterer.labels_, clusterer.probabilities_)]
# plt.scatter(test_data.T[0], test_data.T[1], c=cluster_color, **plot_kwds)
# plt.show()
#
# print(clusterer.labels_)
# print(test_data)
#
# # 找出标签为-1的数据
# print(test_data[clusterer.labels_ == -1])
