import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from print_lib import color


def main(od, td, min_cluster_size, min_samples, open_plot=False, now=None):
    print(color('-' * 15 + ' HDBSCAN start ' + '-' * 15, 'y'))

    sns.set_context('talk')
    sns.set_style('white')
    sns.set_color_codes()
    mpl.rcParams['font.family'] = 'Times New Roman'

    dtype = np.dtype([('id', np.int32), ('u', np.float64, 7), ('p', np.float64, 4), ('l', np.int32)])
    cases = np.array([(x[0], np.array(x[1:8]), np.array(x[-4:]), -1) for x in od],
                     dtype=dtype)
    test_case = np.array([(-1, np.array(list(td.values())), np.array([0, 0, 0, 0]), -1)], dtype=dtype)
    print(color('%d' % len(cases), 'g') + ' cases loaded successfully!')
    print('test case loaded successfully!')
    max_value = np.max(cases['u'], axis=0)
    min_value = np.min(cases['u'], axis=0)
    cases['u'] = (cases['u'] - min_value) / np.where(max_value != min_value, max_value - min_value, 1)
    test_case['u'] = (test_case['u'] - min_value) / np.where(max_value != min_value, max_value - min_value, 1)
    print('all cases normalized successfully!')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True)
    clusterer.fit(cases['u'])
    print(color('start clustering...', 'c'))

    if open_plot:
        clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        if now is not None:
            plt.savefig(f'output/exp_{now}/ct01.png')
        plt.show()
        print('single linkage tree plot successfully!')
        clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
        if now is not None:
            plt.savefig(f'output/exp_{now}/ct02.png')
        plt.show()
        print('condensed tree plot successfully!')

    cases['l'] = clusterer.labels_
    number_of_clusters = clusterer.labels_.max() + 1
    if number_of_clusters == 0:
        return None, None
    else:
        print(color('%d' % number_of_clusters, 'g') + ' clusters found successfully!')
        clusters = [np.argwhere(cases['l'] == i).flatten().tolist() for i in range(number_of_clusters)]
        print('clusters: ' + color(clusters, 'g'))
        centroids = np.array([get_centroid(cases, cluster) for cluster in clusters])
        distances = np.array([np.linalg.norm(test_case['u'] - centroid) for centroid in centroids])
        closest_cluster_label = np.argmin(distances)
        print('closest cluster is: ' + color('cluster %d' % closest_cluster_label, 'g'))
        closest_cluster = clusters[closest_cluster_label]
        print('length of closest cluster: ' + color('%d' % len(clusters[closest_cluster_label]), 'g'))
        closest_cluster_cases = cases[closest_cluster]
        print('closest cluster cases: ' + color(closest_cluster_cases['id'], 'g'))
        upper_limit = np.max(closest_cluster_cases['p'], axis=0)
        lower_limit = np.min(closest_cluster_cases['p'], axis=0)
        print('upper limit: ' + color(upper_limit, 'g'))
        print('lower limit: ' + color(lower_limit, 'g'))
        print(color('-' * 15 + ' HDBSCAN end ' + '-' * 15, 'y'))
        return upper_limit, lower_limit


def normalize_param(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value) if max_value != min_value else 0.5


def get_centroid(cases, cluster):
    return np.mean(cases['u'][cluster], axis=0)


if __name__ == '__main__':
    pass
