import os
import time
from datetime import datetime

from print_lib import *
from test_data import test_data as td
import load as ld
import cluster as ct
import optimization as opt
import sorting as st
import plot as pt
import save as sv
import objective as obj


def main():
    start_time = time.time()
    start_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    # now = None
    if not os.path.exists('output'):
        os.makedirs('output')
    if now is not None:
        os.makedirs(f'output/exp_{now}')

    od, Hob = ld.main()
    ul, ll = ct.main(od, td, min_cluster_size=3, min_samples=2, open_plot=False, now=now)
    if ul is not None:
        Archive, min_fitness = opt.main(N=100, max_arch=100, max_iter=300, ul=ul, ll=ll, expand_rate=0.95, Hob=Hob,
                           fitness=obj.fitness, constraints=obj.constraints)
        Archive_params = Archive[:, :4]
        Archive_fitness = Archive[:, 4:]
        ranks, scores = st.main(Archive_fitness, E2T=2, E2Q=1/3, T2Q=1/6)
        Archive_sorted_params = Archive_params[ranks]
        Archive_sorted_fitness = Archive_fitness[ranks]
        pt.main(Archive_sorted_params, Archive_sorted_fitness, min_fitness, show_champion_number=5, open_plot=True, now=now)
        sv.main(Archive_sorted_params, Archive_sorted_fitness, min_fitness, scores, show_max=8, save_csv=True, now=now)
    else:
        print(color('No cluster found!!!', 'r'))

    end_time = time.time()
    end_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('start: ' + color(start_now, 'c'))
    print('end: ' + color(end_now, 'c'))
    print('time: ' + color('%f s' % (end_time - start_time), 'c'))


if __name__ == '__main__':
    main()
