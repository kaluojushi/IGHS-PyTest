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
    # now = datetime.now().strftime('%Y%m%d-%H%M%S')
    now = None
    if not os.path.exists('output'):
        os.makedirs('output')
    if now is not None:
        os.makedirs(f'output/exp_{now}')

    od, Hob = ld.main()
    k = 1
    mcs = 2
    ul, ll = ct.main(od, td, min_cluster_size=mcs, min_samples=k, open_plot=False, now=now)
    if ul is not None:
        print('ul: ' + color(ul, 'g'))
        print('ll: ' + color(ll, 'g'))
    else:
        print(color('No cluster found!!!', 'r'))

    end_time = time.time()
    end_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('start: ' + color(start_now, 'c'))
    print('end: ' + color(end_now, 'c'))
    print('time: ' + color('%f s' % (end_time - start_time), 'c'))


if __name__ == '__main__':
    main()
