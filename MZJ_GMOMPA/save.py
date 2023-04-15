import numpy as np
import pandas as pd
from print_lib import color
from datetime import datetime
import os


def main(Archive_params, Archive_fitness, min_fitness, scores, show_max=-1, save_csv=False, now=None):
    print(color('-' * 15 + ' Save start ' + '-' * 15, 'y'))
    n = len(Archive_params)
    info = np.hstack((Archive_params, Archive_fitness, scores.reshape((n, 1))))
    df = pd.DataFrame(info, columns=['z0', 'da0', 'n0', 'f', 'E', 'T', 'Q', 'score'])
    print(color('Here is the Archive(', 'c') + color(n, 'g') + color(' solutions total):', 'c'))
    if show_max == -1:
        show_max = n
    pd.set_option('display.max_columns', None)
    print(df.head(show_max))
    print('...')
    max_iter = len(min_fitness) - 1
    fitness_info = np.hstack((np.arange(-1, max_iter).reshape((max_iter + 1, 1)), min_fitness))
    fitness_df = pd.DataFrame(fitness_info, columns=['iter', 'E', 'T', 'Q'])
    if save_csv and now is not None:
        df.to_csv(f'output/exp_{now}/Archive_{now}_n={n}.csv', index=True, header=True)
        print('Archive saved to csv successfully!')
        fitness_df.to_csv(f'output/exp_{now}/min_fitness_iter={max_iter}.csv', index=True, header=True)
        print('min_fitness saved to csv successfully!')
    print(color('-' * 15 + ' Save end ' + '-' * 15, 'y'))
