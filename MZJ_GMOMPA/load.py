import numpy as np
import pandas as pd
from print_lib import color


def main():
    print(color('-' * 15 + ' Load start ' + '-' * 15, 'y'))
    data_df = pd.read_csv('data.csv')
    origin_data = data_df.to_numpy()
    print('origin data loaded successfully!')
    Hob_df = pd.read_csv('Hob.csv')
    Hob = Hob_df.to_numpy()
    print('Hob loaded successfully!')
    print(color('-' * 15 + ' Load end ' + '-' * 15, 'y'))
    return origin_data, Hob


if __name__ == '__main__':
    main()
