from matplotlib import pyplot as plt, font_manager
import matplotlib as mpl

en_font = font_manager.FontProperties(fname='times.ttf')
zh_font = font_manager.FontProperties(fname='仿宋_GB2312.ttf')


def main(Archive_params, Archive_fitness, min_fitness, show_champion_number=3, open_plot=False, now=None):
    plt.rcParams['font.family'] = 'Times New Roman'

    if open_plot:
        draw_iteration(min_fitness, now)
        draw_range(Archive_params, show_champion_number, now)
        draw_3d(Archive_fitness, show_champion_number, now)
        draw_2d(Archive_fitness, show_champion_number, now)


def draw_iteration(min_fitness, now):
    max_iter = len(min_fitness) - 1
    for j in range(3):
        fig, ax = plt.subplots()
        plt.plot(range(-1, max_iter), min_fitness[:, j], c=mpl.cm.viridis([0, 0.4, 0.7][j]))
        plt.xlabel('迭代次数', fontproperties=zh_font, fontsize=18, labelpad=10)
        ax.set_ylabel(['$\it{E}$/kWh', '$\mathrm{\it{T}}$/s', '$\mathrm{\it{Q}}$/mm'][j], fontproperties=en_font,fontsize=18, labelpad=30, rotation=0)
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
        plt.savefig(f'iteration_{j + 1}.png', bbox_inches='tight')
        plt.show()


def draw_range(Archive_params, show_champion_number, now):
    n0 = show_champion_number
    n = len(Archive_params)
    for i in range(4):
        plt.scatter(range(n0), Archive_params[:n0, i], c='r', marker='*')
        plt.scatter(range(n0, n), Archive_params[n0:, i], c='k', marker='.')
        plt.xlabel('index')
        plt.ylabel(['z0', 'da0/mm', 'n0/(r/min)', 'f/(mm/min)'][i])
        if now is not None:
            plt.savefig(f'output/exp_{now}/range_{i + 1}.png', bbox_inches='tight')
        plt.show()


def draw_3d(Archive_fitness, show_champion_number, now):
    n0 = show_champion_number
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Archive_fitness[:n0, 0], Archive_fitness[:n0, 1], Archive_fitness[:n0, 2], c='r', marker='*')
    ax.scatter(Archive_fitness[n0:, 0], Archive_fitness[n0:, 1], Archive_fitness[n0:, 2], c='k', marker='.')
    ax.set_xlabel('energy/kWh')
    ax.set_ylabel('times/s')
    ax.set_zlabel('quality/mm')
    if now is not None:
        plt.savefig(f'output/exp_{now}/3d.png', bbox_inches='tight')
    plt.show()


def draw_2d(Archive_fitness, show_champion_number, now):
    n0 = show_champion_number
    plt.scatter(Archive_fitness[:n0, 0], Archive_fitness[:n0, 1], c='r', marker='*')
    plt.scatter(Archive_fitness[n0:, 0], Archive_fitness[n0:, 1], c='k', marker='.')
    plt.xlabel('energy/kWh')
    plt.ylabel('quality/mm')
    if now is not None:
        plt.savefig(f'output/exp_{now}/2d01.png', bbox_inches='tight')
    plt.show()
    plt.scatter(Archive_fitness[:n0, 0], Archive_fitness[:n0, 2], c='r', marker='*')
    plt.scatter(Archive_fitness[n0:, 0], Archive_fitness[n0:, 2], c='k', marker='.')
    plt.xlabel('energy/kWh')
    plt.ylabel('times/s')
    if now is not None:
        plt.savefig(f'output/exp_{now}/2d02.png', bbox_inches='tight')
    plt.show()
    plt.scatter(Archive_fitness[:n0, 1], Archive_fitness[:n0, 2], c='r', marker='*')
    plt.scatter(Archive_fitness[n0:, 1], Archive_fitness[n0:, 2], c='k', marker='.')
    plt.xlabel('times/s')
    plt.ylabel('quality/mm')
    if now is not None:
        plt.savefig(f'output/exp_{now}/2d03.png', bbox_inches='tight')
    plt.show()
