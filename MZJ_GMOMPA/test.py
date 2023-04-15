import matplotlib.pyplot as plt
from matplotlib import font_manager

en_font = font_manager.FontProperties(fname='times.ttf')
x = [1, 2, 3, 4, 5]
y = [3, 5, 2, 6, 1]

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_ylabel('$123$/k', fontproperties=en_font, fontsize=18, labelpad=30, rotation=0)
plt.show()

