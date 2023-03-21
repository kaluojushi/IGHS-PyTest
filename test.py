import numpy as np
from scipy.special import gamma


def test(val):
    _, val = divmod(val, 20)
    return val if val < 10 else 20 - val


print(test(16.87))
print(test(28.46))
print(test(99.98))
