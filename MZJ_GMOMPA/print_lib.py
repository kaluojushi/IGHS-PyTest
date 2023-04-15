def color(s, c):
    # bk: black, r: red, g: green, y: yellow, b: blue, p: purple, c: cyan, w: white
    code = {'bk': 30, 'r': 31, 'g': 32, 'y': 33, 'b': 34, 'p': 35, 'c': 36, 'w': 37}
    return f'\033[{code[c]}m{s}\033[0m'
