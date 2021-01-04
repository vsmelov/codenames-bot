from pprint import pprint

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import gensim.downloader as api
import pandas as pd

from games import RED, BLUE, GREY, game

from navec import Navec
path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
model = Navec.load(path).as_gensim
# model = api.load("word2vec-ruscorpora-300")


def _(model, name, noun=True):
    words = model.index2word
    if noun:
        ws = [w for w in words if w.split('_')[0] == name.lower() and (len(w.split('_')) == 1 or w.split('_')[1] == 'NOUN')]
    else:
        ws = [w for w in words if w.split('_')[0] == name.lower()]
    if len(ws) == 0:
        raise KeyError(name, ws)
    if len(ws) > 1:
        raise KeyError(name, ws)
    key = ws[0]
    return key



game_all = [(w if isinstance(w, str) else w[0]) for w in game]
game_red = [w[0] for w in game if w[1] == RED]
game_blue = [w[0] for w in game if w[1] == BLUE]
game_grey = [w[0] for w in game if w[1] == GREY]
assert len(game_all) == 25
assert len(game_red) == 9
assert len(game_blue) == 9
assert len(game_grey) == 7
assert not (set(game_red) & set(game_grey))
assert not (set(game_blue) & set(game_grey))
assert not (set(game_red) & set(game_blue))

vocab = [_(model, w) for w in game_all]
X = model[vocab]


#
# tsne = TSNE(n_components=2)
# X_tsne = tsne.fit_transform(X)
#
# df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# ax.scatter(df['x'], df['y'])
#
# for word, pos in df.iterrows():
#     ax.annotate(word.split('_')[0], pos)

tsne = TSNE(n_components=3)
X_tsne = tsne.fit_transform(X)

df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y', 'z'])


def get_color(w):
    w = w.split('_')[0]
    if w in game_grey:
        return 'grey'
    elif w in game_red:
        return 'red'
    elif w in game_blue:
        return 'blue'
    else:
        raise ValueError(w)

df['c'] = [get_color(w) for (w, p) in df.iterrows()]

fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
ax = Axes3D(fig)

ax.scatter(df['x'], df['y'], df['z'], c=df['c'], s=50)

for row in df.iterrows():
    word, (x, y, z, c) = row
    word = word.split('_')[0]
    # ax.annotate(word.split('_')[0], pos)
    ax.text(x, y, z, word)


plt.show()
