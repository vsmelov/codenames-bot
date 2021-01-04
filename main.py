# 5380

import itertools
from pprint import pprint
import functools

from games import game_6137 as game, RED, BLUE, GREY, blacklist
import gensim.downloader as api


def fuzzy(model, prefix):
    words = model.index2word
    ws = [w for w in words if w.split('_')[0].startswith(prefix.lower())]
    return ws


@functools.lru_cache(maxsize=10_000_000)
def _(model, name, noun=True):
    words = model.index2word
    name = name.lower()
    if noun:
        ws = [w for w in words if w.split('_')[0] == name and (len(w.split('_'))==1 or w.split('_')[1] == 'NOUN')]
    else:
        ws = [w for w in words if w.split('_')[0] == name]
    if len(ws) == 0:
        raise KeyError(name, ws)
    if len(ws) > 1:
        raise KeyError(name, ws)
    key = ws[0]
    return key


@functools.lru_cache(maxsize=10_000_000)
def _ru(name):
    return _(model_ru, name)


def _en(name):
    return _(model_en, name)


def v(model, name: str):
    return model[_(model, name)]


def ru(name):
    return v(model_ru, name)


def en(name):
    return v(model_en, name)



from navec import Navec
path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
# model_ru = Navec.load(path).as_gensim

model_ru = api.load("word2vec-ruscorpora-300")
# model_en = api.load("word2vec-google-news-300")


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

errors = []
for w in game_all:
    try:
        tmp = _ru(w)
    except KeyError:
        ws = fuzzy(model_ru, w)
        errors.append((w, ws))
if errors:
    pprint(errors)
    exit(1)

# vector = model.wv['computer']
# pprint(vector)

# model.similar_by_vector(v('королева')-v('женщина')+v('мужчина'))
# model.similar_by_vector(ru('королева')-ru('женщина')+ru('мужчина'))
# model.similar_by_vector(en('queen')-en('woman')+en('man'))

# pprint(game_all)
# pprint(game_red)
# pprint(game_blue)

# pprint(model_ru.most_similar(positive=['король_NOUN', 'женщина_NOUN'], negative=['мужчина_NOUN'], topn=3))
# pprint(model_ru.most_similar(positive=['король_NOUN', 'женщина_NOUN']*2, negative=['мужчина_NOUN'], topn=3))
# pprint(model_ru.most_similar(positive=['король_NOUN', 'женщина_NOUN'], negative=['мужчина_NOUN']*2, topn=3))
# pprint(model_ru.most_similar(positive=['король_NOUN', 'женщина_NOUN']*2, negative=['мужчина_NOUN']*2, topn=3))

# pprint(fuzzy(model_ru, 'налог'))

TEAM = BLUE
assert TEAM in [BLUE, RED]
OPPONENT = RED if TEAM == BLUE else BLUE

game_my = game_blue if TEAM == BLUE else game_red
game_opponent = game_blue if OPPONENT == BLUE else game_red

positive = game_my
negative = game_grey + game_opponent


def same_root(a, b):
    """ todo: use something smart """
    if a == b:
        return True
    if len(a) > 4 and len(b) > 4:
        if a[:-2] in b:
            # print(f'> same root {a=} {b=}')
            return True
        if b[:-2] in a:
            # print(f'> same root {a=} {b=}')
            return True
    return False


def sim(model, positive=[], negative=[], topn=3):
    result = []
    r = model.most_similar(
        positive=[(_ru(w) if isinstance(w, str) else (_ru(w[0]), w[1])) for w in positive],
        negative=[(_ru(w) if isinstance(w, str) else (_ru(w[0]), w[1])) for w in negative],
        topn=100)

    for row in r:
        if len(result) == topn:
            return result
        if '_' in row[0] and '_NOUN' not in row[0]:
            continue
        if '::' in row[0]:
            continue
        if any(same_root(row[0], w) for w in positive):
            continue
        result.append(row)
    return result


def analyze(model, combination_len, show_top, opponent_weight=0, grey_weight=0):
    print(f'==== analyze({combination_len=}, {show_top=}, {opponent_weight=}, {grey_weight=}) ====')
    sims = []
    neg = []
    if opponent_weight != 0:
        assert opponent_weight < 0
        neg.extend([(w, opponent_weight) for w in game_opponent])
    if grey_weight != 0:
        assert grey_weight < 0
        neg.extend([(w, grey_weight) for w in game_grey])

    for combination in itertools.combinations(positive, r=combination_len):
        if any(w in blacklist for w in combination):
            continue
        r = sim(model, positive=combination, negative=neg, topn=show_top)  # fixme: show_top
        for rr in r:
            sims.append((combination, rr[0], rr[1]))

    for sr in sorted(sims, key=lambda r: -r[2])[:show_top]:
        if sr[1] in blacklist:
            continue

        least = []
        for w in sr[0]:
            s = model.similarity(sr[1], _ru(w))
            least.append((w, s))
        least = sorted(least, key=lambda q: q[1])[:3]

        worst = []
        for w in game_grey:
            s = model.similarity(sr[1], _ru(w))
            worst.append((w, s, 'grey'))
        for w in game_opponent:
            s = model.similarity(sr[1], _ru(w))
            worst.append((w, s, 'opponent'))
        worst = sorted(worst, key=lambda q: -q[1])[:3]

        # tothink: it's not a big deal to let grey
        # skip = False
        # for k in range(3):
        #     al = 1.2 if worst[0][2] == 'grey' else 0.95
        #     if least[0][1] * al < worst[k][1]:
        #         skip = True
        #         break
        # if skip:
        #     continue

        skip = False
        for k in range(len(worst)):
            al = 0.2 if worst[k][2] == 'grey' else -0.0
            if least[0][1] + al < worst[k][1]:
                skip = True
                break
        if skip:
            continue

        print(f'word: {sr[1]} FOR combination: {sr[0]}')
        print(f'combination similarity: {sr[2]}')
        print('   least similar words from combination: ')
        for q in least:
            print(f'    > {q}')
        print('   worst possible mistakes: ')
        for q in worst:
            print(f'    > {q}')


# analyze(model_ru, 3, 10)
# print('='*20)

# analyze(model_ru, 2, 10, grey_weight=0, opponent_weight=0)
# analyze(model_ru, 3, 10, grey_weight=0, opponent_weight=0)
# analyze(model_ru, 4, 10, grey_weight=0, opponent_weight=0)
# analyze(model_ru, 5, 10, grey_weight=0, opponent_weight=0)
#
# analyze(model_ru, 2, 10, grey_weight=-0.1, opponent_weight=-0.2)
# analyze(model_ru, 3, 10, grey_weight=-0.1, opponent_weight=-0.2)
# analyze(model_ru, 4, 10, grey_weight=-0.1, opponent_weight=-0.2)
# analyze(model_ru, 5, 10, grey_weight=-0.1, opponent_weight=-0.2)
#
# analyze(model_ru, 2, 10, grey_weight=-0.2, opponent_weight=-0.4)
# analyze(model_ru, 3, 10, grey_weight=-0.2, opponent_weight=-0.4)
# analyze(model_ru, 4, 10, grey_weight=-0.2, opponent_weight=-0.4)
# analyze(model_ru, 5, 10, grey_weight=-0.2, opponent_weight=-0.4)
#
#
# analyze(model_ru, 2, 10, grey_weight=-0.2, opponent_weight=-0.7)
# analyze(model_ru, 3, 10, grey_weight=-0.2, opponent_weight=-0.7)
# analyze(model_ru, 4, 10, grey_weight=-0.2, opponent_weight=-0.7)
# analyze(model_ru, 5, 10, grey_weight=-0.2, opponent_weight=-0.7)

alpha = 1
beta = 2
show_top = 20
# analyze(model_ru, 2, show_top=show_top, grey_weight=-2*alpha/(alpha*len(game_grey) + beta*len(game_opponent)), opponent_weight=-2*beta/(alpha*len(game_grey) + beta*len(game_opponent)))
analyze(model_ru, 3, show_top=show_top, grey_weight=-3*alpha/(alpha*len(game_grey) + beta*len(game_opponent)), opponent_weight=-3*beta/(alpha*len(game_grey) + beta*len(game_opponent)))
analyze(model_ru, 4, show_top=show_top, grey_weight=-4*alpha/(alpha*len(game_grey) + beta*len(game_opponent)), opponent_weight=-4*beta/(alpha*len(game_grey) + beta*len(game_opponent)))
analyze(model_ru, 5, show_top=show_top, grey_weight=-5*alpha/(alpha*len(game_grey) + beta*len(game_opponent)), opponent_weight=-5*beta/(alpha*len(game_grey) + beta*len(game_opponent)))
analyze(model_ru, 6, show_top=show_top, grey_weight=-5*alpha/(alpha*len(game_grey) + beta*len(game_opponent)), opponent_weight=-5*beta/(alpha*len(game_grey) + beta*len(game_opponent)))


# analyze(model_ru, 4, 30)


# pprint(model_ru.most_similar_cosmul(positive=[_ru(w) for w in positive], topn=10))
# print('='*20)
# pprint(model_ru.most_similar(positive=[_ru(w) for w in positive], negative=[_ru(w) for w in negative], topn=10))
# pprint(model_ru.most_similar_cosmul(positive=[_ru(w) for w in positive], negative=[_ru(w) for w in negative], topn=10))

