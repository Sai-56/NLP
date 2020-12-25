import operator
from itertools import islice
from itertools import tee

import time


def getCos(d, word1, word2):
    targetWord1 = []
    targetWord2 = []

    for item in d.items():
        if item[0][0] == word1:
            targetWord1.append(item)
        if item[0][0] == word2:
            targetWord2.append(item)

    commonWords1 = []
    commonWords2 = []

    for item in targetWord1:
        target = item[0][1]
        for item2 in targetWord2:
            if(item2[0][1] == target):
                commonWords1.append(item)
                commonWords2.append(item2)
    num = 0
    denw1 = 0
    denw2 = 0
    for l, n in zip(commonWords1, commonWords2):
        num += l[1]*n[1]
        denw1 += l[1]**2
        denw2 += n[1]**2
    try:
        return num/((denw1**.5)*(denw2**.5))
    except:
        print("0.0")


def take(n, iterable):
    return list(islice(iterable, n))


def getMatrix(arr, window=4):
    # it = iter(arr)
    vocab = {}
    gen1, gen2 = tee(arr)

    for i in gen1:
        if i in vocab.keys():
            vocab[i] += 1
        else:
            vocab[i] = 1

    # it = iter(arr)
    vocab = dict(
        sorted(vocab.items(), key=operator.itemgetter(1), reverse=True))
    vocab = dict(take(1000, vocab.items()))

    dictn = {}
    context = []

    q = 0
    while q < window*2+1:
        w = next(gen2)
        context.append(w)
        q = q+1

    for target in gen2:
        for j in range(len(context)):
            if context[window] in vocab and context[j] in vocab and j != window:
                if (context[window], context[j]) in dictn:
                    dictn[(context[window], context[j])] += 1
                else:
                    dictn[(context[window], context[j])] = 1
        context.pop(0)
        context.append(target)

    for j in range(len(context)):
        if context[window] in vocab and context[j] in vocab and j != window:
            if (context[window], context[j]) in dictn:
                dictn[(context[window], context[j])] += 1
            else:
                dictn[(context[window], context[j])] = 1

    return dictn


def readFileWGenerator(file):
    file = open(file)
    dat = file.read(1)
    word = ""
    while dat:
        if dat == " ":
            yield word
            word = ""
        else:
            word = word+dat
        dat = file.read(1)
    yield word


def readFileWList(file):
    file = open(file)
    arr = file.readlines()
    arr = arr[0].split(" ")
    return arr

# file = open("demo.txt")


# def f():
#     dat = file.read(1)
#     word = ""
#     while dat:
#         if dat == " ":
#             yield word
#             word = ""
#         else:
#             word = word+dat
#         dat = file.read(1)
#     yield word

# f = f()
# for i in f:
#     print(i)

start_time = time.time()
fileData = readFileWGenerator("text8")

dictData = getMatrix(fileData, 4)

print(getCos(dictData, "there", "is"))
print("--- %s seconds ---" % (time.time() - start_time))
