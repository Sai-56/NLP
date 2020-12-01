import operator
from itertools import islice
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
        context = item[0][1]
        for item2 in targetWord2:
            if(item2[0][1] == context):
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
    vocab = {}
    for i in arr:
        if i in vocab.keys():
            vocab[i] += 1
        else:
            vocab[i] = 1

    vocab = dict(
        sorted(vocab.items(), key=operator.itemgetter(1), reverse=True))
    vocab = dict(take(10000, vocab.items()))

    # print(vocab)

    dictn = {}
    stop = len(arr)-window
    for i in range(window, stop):
        for j in range(i-window, i):
            if arr[i] in vocab and arr[j] in vocab:
                if (arr[i], arr[j]) in dictn:
                    dictn[(arr[i], arr[j])] += 1
                else:
                    dictn[(arr[i], arr[j])] = 1
        for j in range(i+1, i+window+1):
            if arr[i] in vocab and arr[j] in vocab:
                if (arr[i], arr[j]) in dictn:
                    dictn[(arr[i], arr[j])] += 1
                else:
                    dictn[(arr[i], arr[j])] = 1
        

    # numNonZero = len([k for k in dictn.values() if k != 1])

    return dictn


def readFile(file, numChars):
    file = open(file)
    arr = file.read(numChars)
    arr = arr.split(" ")
    return arr

start_time = time.time()
fileData = readFile("text8", 100000000)
dictData = getMatrix(fileData)
print(getCos(dictData, "there", "is"))
print("--- %s seconds ---" % (time.time() - start_time))
