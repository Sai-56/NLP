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
    dictn = {}
    count = 1
    arrSize = len(arr)
    i = window
    stop = len(arr)-window
    while i < stop:
        j = i-window
        while j < i:
            if (arr[i], arr[j]) in dictn:
                dictn[(arr[i], arr[j])] += 1
            else:
                dictn[(arr[i], arr[j])] = 0
            j = j+1
        j = i+1
        while j < i+window+1:
            if (arr[i], arr[j]) in dictn:
                dictn[(arr[i], arr[j])] += 1
            else:
                dictn[(arr[i], arr[j])] = 0
            j = j+1
        if count % (arrSize//15) == 0:
            numNonZero = len([k for k in dictn.values() if k != 0])
            if numNonZero > 80000:
                dictn = dict(
                    sorted(dictn.items(), key=operator.itemgetter(1), reverse=True))
                dictn = dict(take(50000, dictn.items()))
        count = count + 1
        i = i+1
    numNonZero = len([k for k in dictn.values() if k != 0])
    dictn = dict(
        sorted(dictn.items(), key=operator.itemgetter(1), reverse=True))
    dictn = dict(take(50000, dictn.items()))
    return dictn


def readFile(file, numChars):
    file = open(file)
    arr = file.read(numChars)
    arr = arr.split(" ")
    return arr


start_time = time.time()
fileData = readFile("text8", 100000000)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
dictData = getMatrix(fileData)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print(getCos(dictData, "car", "food"))
print(getCos(dictData, "revolution", "positive"))
print(getCos(dictData, "revolution", "negative"))
print("--- %s seconds ---" % (time.time() - start_time))
