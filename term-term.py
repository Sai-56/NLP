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

    # print(dict(targetWord1))
    # print(dict(targetWord2))
    commonWords1 = []
    commonWords2 = []

    for item in targetWord1:
        context = item[0][1]
        for item2 in targetWord2:
            if(item2[0][1] == context):
                commonWords1.append(item)
                commonWords2.append(item2)
    # print(commonWords1)
    # print(commonWords2)
    num = 0
    denw1 = 0
    denw2 = 0
    for l, n in zip(commonWords1, commonWords2):
        num += l[1]*n[1]
        denw1 += l[1]**2
        denw2 += n[1]**2
    try:
        # print(num, denw1, denw2)
        return num/((denw1**.5)*(denw2**.5))
    except:
        print("zero")


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def dictMethod(arr, window=4):
    dictn = {}
    count = 1
    arrSize = len(arr)
    # for i in range(4, len(arr)-window):
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
        if count % (arrSize//20) == 0:
            numNonZero = len([k for k in dictn.values() if k != 0])
            print(numNonZero, "--")
            # print(arrSize//30, count)
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
    print(len(arr))
    arr = arr.split(" ")
    print(len(arr))
    return arr


start_time = time.time()
fileData = readFile("text8", 100000000)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
dictData = dictMethod(fileData)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print(getCos(dictData, "one", "two"))
print("--- %s seconds ---" % (time.time() - start_time))
