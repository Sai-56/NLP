import pickle
import numpy as np
import operator
from itertools import islice
from itertools import tee

import time


class Embedding:

    def __init__(self, vocab):
        self.vocab = vocab
        self.dictn = {}

    def getCos(self, word1, word2):
        targetWord1 = []
        targetWord2 = []

        for item in self.dictn.items():
            if item[0][0] == word1:
                targetWord1.append(item)
            if item[0][0] == word2:
                targetWord2.append(item)

        commonWords1 = []
        commonWords2 = []

        for item in targetWord1:
            target = item[0][1]
            for item2 in targetWord2:
                if (item2[0][1] == target):
                    commonWords1.append(item)
        for item in targetWord2:
            target = item[0][1]
            for item2 in targetWord1:
                if (item2[0][1] == target):
                    commonWords2.append(item)

        cw1 = {}
        cw = []
        for i in commonWords1:
            cw1[i[0][1]] = i[1]
            cw.append(i[0][1])
        cw2 = {}
        for i in commonWords2:
            cw2[i[0][1]] = i[1]

        num = 0
        denw1 = 0
        denw2 = 0

        for i in range(len(cw1)):
            num += cw1[cw[i]] * cw2[cw[i]]
            denw1 += cw1[cw[i]] ** 2
            denw2 += cw2[cw[i]] ** 2
        try:
            return num / ((denw1 ** .5) * (denw2 ** .5))
        except:
            return 0


class TermTermEmbedding(Embedding):

    def __init__(self, vocab, window, iterable):
        super().__init__(vocab)
        context = []
        q = 0
        while q < window * 2 + 1:
            w = next(iterable)
            context.append(w)
            q = q + 1

        for target in iterable:
            for j in range(len(context)):
                if context[window] in self.vocab and context[j] in self.vocab and j != window:
                    if (context[window], context[j]) in self.dictn:
                        self.dictn[(context[window], context[j])] += 1
                    else:
                        self.dictn[(context[window], context[j])] = 1
            context.pop(0)
            context.append(target)

        for j in range(len(context)):
            if context[window] in self.vocab and context[j] in self.vocab and j != window:
                if (context[window], context[j]) in self.dictn:
                    self.dictn[(context[window], context[j])] += 1
                else:
                    self.dictn[(context[window], context[j])] = 1


def getCos(d, word1, word2):
    targetWord1 = []
    targetWord2 = []

    for item in d.items():
        if item[0][0] == word1:
            targetWord1.append(item)
        if item[0][0] == word2:
            targetWord2.append(item)

    commonContext1 = []   # elements in form of (target, context, frequency)
    commonContext2 = []

    for item in targetWord1:
        target = item[0][1]
        for item2 in targetWord2:
            if item2[0][1] == target:
                commonContext1.append(item)
    for item in targetWord2:
        target = item[0][1]
        for item2 in targetWord1:
            if item2[0][1] == target:
                commonContext2.append(item)

    cw1 = {}  # common word : freq
    cw = []  # list common words

    for i in commonContext1:
        cw1[i[0][1]] = i[1]
        cw.append(i[0][1])
    cw2 = {}
    for i in commonContext2:
        cw2[i[0][1]] = i[1]

    num = 0
    denw1 = 0
    denw2 = 0

    for i in range(len(cw1)):
        num += cw1[cw[i]] * cw2[cw[i]]
        denw1 += cw1[cw[i]] ** 2
        denw2 += cw2[cw[i]] ** 2
    try:
        return num / ((denw1 ** .5) * (denw2 ** .5))
    except:
        return 0


def take(n, iterable):
    return list(islice(iterable, n))


def readFileWGenerator(file):
    file = open(file)
    dat = file.read(1)
    word = ""
    while dat:
        if dat == " ":
            yield word
            word = ""
        else:
            word = word + dat
        dat = file.read(1)
    yield word


def _PCA(X, num_components):
    X_meaned = X - np.mean(X, axis=0)

    cov_mat = np.cov(X_meaned, rowvar=False)

    variance_explained = []
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    for i in eigen_values:
        variance_explained.append((i / sum(eigen_values)) * 100)
    # print("Variance:", np.cumsum(
        # [i * (100 / sum(eigen_values)) for i in eigen_values]))

    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    X_reduced = np.dot(eigenvector_subset.transpose(),
                       X_meaned.transpose()).transpose()

    return X_reduced


def getMatrix(dictn):
    '''Returns a matrix of the frequency and the keys as tuple'''
    keys = np.array(list(dictn.keys()))
    vals = np.array(list(dictn.values()))
    unq_keys, key_idx = np.unique(keys, return_inverse=True)

    key_idx = key_idx.reshape((-1, 2))
    n = len(unq_keys)
    mat = np.zeros((n, n), dtype=vals.dtype)
    mat[key_idx[:, 0], key_idx[:, 1]] = vals

    return (mat, unq_keys)


def getCosMatrix(mat, unq_keys, word1, word2):
    '''Get cos through a numpy matrix'''
    a = mat[np.where(unq_keys == word1)].astype("int64")
    b = mat[np.where(unq_keys == word2)].astype("int64")

    num = 0
    denw1 = 0
    denw2 = 0

    for i in range(len(a)):
        num = num + a[i] * b[i]
        denw1 += a[i] ** 2
        denw2 += b[i] ** 2

    numerator = np.sum(num)
    den = np.sqrt(np.sum(denw1))
    den1 = np.sqrt(np.sum(denw2))

    return numerator / (den * den1)


def getVocab(file, vocabSize):
    """given a text file, returns a tuple of the vocabulary of the most frequent words and a generator"""

    fileReader = readFileWGenerator(file)

    gen1, gen2 = tee(fileReader)
    vocab = {}

    for i in gen1:
        if i in vocab.keys():
            vocab[i] += 1
        else:
            vocab[i] = 1

    vocab = dict(
        sorted(vocab.items(), key=operator.itemgetter(1), reverse=True))
    vocab = dict(take(vocabSize, vocab.items()))

    return vocab, gen2


start_time = time.time()

vocabSize = 1000

# Pickling dictionary
# Comment once pickled to avoid re running every time.
vocab = getVocab("text8", vocabSize)
E = TermTermEmbedding(vocab[0], 4, vocab[1])
with open('filename.pickle', 'wb') as handle:
    pickle.dump(E.dictn, handle, protocol=pickle.HIGHEST_PROTOCOL)
dictionary = E.dictn


# Once The vocabulary has been pickled, this can be used
# with open('filename.pickle', 'rb') as handle:
#     dictionary = pickle.load(handle)


matrix = getMatrix(dictionary)
w1 = "food"
topCos = []
topCos2 = []
pca = _PCA(matrix[0], 1000)

print("Matrix", getCosMatrix(matrix[0], matrix[1], "food", "car"))
print("PCA", getCosMatrix(pca, matrix[1], "food", "car"))
print("Dictn", getCos(dictionary, "food", "car"))

#   before pca
for i in range(vocabSize):
    cos = getCosMatrix(matrix[0], matrix[1], w1, matrix[1][i])
    if i < 10:
        topCos.append((w1, matrix[1][i], cos))
    else:
        for j in range(len(topCos)):
            if cos > topCos[j][2]:
                topCos[j] = (w1, matrix[1][i], cos)
                break

print("Matrix\n", topCos)


# after pca
topCos2 = []
for i in range(vocabSize):
    cos = getCosMatrix(pca, matrix[1], w1, matrix[1][i])
    if i < 10:
        topCos2.append((w1, matrix[1][i], cos))
    else:
        for j in range(len(topCos2)):
            if cos > topCos2[j][2]:
                topCos2[j] = (w1, matrix[1][i], cos)
                break
print("PCA\n",      topCos2)


# Using regular getCos, This is very slow and
# takes very long to compile (~ 150 to 350 secs for vocab of 1000 words)
'''
topCos3 = []
for i in range(vocabSize):
    cos = getCos(dictionary, w1, matrix[1][i])
    if i < 10:
        topCos3.append((w1, matrix[1][i], cos))
    else:
        for j in range(len(topCos3)):
            if cos > topCos3[j][2]:
                topCos3[j] = (w1, matrix[1][i], cos)
                break
print("Dictn\n",      topCos3)
'''

originalWords = []
PCAWords = []
DictnWords = []
# print(f'{"Matrix":<15} {"PCA":<15} {"Dictn":<15}')
print(f'{"Matrix":<15} {"PCA":<15}')


for i in range(len(topCos)):

    originalWords.append(topCos[i][1])
    PCAWords.append(topCos2[i][1])
    # DictnWords.append(topCos3[i][1])

    # print(f'{topCos[i][1]:<15} {topCos2[i][1]:<15} {topCos3[i][1]:<15}')
    print(f'{topCos[i][1]:<15} {topCos2[i][1]:<15}')

common = []
common2 = []

for i in range(len(originalWords)):
    if PCAWords[i] in originalWords:
        common.append(PCAWords[i])
    # if PCAWords[i] in DictnWords:
    #     common2.append(PCAWords[i])

print("Words in PCA list also in Original Matrix:", common)
# print("Words in PCA list also in Original Dictn:", common2)


print("--- %s seconds ---" % (time.time() - start_time))
