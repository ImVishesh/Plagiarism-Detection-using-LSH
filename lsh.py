#!/usr/bin/env python
# coding: utf-8

# # **IMPORT MODULES**
import os
import pickle
import random
import numpy as np


# ## **DEFINING CONSTANTS**

# Location of the Dataset
ASSETS_LOCATION = "dataset"

# SHINGLING
SHINGLE_SIZE = 9  # Size of the shingles

# MIN HASHING
NO_OF_HASH_FUNCTIONS = 100  # Number of min hash functions we are using
HASH_MOD = 100003

# LOCALITY SENSITIVE HASHING (LSH)
CONSTANT_BAND = 20  # Number of Bands
CONSTANT_ROW = 2  # Number of Rows


# # **SHINGLING OPERATION FUNCTIONS**

# ## **GENERAL SHINGLING FUNCTIONS**


# ### **This function returns a set of k-shingles for the string passed as argument.**
def findShingles(docData):
    shingles = []
    for i in range(0, len(docData) - SHINGLE_SIZE + 1):
        shingles.append(docData[i : i + SHINGLE_SIZE])

    return set(shingles)  # set


# ## **SHINGLING OF CORPUS FUNCTIONS**

# ### **This function returns Vocabulary of shingles for entire corpus**


def createShinglesVocab():
    dataFolder = os.listdir(ASSETS_LOCATION)
    dataFolder.sort()

    nameOfDocuments = []
    vocab = set()

    print("[.] Creating Shingles for the corpus!")

    for doc in dataFolder:
        fileName = ASSETS_LOCATION + f"/{doc}"
        nameOfDocuments.append(doc)
        filePtr = open(fileName, "r", encoding="utf-8")
        docData = filePtr.read()
        docVocab = findShingles(docData)
        filePtr.close()
        # print(doc + ": File has been shingled successfully!")
        vocab.update(docVocab)

    print("[+] Shingles Vocabulary created successfully!")

    return nameOfDocuments, vocab  # list, set


# ### **This Function assigns a unique id to each shingle of the Vocabulary**


def assignIdToShingles(vocab):
    dictShinglesId = {}
    id = 0
    for shingle in vocab:
        dictShinglesId[shingle] = id
        id += 1

    return dictShinglesId  # dict


# ### **Creating Shingles Matrix**


def createShinglesMatrix(dictShinglesId):
    shingleMatrix = []

    print("[.] Creating Shingle Matrix...")
    dataFolder = os.listdir(ASSETS_LOCATION)
    dataFolder.sort()
    shingleMatrix = []

    for doc in dataFolder:
        fileName = ASSETS_LOCATION + f"/{doc}"
        filePtr = open(fileName, "r", encoding="utf-8")
        docData = filePtr.read()
        docVocab = findShingles(docData)
        filePtr.close()

        docRow = []

        for item in docVocab:
            docRow.append(dictShinglesId[item])

        docRow.sort()

        shingleMatrix.append(docRow)

    print("[+] Shingle Matrix Created Successfully!")

    return shingleMatrix  # Matrix (where row = number of docs and cols = shingles id)


# ## **SHINGLING OF QUERY**

# ### **Creating Query Matirx**


def createQueryMatrix(userQuery, dictShinglesId):
    # Generating k-Shingles for the userQuery
    setQueryShingles = findShingles(userQuery)  # set

    listQueryShinglesId = []

    for shingle in setQueryShingles:
        if shingle in dictShinglesId.keys():
            listQueryShinglesId.append(dictShinglesId[shingle])

    matrixQueryShinglesId = []
    listQueryShinglesId.sort()
    matrixQueryShinglesId.append(listQueryShinglesId)

    return matrixQueryShinglesId  # matrix


# # **MIN HASH FUNCTIONS**

# ## **Generating Random Hash Functions**

# ### **This function generates {NO_OF_HASH_FUNCTIONS} random hash function**


def generateRandomMinHashFunctions():
    hashFunctions = []

    random.seed(25)

    # Linear Hash Function =>> ax+b
    # coefficient = [a,b]
    for id in range(0, NO_OF_HASH_FUNCTIONS):
        a = random.randint(1, 100000)
        b = random.randint(1, 100000)
        coefficient = [a, b]

        hashFunctions.append(coefficient)

    return hashFunctions  # list of min Hash functions


# # **Min Hash Technique Implementation**

# ## **Generating Signature Matrix**

# ### **Return Matrix of size nxm (n,m are passed as parameters) with all values as INF (very large)**


def intitlizeMatrixWithInfinity(numberOfDocs):
    matrix = []
    for i in range(NO_OF_HASH_FUNCTIONS):
        row = []
        for j in range(numberOfDocs):
            row.append(HASH_MOD)
        matrix.append(row)

    return matrix


# ### **This function generates signature matrix by using min hashing technique using {NO_OF_HASH_FUNCTIONS}**


def generateSignatureMatrix(shingleMatrix):
    # Stores Random Min Hash Fucntions
    hashFunction = generateRandomMinHashFunctions()

    numberOfDocs = len(shingleMatrix)
    # print(numberOfDocs)
    # Initilizing all the rows and cols of signatureMatrix with INFINITY (Very Large Value)
    signatureMatrix = intitlizeMatrixWithInfinity(numberOfDocs)

    # Min Hash Algorithm
    print("[.] Processing Signature Matrix...")
    for i in range(len(shingleMatrix)):
        for j in range(NO_OF_HASH_FUNCTIONS):
            # a and b are the constants of the min hash function
            a = hashFunction[j][0]
            b = hashFunction[j][1]

            for k in shingleMatrix[i]:
                # a * x + b is the hash function where a and b are constants we generated during random min hash function generator while x is a variable which take the value to be hashed.
                hashKey = ((a * (k + 1)) + b) % HASH_MOD
                if hashKey < signatureMatrix[j][i]:
                    signatureMatrix[j][i] = hashKey

    print("[+] Signature Matrix Creating Successfull")

    """
    Signature Matrix is a NO_OF_HASH_FUNCTION x NO_OF_DOCUMENTS sized matrix where each cell stores the hash value.
    """

    return (
        signatureMatrix  # Matrix (rows = number of hash functions & cols = no of docs)
    )


# # **LSH: Locality Sensitive Hashing**

# ### **LSH Implementation Function**


def lsh(signatureMatrix):
    """
    Input: Signature Matrix, number of bands & number of rows.
    We perform LSH on the signature matrix by divinding the signature matrix in b bands where each bans contains r rows.
    """

    print("[.] LSH of Signature Matrix Started...")

    bucketForBands = {}  # bucket (dictinory) that stores sub buckets for all band
    numberOfDocuments = len(signatureMatrix[0])

    for bandB in range(0, CONSTANT_BAND):
        bucketForBandB = {}

        for docNumber in range(numberOfDocuments):
            hashVector = []
            try:
                hashVector = [
                    signatureMatrix[row][docNumber]
                    for row in range(bandB * CONSTANT_ROW, ((bandB + 1) * CONSTANT_ROW))
                ]
            except:
                hashVector = [
                    signatureMatrix[row][docNumber]
                    for row in range(bandB * CONSTANT_ROW, ((bandB) * CONSTANT_ROW))
                ]
                pass  # I passed this statement and didn't wrote anything

            bucketId = "".join(map(str, hashVector))

            if not bucketForBandB.get(bucketId):
                bucketForBandB[bucketId] = set()

            bucketForBandB[bucketId].add(docNumber)
        bucketForBands[bandB] = bucketForBandB

    print("[+] LSH Created Successfully")

    return bucketForBands  # dict


# ### **Function to perform LSH on CORPUS**


def performLSHcorpus():
    """
    This is the function that call the relevant functions to perform LSH on the CORPUS of data. The Steps involved are:

    1. Creating Shingles Vocabulary
    2. Creating Shingles Matrix
    3. Creating Signature Matrix using Min Hashing Technique
    4. Performing LSH on Signature Matrix

    This function return the bukcet formed by the LSH Function on corpus.
    """

    # 1. Creating Shingles Vocabulary
    nameOfDocuments, vocab = createShinglesVocab()

    # 2. Creating Shingles Matrix
    dictShinglesId = assignIdToShingles(vocab)
    shingleMatrix = createShinglesMatrix(dictShinglesId)
    # print(shingleMatrix)

    # 3. Creating Signature Matrix
    signatureMatrix = generateSignatureMatrix(shingleMatrix)
    # print(signatureMatrix)

    # 4. Performing LSH on Signature Matrix
    corpusBucket = lsh(signatureMatrix)
    # print(corpusBucket)

    return corpusBucket, dictShinglesId


# corpusBucket, dictShinglesId = performLSHcorpus()


# ### **Function to perform LSH on Query**


def performLSHquery(query, dictShinglesId):
    """
    This function takes the query from the user and return the bucket formed by the LSH Function on query. The Steps involved are:

    1. Creating Shingles Vocabulary
    2. Creating Shingles Matrix
    3. Creating Signature Matrix using Min Hashing Technique
    4. Performing LSH on Signature Matrix

    This function return the bucket formed by the LSH Function on query.
    """

    # 1. Creating Shingles Vocabulary
    # Already done while performing shingling of corpus
    # Result in vocab set

    # 2. Creating Shingles Matrix
    shingleMatrix = createQueryMatrix(query, dictShinglesId)

    # 3. Creating Signature Matrix
    signatureMatrix = generateSignatureMatrix(shingleMatrix)

    # 4. Performing LSH on Signature Matrix
    queryBucket = lsh(signatureMatrix)

    return queryBucket

# Fining False Positive and False Negatives using a similarity matrix

# This is for finding jaccard similarity of two files.
def jaccard_binary(x,y):
    """A function for finding the similarity between two binary vectors"""
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    similarity = intersection.sum() / float(union.sum())
    return similarity

# This is for finding jaccard similarity of two files.
def jaccard_set(list1, list2):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def getJaccardSimilarity(query, docId):
    documentData = getDataForDocumentById(docId)[1]
    vocabDoc = findShingles(documentData)
    vocabQuery = findShingles(query)
    vocab = vocabDoc.update(vocabQuery)
    list1 = []
    list2 = []
    for i in vocab:
        if i in vocabQuery:
            list1.append(1)
        else:
            list1.append(0)
        
        if i in vocabDoc:
            list2.index(1)
        else:
            list2.append(0)
        
    return jaccard_set(list1, list2)


# # **Functions for Generating OUTPUT for User's Query**

# ### **Retrieving Data of a specific document by using Doc ID**


def getDataForDocumentById(docId):
    dataFolder = os.listdir(ASSETS_LOCATION)
    dataFolder.sort()

    docName = dataFolder[docId]
    docLocation = ASSETS_LOCATION + f"/{docName}"
    filePtr = open(docLocation, "r")
    docData = filePtr.read()

    return docName, docData


# ### **Finding similar document**


def findSimilarDocs(corpusBucket, queryBucket):
    """
    This Function return the set of all the documents that are similar to the query input by the user. This finds and return those documents from the corpus that are present in the same bucket in which the query is also present.
    """

    similarDocs = set()

    for queryBand in queryBucket.keys():
        for queryBucketIndex, queryBucketDocs in queryBucket[queryBand].items():
            if queryBucketDocs:
                if (
                    queryBand in corpusBucket
                    and queryBucketIndex in corpusBucket[queryBand]
                ):
                    similarDocs.update(corpusBucket[queryBand][queryBucketIndex])

    return similarDocs  # list


