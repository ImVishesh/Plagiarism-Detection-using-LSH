import os


def removeExtraNewlines():
    location = "dataset"
    docs = os.listdir(location)

    for doc in docs:
        filePtr = open(f"{location}/{doc}", "r", encoding="utf-8")
        filedata = filePtr.read()
        listOfWords = filedata.split("\n")
        print(listOfWords)
        newList = []
        for item in listOfWords:
            if item != "":
                newList.append(item)

        newData = "\n".join(newList)
        filePtr.close()

        filePtr = open(f"{location}/{doc}", "w")
        filePtr.write(newData)

    return


if __name__ == "__main__":
    removeExtraNewlines()
