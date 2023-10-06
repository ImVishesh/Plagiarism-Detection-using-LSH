from flask import Flask, render_template, request

app = Flask(
    __name__, static_url_path="", static_folder="templates", template_folder="templates"
)

# Importing CUSTOM LSH MODULE
from lsh import (
    performLSHcorpus,
    performLSHquery,
    findSimilarDocs,
    getDataForDocumentById,
    getJaccardSimilarity,
)

# PRE PROCESSING: Creating Corpus Bucket
corpusBucket, dictShinglesId = performLSHcorpus()


# Function to fetch result for the user's query.
def get_result(query_inp):
    """
    This Function fetches the result for the user's query by calling the function from LSH modules and return the output to the user.
    """
    global corpusBucket
    global dictShinglesId

    queryBucket = performLSHquery(query_inp, dictShinglesId)
    docIdList = findSimilarDocs(corpusBucket, queryBucket)

    list_of_docs = []
    for item in docIdList:
        list_of_docs.append(getDataForDocumentById(item)[0])
        print(getJaccardSimilarity(query_inp, docIdList))

    return list_of_docs  # list


# disable favicon.ico
@app.route("/favicon.ico")
def favicon():
    return ""


@app.route("/", methods=["GET", "POST"])
def hello_world():
    # make a variable to store the input text
    if request.method == "POST":
        # store the input text in the variable
        input_text = request.form["query"]
        return render_template("index.html", result_op=get_result(input_text))
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
