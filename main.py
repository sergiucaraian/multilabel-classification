from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


# Parameters
categories = [
    "earn",
    "acq",
    "money-fx",
    "grain",
    "crude",
    "trade",
    "interest",
    "wheat",
    "ship",
    "corn"
]
trainingEpochs = 20

# Get the documentIDs for all categories
arrTrainingFileNames = list(filter(lambda fileName : fileName.startswith("training/"), reuters.fileids()))

# Get the fileNames for 10 categories
arrTrainingFileNamesCategories = []
arrTestingFileNames = []
for category in categories:
    arrTrainingFileNamesCategories = arrTrainingFileNamesCategories + list(filter(lambda fileName : fileName.startswith("training/"), reuters.fileids(category)))
    arrTestingFileNames = arrTestingFileNames + list(filter(lambda fileName : fileName.startswith("test/"), reuters.fileids(category)))

arrTrainingFilesRaw = [reuters.raw(fileName) for fileName in arrTrainingFileNamesCategories]
arrTestingFilesRaw = [reuters.raw(doc_id) for doc_id in arrTestingFileNames]

# Build one-hot outputs representing apartenence to class
multiLabelBinarizer = MultiLabelBinarizer()
outputTraining = multiLabelBinarizer.fit_transform([
    list(filter(lambda category : category in categories, reuters.categories(fileName))) for fileName in arrTrainingFileNamesCategories
])
outputTesting = multiLabelBinarizer.transform([
    list(filter(lambda category: category in categories, reuters.categories(fileName))) for fileName in arrTestingFileNames
])


def buildMultilabelClassificationPipeline(classifier):
    return Pipeline([
        ("Vectorize inputs into tf-idf form.", TfidfVectorizer(analyzer='word', stop_words='english', lowercase=True)),
        ("Chi^2 Feature selection", SelectPercentile(chi2, percentile=30)),
        ("Classifier", OneVsRestClassifier(classifier))
    ])

def evaluatePipeline(pipeline, pipelineName):
    
    pipeline.fit(arrTrainingFilesRaw, outputTraining)
    result = pipeline.predict(arrTestingFilesRaw)

    print(pipelineName)
    print("Accuracy:    ", accuracy_score(outputTesting, result))
    print("Precission:  ", precision_score(outputTesting, result, average="micro"))
    print("Recall:  ", recall_score(outputTesting, result, average="micro"))
    print("F1:  ", f1_score(outputTesting, result, average="micro"))
    print("")
    


pipelineNaiveBayes = buildMultilabelClassificationPipeline(MultinomialNB())
evaluatePipeline(pipelineNaiveBayes, "Naive Bayes")

pipelineKNeighborsClassifier = buildMultilabelClassificationPipeline(KNeighborsClassifier(10))
evaluatePipeline(pipelineKNeighborsClassifier, "K Neighbors")

pipelineLinearSVC = buildMultilabelClassificationPipeline(LinearSVC())
evaluatePipeline(pipelineLinearSVC, "LinearSVC")

pipelineMultiLayerPerceptron = buildMultilabelClassificationPipeline(MLPClassifier(hidden_layer_sizes=(5,)))
evaluatePipeline(pipelineMultiLayerPerceptron, "Multi-layer perceptron")
