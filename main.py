from sklearn import tree
from sklearn.metrics import roc_curve, auc
import pylab as pl
import csv
import sys


class TestData(object):
    """Anemic class to hold test data"""
    def __init__(self, userid, age_range, gender, merchantid, label='0'):
        super(TestData, self).__init__()
        self.userid = int(userid)
        if age_range == '':
            self.age_range = 2
        else:
            self.age_range = int(age_range)
        if gender == '':
            self.gender = 0
        else:
            self.gender = int(gender)

        self.merchantid = int(merchantid)
        self.label = int(label)

    def toList(self):
        return [self.age_range, self.gender, self.merchantid]


#Max is about 7M so this should be enough to cover all records
limit = 10000000
def build_training_dataset(filename):
    returnData = []
    csv.field_size_limit(sys.maxsize)

    with open(filename, 'rb') as f:
        print "Building csv"
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if (index % 1000000) == 0:
                print "Processed {0} records".format(index)
            if index == 0 or row[4] == '-1':
                continue
            if index >= limit:
                break
            returnData.append(TestData(userid=row[0], age_range=row[1], gender=row[2], merchantid=row[3], label=row[4]))
        print "CSV built"
        return returnData



def buildTestData(testData):
    returnData = []
    print "Building test data"
    csv.field_size_limit(sys.maxsize)
    with open(testData, 'rb') as userMerchantF:
        reader = csv.reader(userMerchantF)
        for index, row in enumerate(reader):
            if index % 1000000 == 0:
                print "Processed {0} records".format(index)
            if index == 0 or row[4] == '-1':
                continue
            if index >= limit:
                break
            returnData.append(TestData(userid=row[0], age_range=row[1], gender=row[2], merchantid=row[3],label=row[4]))
    print "Test data built"
    return returnData

def test_agains_training_set(clf, training_dataset):
    totalError = 0.0
    cont = 0
    correct = 0
    for index, item in enumerate(training_dataset):
        proba = clf.predict_proba([[item.age_range, item.gender, item.merchantid]])[0][1]
        classification = clf.predict([[item.age_range, item.gender, item.merchantid]])[0]
        error = 0.0

        if item.label == 1:
            error += 1 - proba
            if classification == 1:
                correct += 1
        else:
            error += proba
            if classification == 0:
                correct += 1
        cont += 1
        totalError += error
    print "Error total: {0}%".format(totalError * 100 / cont)
    print "Correctos: {0}. Incorrectos: {1}".format(correct, cont - correct)
    print "Porcentaje correctos: {0}%".format(correct * 100 / cont)

def compute_roc_curve(clf):
    testDataset = buildTestData('processedtrain_format2.csv')
    y_true = []
    y_score = []
    for testData in testDataset:
        classification = testData.label
        proba = clf.predict_proba([[testData.age_range, testData.gender, testData.merchantid]])[0][1]
        y_true.append(classification)
        y_score.append(proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([-0.01, 1.01])
    pl.ylim([-0.01, 1.01])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()

def main():
    training_dataset = build_training_dataset("processedtrain_format2.csv")
    print "Number of records: {0}".format(len(training_dataset))
    ins = []
    outs = []
    for item in training_dataset:
        ins.append([item.age_range, item.gender, item.merchantid])
        outs.append(item.label)
    clf = tree.DecisionTreeClassifier(max_depth=32, min_samples_leaf=1)
    clf = clf.fit(ins, outs)

    print "Testing tree with train data"
    test_agains_training_set(clf, training_dataset)

    print "Computing ROC curve"
    compute_roc_curve(clf)




if __name__ == "__main__":
    main()
