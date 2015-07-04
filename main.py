from sklearn import tree
import csv
import sys

class TestData(object):
    """Anemic class to hold test data"""
    def __init__(self, userid, age_range, gender, merchantid, label):
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
    def toList():
        return [self.age_range, self.gender, self.merchantid]
 
# Por ahi esto no entra todo en memoria. Ver
def build_training_dataset(filename):
    ds = []
    ds.append([])
    ds.append([])
    numberOfLoyals = 0;
    numberOfOneTimers = 0;
    with open(filename, 'rb') as f:
        print "Building csv"
        reader = csv.reader(f)
        print "CSV built"
        for index, data in enumerate(reader):
            if index == 100000:
                break;
            if data[4] == '-1' or index == 0 or data[4] == '' or data[1] == '' or data[2] == '' or data[3] == '':
                continue
            inp = [int(data[1]), int(data[2]), int(data[3])]
            if (int(data[4])==0):
                #Balance
                ds[1].append(-1)
                numberOfOneTimers = numberOfOneTimers + 1;
                ds[0].append(inp)
            else:
                ds[1].append(1)    
                numberOfLoyals = numberOfLoyals + 1;  
                ds[0].append(inp)
        return ds


# Por ahi esto no entra todo en memoria. Ver
# Devuelve lista de TestData
def buildTestData(testData):
    returnData = []
    print "Building test data"
    csv.field_size_limit(sys.maxsize)
    with open(testData, 'rb') as userMerchantF:
        reader = csv.reader(userMerchantF)
        for index, row in enumerate(reader):
            if index == 0 or row[4] == '':
                continue
            if index == 30000:
                break;
            returnData.append(TestData(userid=row[0], age_range=row[1], gender=row[2], merchantid = row[3], label=row[4]))
    return returnData


def toList(testData):
    return testData.toList()

def main():
    training_dataset = build_training_dataset("./datasource.csv")
    print "Number of records: {0}".format(len(training_dataset[1])+len(training_dataset[0]))
    clf = tree.DecisionTreeClassifier(max_depth=8)
    clf = clf.fit(training_dataset[0],training_dataset[1])

    testData = buildTestData("test_format2.csv")
    totalError = 0.0
    cont = 0
    correct = 0
    print "Testing tree with test data"
    for index, item in enumerate(testData):
        proba = clf.predict_proba([[item.age_range, item.gender, item.merchantid]])[0][1]
        classification = clf.predict([[item.age_range, item.gender, item.merchantid]])[0]
        error = 0.0
        if item.label==1:
            error+=1-proba
            if classification==1:
                correct+=1
        else:
            error+=proba
            if classification==-1:
                correct+=1
        totalError+=error
        cont+=1
    print "Error total: {0}%".format(totalError*100 / cont)
    print "Correctos: {0}. Incorrectos: {1}".format(correct, cont-correct)
    print "Porcentaje correctos: {0}%".format(correct*100 / cont)

if __name__ == "__main__":
    main()