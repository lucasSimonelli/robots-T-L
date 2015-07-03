from sklearn import tree
import csv

class TestData(object):
    """Anemic class to hold test data"""
    def __init__(self, userid, age_range, gender, merchantid):
        super(TestData, self).__init__()
        self.userid = int(userid)
        if age_range == '':
            self.age_range = 2
        else:
            self.age_range = int(age_range)
        if gender == '':
            self.age_range = 0
        else:
            self.gender = int(gender)

        self.merchantid = int(merchantid)

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
                if (numberOfOneTimers <= numberOfLoyals):
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
def buildTestData(userMerchantFile, userInfoFile):
    userProfiles = {}
    with open(userInfoFile, 'rb') as userInfoF:
        reader = csv.reader(userInfoF)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            userProfiles[row[0]] = row[1:]  
    returnData = []
    with open(userMerchantFile, 'rb') as userMerchantF:
        reader = csv.reader(userMerchantF)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            returnData.append(TestData(row[0], userProfiles[row[0]][0], userProfiles[row[0]][1], row[1]))
    return returnData


def toList(testData):
    return testData.toList()

def main():

    

    training_dataset = build_training_dataset("./datasource.csv")
    print len(training_dataset[1])
    clf = tree.DecisionTreeClassifier(max_depth=10)
    clf = clf.fit(training_dataset[0],training_dataset[1])

    testData = buildTestData("./heavyData/test_format1.csv","./heavyData/user_info_format1.csv")
    for index, item in enumerate(testData):
        if index > 100:
            break;
        print "{0},{1},{2}".format(item.userid, item.merchantid, clf.predict_proba([[item.age_range, item.gender, item.merchantid]]))


if __name__ == "__main__":
    main()