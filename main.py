from pybrain.structure import SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
import csv

__author__ = 'tomas'

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
        
def build_network():
    return buildNetwork(4, 8, 1, bias=True, hiddenclass=SigmoidLayer)

# Por ahi esto no entra todo en memoria. Ver
def build_training_dataset(filename):
    ds = SupervisedDataSet(4, 1)
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for index, data in enumerate(reader):
            if data[4] == '-1' or index == 0:
                continue
            inp = (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
            out = (int(data[4]),)
            ds.addSample(inp, out)
            print "line{0}: input {1} -> output {2}".format(index, inp, out)         
        return ds


def train_network(network, training_dataset):
    trainer = BackpropTrainer(module=network, dataset=training_dataset)
    trainer.trainUntilConvergence()

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


def data():
    print buildTestData("./heavyData/test_format1.csv","./heavyData/user_info_format1.csv")


def main():
    network = build_network()
    training_dataset = build_training_dataset("./datasource.csv")
    train_network(network, training_dataset)

    testData = buildTestData("./heavyData/test_format1.csv","./heavyData/user_info_format1.csv")
    for index, item in enumerate(testData):
        if index > 100:
            break;
        print "{0},{1},{2}".format(item.userid, item.merchantid, network.activate((item.userid, item.age_range, item.gender, item.merchantid))[0])


if __name__ == "__main__":
    main()