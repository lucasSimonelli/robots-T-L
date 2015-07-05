import csv
import sys
import random

csv.field_size_limit(sys.maxsize)
my_list = ['test'] * 10 + ['train'] * 90


def preprocessFile(file):
    trainFile = open("train_" + file, 'w')
    testFile = open("test_" + file, 'w')

    writer = csv.writer(trainFile)
    writer2 = csv.writer(testFile)

    with open(file, 'rb') as csvFile:
        reader = csv.reader(csvFile)
        for index, row in enumerate(reader):
            if index % 1000000 == 0:
                print "Processed {0} records".format(index)
            if index == 0 or row[4] == '-1':
                continue

            choice = random.choice(my_list)
            if choice == 'test':
                writer2.writerow(row)
            else:
                writer.writerow(row)
    trainFile.close()
    testFile.close()


preprocessFile("train_format2.csv")
