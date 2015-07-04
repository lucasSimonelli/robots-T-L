import csv
import sys
csv.field_size_limit(sys.maxsize)

def preprocessFile(file):
    preprocessedFile = open("processed"+file, 'w')
    writer = csv.writer(preprocessedFile)
    with open(file, 'rb') as csvFile:
        reader = csv.reader(csvFile)
        for index, row in enumerate(reader):
            if index % 1000000 == 0:
                print "Processed {0} records".format(index)
            if index == 0 or row[4] == '-1':
                continue
            writer.writerow(row)
    preprocessedFile.close()


preprocessFile("train_format2.csv")


