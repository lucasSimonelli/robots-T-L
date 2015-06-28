from pybrain.structure import SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

__author__ = 'tomas'


def build_network():
    return buildNetwork(4, 8, 1, bias=True, hiddenclass=SigmoidLayer)


def build_training_dataset(filename):
    ds = SupervisedDataSet(4, 1)
    with open(filename, 'rb') as f:
        lis = [line.split() for line in f]
        for index, data in enumerate(lis):
            if index == 0:
                continue
            data = data[0].split(',')
            if data[4] == '-1':
                continue
            inp = (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
            out = (int(data[4]),)
            ds.addSample(inp, out)
            print "line{0}: input {1} -> output {2}".format(index, inp, out)
    return ds


def train_network(network, training_dataset):
    trainer = BackpropTrainer(module=network, dataset=training_dataset)
    trainer.trainUntilConvergence()


def main():
    network = build_network()
    training_dataset = build_training_dataset("./datasource.csv")
    train_network(network, training_dataset)

    print network.activate([299904, 3, 0, 1742])


if __name__ == "__main__":
    main()