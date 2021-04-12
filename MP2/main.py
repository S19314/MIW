import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions

class Perceptron(object):
    data_train_01_subset = None
    target_train_01_subset = None

    def __init__(self, eta=0.01, n_iter=10, data_train_01_subset= None, target_train_01_subset = None):
        self.eta = eta
        self.n_iter = n_iter
        self.data_train_01_subset = data_train_01_subset
        self.target_train_01_subset = target_train_01_subset

    def getData_train_01_subset(self):
        return self.data_train_01_subset

    def getTarget_train_01_subset(self):
        return self.target_train_01_subset

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        result = np.where(self.net_input(X) >= 0.0, 1, -1)
        '''
        if(len(X) > 2) :
            print("X")
            print(X)
            print("Perceptron predict result ")
            print(result)
        '''
        # return np.where(self.net_input(X) >= 0.0, 1, -1)
        return  result

class Classifier(object):
    ppnArray = []
    def __init__(self, ppnArray):
        self.ppnArray = ppnArray.copy()
    '''
    def predict(self, x):
        predictionsArray = []
        quantityPerceptrons = len(self.ppnArray)

         for objectForDefining in x:
            for i in range(quantityPerceptrons):
              ppn = self.ppnArray[i]
              if (ppn.predict(objectForDefining) == 1 ):
                  predictionsArray.append(i) # quantityPerceptrons # Optional...

        return predictionsArray.copy()
    '''

    def predict(self, x):
        quantityPerceptrons = len(self.ppnArray)
        for i in range(quantityPerceptrons):
          ppn = self.ppnArray[i]
          if (ppn.predict(x) == 1):
              return  i # quantityPerceptrons # Optional...

class PerceptronManager(object):

    def createPerceptrons(self, quantity):
        perceptrons = []
        for i in range(quantity):
            perceptrons.append(Perceptron(eta=0.2, n_iter=200))
        return  perceptrons.copy()

    def teachPerceptrons(self, perceptrons, iris_data, iris_target):
        data_train, data_test, target_train, target_test = train_test_split(iris_data, iris_target, test_size=0.3,
                                                                            random_state=1, stratify=iris_target)
        # data_train_01_subset = data_train
        # target_train_01_subset = target_train

        for perceptronId in range(len(perceptrons)):
            perceptrons[perceptronId].data_train_01_subset = data_train.copy()
            perceptrons[perceptronId].target_train_01_subset = target_train.copy()
            perceptrons[perceptronId] = self.teachPerceptron(perceptrons[perceptronId], perceptronId, perceptrons[perceptronId].getData_train_01_subset(),perceptrons[perceptronId].getTarget_train_01_subset(), target_train)
        return perceptrons.copy()

    def teachPerceptron(self, perceptron,perceptronId, data_train_01_subset, target_train_01_subset, target_train):
        perceptron.target_train_01_subset[(target_train != perceptronId)] = -1
        perceptron.target_train_01_subset[(target_train == perceptronId)] = 1
        # print(perceptron.target_train_01_subset)
        perceptron.fit(perceptron.getData_train_01_subset(), perceptron.getTarget_train_01_subset())
        return  perceptron


def showResultsTask1(perceptron, classificator):
    plot_decision_regions(X=perceptron.getData_train_01_subset(),
                          y=perceptron.getTarget_train_01_subset(), classifier=classificator)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


def punkt1():
    print("initialization punkt 1")
    # print(datasets.load_iris())
    iris = datasets.load_iris()
    # iris_data = iris.data # zrobić i dla calego zbioru data.
    #    iris_data = iris.data[:, [2, 3]]
    iris_data = iris.data
    iris_target = iris.target
    '''
    print(iris_data)
    print()
    print(iris_target)
    '''
    quantityOfPerceptrons = max(iris_target) + 1
    perceptronManager = PerceptronManager()

    perceptrons = perceptronManager.createPerceptrons(quantityOfPerceptrons)
    perceptrons = perceptronManager.teachPerceptrons(perceptrons, iris_data, iris_target)

    classificator = Classifier(perceptrons)

    correctAnswers = 0
    for i in range(len(iris_data)):
        perceptronId = classificator.predict(iris_data[i])
        if (iris_target[i] == perceptronId):
            correctAnswers = correctAnswers + 1

    print("Number of objects: " + str(len(iris_data)))
    print("Correct answers: " + str(correctAnswers))
    print("Accurancy " + str(correctAnswers / len(iris_data)))


# showResultsTask1(perc, classificator)

# DOWN Punkt 2

class LogisticRegressionGD(object):
    data_train_01_subset = None
    target_train_01_subset = None

    def getData_train_01_subset(self):
        return self.data_train_01_subset

    def getTarget_train_01_subset(self):
        return self.target_train_01_subset

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def getPropabilityOfBelongingToClass(self, X):
        net_input = self.net_input(X)
        return self.activation(net_input)

class LogisticRegresionClassifier(object):
    logisticRegresionArray = []
    def __init__(self, logisticRegresionArray):
        self.logisticRegresionArray = logisticRegresionArray.copy()
    '''
    def predict(self, x):
        predictionsArray = []
        quantityPerceptrons = len(self.ppnArray)

         for objectForDefining in x:
            for i in range(quantityPerceptrons):
              ppn = self.ppnArray[i]
              if (ppn.predict(objectForDefining) == 1 ):
                  predictionsArray.append(i) # quantityPerceptrons # Optional...

        return predictionsArray.copy()
    '''

    def predict(self, x):
        quantityRegresions = len(self.logisticRegresionArray)
        for i in range(quantityRegresions):
          regresion = self.logisticRegresionArray[i]
          prediction = regresion.predict(x)
          if ( prediction == 1):
              return  i
        return  -1



class LogisticRegresionManager(object):

    def createLogisticRegresions(self, quantity):
        regresions = []
        for i in range(quantity):
            regresions.append(LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1))
        return  regresions.copy()

    def teachLogisticRegresions(self, regresions, iris_data, iris_target):
        data_train, data_test, target_train, target_test = train_test_split(iris_data, iris_target, test_size=0.3,
                                                                            random_state=1, stratify=iris_target)
        # data_train_01_subset = data_train
        # target_train_01_subset = target_train

        for regresionId in range(len(regresions)):
            regresions[regresionId].data_train_01_subset = data_train.copy()
            regresions[regresionId].target_train_01_subset = target_train.copy()
            regresions[regresionId] = self.teachRegresion(regresions[regresionId], regresionId, regresions[regresionId].getData_train_01_subset(),regresions[regresionId].getTarget_train_01_subset(), target_train)
        return regresions.copy()

    def teachRegresion(self, regresion, regresionId, data_train_01_subset, target_train_01_subset, target_train):
        regresion.target_train_01_subset[(target_train != regresionId)] = 0 # -1
        regresion.target_train_01_subset[(target_train == regresionId)] = 1
        # print(regresion.target_train_01_subset)
        regresion.fit(regresion.getData_train_01_subset(), regresion.getTarget_train_01_subset())
        return  regresion


def punkt2():
    print("initialization punkt 2")
    # print(datasets.load_iris())
    iris = datasets.load_iris()
    # iris_data = iris.data # zrobić i dla calego zbioru data.
    #    iris_data = iris.data[:, [2, 3]]
    iris_data = iris.data
    iris_target = iris.target
    ''' 
    print(iris_data)
    print()
    print(iris_target)
    '''
    quantityOfRegresion = max(iris_target) + 1
    regresionManager = LogisticRegresionManager()

    regresions = regresionManager.createLogisticRegresions(quantityOfRegresion)
    regresions = regresionManager.teachLogisticRegresions(regresions, iris_data, iris_target)

    classificator = LogisticRegresionClassifier(regresions)

    correctAnswers = 0
    for i in range(len(iris_data)):
        regresionId = classificator.predict(iris_data[i])
        if (iris_target[i] == regresionId):
            correctAnswers = correctAnswers + 1

    print("Number of objects: " + str(len(iris_data)))
    print("Correct answers: " + str(correctAnswers))
    print("Accurancy " + str(correctAnswers / len(iris_data)))
    print()
    print()
    # punkt 3
    print("Punkt 3")
    print("Propability Of Belonging To Class ")
    print("For one element", end="")
    print(regresions[0].getPropabilityOfBelongingToClass(iris_data[0]))

    print("For list of elements", end="")
    propabilities = []
    for i in range(24):
        propabilities.append(regresions[0].getPropabilityOfBelongingToClass(iris_data[i]))
    print("Average: ", end="")
    print(np.average(propabilities))
    print("Median: ", end="")
    print(np.median(propabilities))
def main():
     punkt1()
     punkt2()


if __name__ == '__main__':
    main()
