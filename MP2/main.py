import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions

class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

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
        return np.where(self.net_input(X) >= 0.0, 1, -1)

class Classifier:
    def __init__(self, ppnArray):
        self.ppnArray = ppnArray

    def predict(self, x):
        quantityPerceptrons = len(self.ppnArray)
        for i in range(quantityPerceptrons):
          prediction = np.where(self.ppnArray[i].predict(x) == 1, i, i+1) # Is it a good idea to use i + 1?
          if self.ppnArray[i].predict(x) == 1 :
            return i

        return quantityPerceptrons

'''
class DataSetManager(object):
    
    def preparingTarget(self, targets, minValueInTargets, maxValueInTargets):
        if(maxValueInTargets-minValueInTargets == 0):
            targets.
            return  targets
        for i in range(maxValueInTargets-minValueInTargets)
        return
    '''

class PerceptronManager(object):
    data_train_01_subset = None
    target_train_01_subset = None
    def getData_train_01_subset(self):
        return self.data_train_01_subset

    def getTarget_train_01_subset(self):
        return self.target_train_01_subset

    def createPerceptrons(self, quantity):
        perceptrons = []
        for i in range(quantity):
            perceptrons.append(Perceptron(eta=0.2, n_iter=200))
        return  perceptrons.copy()

    def teachPerceptrons(self, perceptrons, iris_data, iris_target):
        data_train, data_test, target_train, target_test = train_test_split(iris_data, iris_target, test_size=0.3,
                                                                            random_state=1, stratify=iris_target)
        data_train_01_subset = data_train
        target_train_01_subset = target_train
        self.data_train_01_subset = data_train_01_subset
        self.target_train_01_subset = target_train_01_subset
        for perceptronId in range(len(perceptrons)):
            perceptrons[perceptronId] = self.teachPerceptron(perceptrons[perceptronId], perceptronId, data_train_01_subset, target_train_01_subset, target_train)

        return perceptrons.copy()

    def teachPerceptron(self, perceptron,perceptronId, data_train_01_subset, target_train_01_subset, target_train):
        target_train_01_subset[(target_train != perceptronId)] = -1
        target_train_01_subset[(target_train == perceptronId)] = 1
        print(target_train_01_subset)
        perceptron.fit(data_train_01_subset, target_train_01_subset)
        return  perceptron



def showResultsTask1(perceptronManager, perceptron):
    plot_decision_regions(X=perceptronManager.getData_train_01_subset(),
                          y=perceptronManager.getTarget_train_01_subset(), classifier=perceptron)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


def main():
    print("initialization")
    # print(datasets.load_iris())
    iris = datasets.load_iris()
    # iris_data = iris.data # zrobić i dla calego zbioru data.
    iris_data = iris.data[:, [2,3]]
    iris_target = iris.target
    print(iris_data)
    print()
    print(iris_target)
    quantityOfPerceptrons = max(iris_target) + 1
    perceptronManager =  PerceptronManager()

    perceptrons = perceptronManager.createPerceptrons(quantityOfPerceptrons)
    perceptrons = perceptronManager.teachPerceptrons(perceptrons, iris_data, iris_target)

    data_train, data_test, target_train, target_test = train_test_split(iris_data, iris_target, test_size=0.3,
                                                                        random_state=1, stratify=iris_target)
    data_train_01_subset = data_train
    target_train_01_subset = target_train
    target_train_01_subset[(target_train != 2)] = -1
    target_train_01_subset[(target_train == 2)] = 1
    print(target_train_01_subset)
    ppn = Perceptron(eta=0.2, n_iter=200)
    ppn.fit(data_train_01_subset, target_train_01_subset)
    for perc in perceptrons:
        showResultsTask1(perceptronManager, perc)

    '''
    data_train, data_test, target_train, target_test = train_test_split(iris_data, iris_target, test_size=0.3, random_state=1, stratify=iris_target)

    data_train_01_subset = data_train
    target_train_01_subset = target_train
    target_train_01_subset[(target_train != 2)] = -1
    target_train_01_subset[(target_train == 2)] = 1
    print(target_train_01_subset)

   
    ppn.fit(data_train_01_subset, target_train_01_subset)
    '''
    '''
    # X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
    # y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
    # w perceptronie wyjście jest albo 1 albo -1
    '''
    '''
    # plot_decision_regions(X=data_train_01_subset, y=target_train_01_subset, classifier=perceptrons[0])
    plot_decision_regions(X=perceptronManager.getData_train_01_subset(), y=perceptronManager.getTarget_train_01_subset(), classifier=perceptrons[0])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()
'''

if __name__ == '__main__':
    main()
'''

print("initialization")
# print(datasets.load_iris())
iris = datasets.load_iris()
# iris_data = iris.data # zrobić i dla calego zbioru data.
iris_data = iris.data[:, [2,3]]
iris_target = iris.target
print(iris_data)
print()
print(iris_target)
'''