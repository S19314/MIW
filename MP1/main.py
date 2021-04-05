import numpy as np


class MarkovChain(object):
    gamestates = 0
    columnNames = [["Kamień", "Papier", "Nożyce"], ["Kamień", "Papier", "Nożyce"]]
    TransitionMatrix = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    NumberStepsMatrix = [3, 3, 3]
    previousStep = columnNames[0][0]
    # TransitionProbabilityMatrix = [[0.33,0.33,0.33],[0.33,0.33,0.33],[0.33,0.33,0.33]]
    # SUM()

    def getTransitionProbabilityMatrix(self):
        # TransitionProbabilityMatrix =
            return [[self.TransitionMatrix[0][0] / self.NumberStepsMatrix[0],
                                        self.TransitionMatrix[0][1] / self.NumberStepsMatrix[0],
                                        self.TransitionMatrix[0][2] / self.NumberStepsMatrix[0]],
                                       [self.TransitionMatrix[1][0] / self.NumberStepsMatrix[1],
                                        self.TransitionMatrix[1][1] / self.NumberStepsMatrix[1],
                                        self.TransitionMatrix[1][2] / self.NumberStepsMatrix[1]],
                                       [self.TransitionMatrix[2][0] / self.NumberStepsMatrix[2],
                                        self.TransitionMatrix[2][1] / self.NumberStepsMatrix[2],
                                        self.TransitionMatrix[2][2] / self.NumberStepsMatrix[2]]]

    def printGameStates(self):
        print("GameStates: " + str(self.gamestates))

    def printTransitionProbabilityMatrix(self):
        print("TransitionProbabilityMatrix")
        print("\t\t", end='')
        print(self.columnNames[0])
        transitionProbabilityMatrix = self.getTransitionProbabilityMatrix()
        for i in range(len(self.columnNames[1])):
            print("\t" + self.columnNames[1][i], end='')
            print("     ", end='')
            print(transitionProbabilityMatrix[i])

    def printTransitionMatrix(self):
        print("TransitionMatrix")
        print("\t\t", end='')
        print(self.columnNames[0])
        for i in range(len(self.columnNames[1])):
            print("\t" + self.columnNames[1][i], end='')
            print("     ", end='')
            print(self.TransitionMatrix[i])

    def whoWin(self, pythonStep, humanStep):  # whoWin - необходимо переделать/доделать
        print("In WhoWin()")
        print("pythonStep: " + pythonStep)
        print("humanStep: " + humanStep)
        if (pythonStep == humanStep):
            print("pythonStep == humanStep")
            self.gamestates += 0
            for i in range(len(self.columnNames[0])) :
                if(pythonStep == self.columnNames[0][i]) :
                    self.TransitionMatrix[i][i] = self.TransitionMatrix[i][i] + 1
                    self.NumberStepsMatrix[i] = self.NumberStepsMatrix[i] + 1
                    print("Remis in WhoWin")
                    return "Remis"
        else:
            if (humanStep == "Kamień"):
            # if (pythonStep == "Kamień"):
                if (pythonStep == "Nożyce"):
                    self.TransitionMatrix[0][2] = self.TransitionMatrix[0][2] + 1
                    self.NumberStepsMatrix[0] = self.NumberStepsMatrix[0] + 1
                    self.gamestates -= 1
                    return "Win: Human!"
                    # return "Win: Python Programm"
                elif (pythonStep == "Papier"):
                    self.TransitionMatrix[0][1] = self.TransitionMatrix[0][1] + 1
                    self.NumberStepsMatrix[0] = self.NumberStepsMatrix[0] + 1
                    self.gamestates += 1
                    return "Win: Python Programm"
                    # return "Win: Human!"
            if (humanStep == "Nożyce"):
                if (pythonStep == "Papier"):
                    self.TransitionMatrix[2][1] = self.TransitionMatrix[2][1] + 1
                    self.NumberStepsMatrix[2] = self.NumberStepsMatrix[2] + 1
                    self.gamestates -= 1
                    return "Win: Human!"
                    # return "Win: Python Programm"
                elif (pythonStep == "Kamień"):
                    self.TransitionMatrix[2][0] = self.TransitionMatrix[2][0] + 1
                    self.NumberStepsMatrix[2] = self.NumberStepsMatrix[2] + 1
                    self.gamestates += 1
                    return "Win: Python Programm"
                    # return "Win: Human!"
            if (humanStep == "Papier"):
                if (pythonStep == "Kamień"):
                    self.TransitionMatrix[1][0] = self.TransitionMatrix[1][0] + 1
                    self.NumberStepsMatrix[1] = self.NumberStepsMatrix[1] + 1
                    self.gamestates -= 1
                    return "Win: Human!"
                    # return "Win: Python Programm"
                elif (pythonStep == "Nożyce"):
                    self.TransitionMatrix[1][2] = self.TransitionMatrix[1][2] + 1
                    self.NumberStepsMatrix[1] = self.NumberStepsMatrix[1] + 1
                    #self.gamestates -= 1
                    self.gamestates += 1
                    return "Win: Python Programm"
                    # return "Win: Human!"

    def getNextStep(self, columnId):
        randomFloat = np.random.uniform(0, 1)
        print("randomFloat: " + str(randomFloat))
        indexResult = -1
        previousPropability = 0
        currentPropability = 0
        transitionProbabilityMatrix = self.getTransitionProbabilityMatrix()
        for i in range(len(transitionProbabilityMatrix[columnId])) :
            iterationPropability = transitionProbabilityMatrix[columnId][i]
            currentPropability += iterationPropability
            print("currentPropability: " + str(currentPropability))
            print("previousPropability: " + str(previousPropability))
            if(previousPropability <= randomFloat and randomFloat <= currentPropability ) :
                    indexResult = i
            previousPropability = currentPropability

        print("IndexResult: " + str(indexResult))
        return  indexResult

    def getDominationStep(self, step):
        if (step.startswith("K")):  # == "K") :
            return self.columnNames[0][1]
        elif (step.startswith("P")):  # == "P") :
            return self.columnNames[0][2]
        elif (step.startswith("N")):  # == "N"):
            return self.columnNames[0][0]


    def doStep(self, pythonStep):# humanStep):
        state = ""
        if(humanStep.startswith("K")) : # == "K") :
            state = self.columnNames[0][self.getNextStep(0)]
            # self.TransitionProbabilityMatrix[0][TransitionProbabilityMatrix <= randomFloat]
             # g[any(g==4, g==32)] # something like that
            # state = np.random.choice(self.columnNames, self.TransitionProbabilityMatrix[0])
        elif(humanStep.startswith("P")) : # == "P") :
            state = self.columnNames[0][self.getNextStep(1)]
            # state = self.columnNames[self.getNextStep(self, 1)]
            # state = np.random.choice(self.columnNames, self.TransitionProbabilityMatrix[1])
        elif (humanStep.startswith("N")) : # == "N"):
            state = self.columnNames[0][self.getNextStep(2)]
            # state = self.columnNames[self.getNextStep(self, 2)]
            # state = np.random.choice(self.columnNames, self.TransitionProbabilityMatrix[2])

        print("State before getDomination " + state)
        return  self.getDominationStep(state)
        # return state


markovChain = MarkovChain()

print("Is It rock? " + markovChain.columnNames[0][0])

print('Initialization')
markovChain.printTransitionProbabilityMatrix()
markovChain.printTransitionMatrix()

command = input()
stepNumber  = 0
while(command != "END"):
    stepNumber += 1
    print('StepNumber: ' + str(stepNumber))
    humanStep = command
    pythonStep = markovChain.previousStep # markovChain.doStep(humanStep)
    print(markovChain.whoWin(pythonStep, humanStep))
    markovChain.printGameStates()
    markovChain.printTransitionProbabilityMatrix()
    markovChain.printTransitionMatrix()
    command = input()
    print("Before doStep " + markovChain.previousStep)
    markovChain.previousStep = markovChain.doStep(pythonStep)
    print("After doStep " + markovChain.previousStep)
    '''
    # Just for fun
    # I decided to share it with you.
    
    pop = 0
    while( pop < 5):
        pop += 1
        print(pop)
    else: 
        print('Is else working now?')    
    '''
