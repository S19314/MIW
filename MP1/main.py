import numpy as np

class MarkovChain(object):
    gamestates = 0
    humanGameState = 0
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

    def printHumanGameState(self):
        print("HumanGameStates: " + str(self.humanGameState))

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

    def actualizationValueInTransitionMatrix(self, humanStep):
        if (self.previousStep == humanStep):
            for i in range(len(self.columnNames[0])):
                if (self.previousStep == self.columnNames[0][i]):
                    for j in range(len(self.columnNames[i])):
                        if humanStep == self.columnNames[i][j]:
                            self.TransitionMatrix[i][j] += 1
                            self.NumberStepsMatrix[i] += 1
        else:
            if (self.previousStep == "Kamień"):
                if (humanStep == "Nożyce"):
                    self.TransitionMatrix[0][2] = self.TransitionMatrix[0][2] + 1
                    self.NumberStepsMatrix[0] = self.NumberStepsMatrix[0] + 1
                elif (humanStep == "Papier"):
                    self.TransitionMatrix[0][1] = self.TransitionMatrix[0][1] + 1
                    self.NumberStepsMatrix[0] = self.NumberStepsMatrix[0] + 1
            if (self.previousStep == "Nożyce"):
                if (humanStep == "Papier"):
                    self.TransitionMatrix[2][1] = self.TransitionMatrix[2][1] + 1
                    self.NumberStepsMatrix[2] = self.NumberStepsMatrix[2] + 1
                elif (humanStep == "Kamień"):
                    self.TransitionMatrix[2][0] = self.TransitionMatrix[2][0] + 1
                    self.NumberStepsMatrix[2] = self.NumberStepsMatrix[2] + 1
            if (self.previousStep == "Papier"):
                if (humanStep == "Kamień"):
                    self.TransitionMatrix[1][0] = self.TransitionMatrix[1][0] + 1
                    self.NumberStepsMatrix[1] = self.NumberStepsMatrix[1] + 1
                elif (humanStep == "Nożyce"):
                    self.TransitionMatrix[1][2] = self.TransitionMatrix[1][2] + 1
                    self.NumberStepsMatrix[1] = self.NumberStepsMatrix[1] + 1

    def whoWin(self, pythonStep, humanStep):  # whoWin - необходимо переделать/доделать
        print("In WhoWin()")
        print("pythonStep: " + pythonStep)
        print("humanStep: " + humanStep)
        if (pythonStep == humanStep):
            self.gamestates += 0
            self.humanGameState += 0
            print("Remis in WhoWin")
            return "Remis"
        else:
            if (humanStep == "Kamień"):
                if (pythonStep == "Nożyce"):
                    self.gamestates -= 1
                    self.humanGameState += 1
                    return "Win: Human!"
                elif (pythonStep == "Papier"):
                    self.gamestates += 1
                    self.humanGameState -= 1
                    return "Win: Python Programm"
            if (humanStep == "Nożyce"):
                if (pythonStep == "Papier"):
                    self.gamestates -= 1
                    self.humanGameState += 1
                    return "Win: Human!"
                elif (pythonStep == "Kamień"):
                    self.gamestates += 1
                    self.humanGameState -= 1
                    return "Win: Python Programm"
            if (humanStep == "Papier"):
                if (pythonStep == "Kamień"):
                    self.gamestates -= 1
                    self.humanGameState += 1
                    return "Win: Human!"
                elif (pythonStep == "Nożyce"):
                    self.gamestates += 1
                    self.humanGameState -= 1
                    return "Win: Python Programm"

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


    def doStep(self):
        state = ""
        if(self.previousStep.startswith("K")) :
            state = self.columnNames[0][self.getNextStep(0)]
        elif(self.previousStep.startswith("P")) :
            state = self.columnNames[0][self.getNextStep(1)]
        elif (self.previousStep.startswith("N")) :
            state = self.columnNames[0][self.getNextStep(2)]


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
    pythonStep = markovChain.doStep()
    # pythonStep = markovChain.previousStep # markovChain.doStep(humanStep)
    print(markovChain.whoWin(pythonStep, humanStep))
    markovChain.actualizationValueInTransitionMatrix(humanStep)
    markovChain.printGameStates()
    markovChain.printHumanGameState()
    markovChain.printTransitionProbabilityMatrix()
    markovChain.printTransitionMatrix()
    markovChain.previousStep = humanStep
    command = input()
    # print("Before doStep " + markovChain.previousStep)
    # markovChain.previousStep = markovChain.doStep(pythonStep)
    # print("After doStep " + markovChain.previousStep)
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
