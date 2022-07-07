from venv import create
import matplotlib.pyplot as plt
from matrice import *
import math

class Reseau:
    def __init__(self, activations, couchesIntermediaire, nbParCoucheInter, resultats):
        self.activations = activations
        self.couchesIntermediaire = couchesIntermediaire
        self.nbParCoucheInter = nbParCoucheInter
        self.resultats = resultats

        self.COUCHE_TOTAL = couchesIntermediaire + 2

        # Training data
        self.TrainingX = []
        self.TrainingY = []

        #Guessing data
        self.ValueToGuess = None

        #Listes des matrices
        self.poids = []
        self.biais = []
        self.layer = []

        #Initialisation des matrices couches
        self.layer.append(createMatriceOfK(activations, 1, 0))
        for i in range(1, self.COUCHE_TOTAL-1):
            pass
        #Initialisation des matrices poids et biais de manière aléatoire
        self.poids.append(createRandomMatrice(self.nbParCoucheInter, self.activations))
        self.biais.append(createRandomMatrice(self.activations, 1))
        for i in range(1, self.COUCHE_TOTAL-1):
            self.biais.append(createRandomMatrice(self.nbParCoucheInter, 1))
        for i in range(1, self.COUCHE_TOTAL-2):
            self.poids.append(createRandomMatrice(self.nbParCoucheInter, self.nbParCoucheInter))
        self.poids.append(createRandomMatrice(self.resultats, self.nbParCoucheInter))
        self.biais.append(createRandomMatrice(self.resultats, 1))
    
    def setTrainingData(self, X, Y):
        self.TrainingX = X
        self.TrainingY = Y

    def calculateNextLayer(self, layer, X):
        produit = X
        if(layer < len(self.poids)):
            produit = self.poids[layer].multiply(X)
        somme = produit
        if layer < self.COUCHE_TOTAL-1:
            somme = produit.add(self.biais[layer+1])
        somme = self.sigmoid(somme)
        return somme

    def show(self):
        print("Couche d'activation: " + str(self.activations))
        print("Couches intermédiaires: " + str(self.couchesIntermediaire))
        print("Couches de sortie: " + str(self.resultats))
        print("-----------------------------------------------------")
        self.printPoids()
        print("-----------------------------------------------------")
        self.printBiais()
    
    def printPoids(self):
        print("Poids d'activation: \n" + str(self.poids[0]) + '\n')
        for i in range(1, self.couchesIntermediaire):
            print("Poids de couche intermédiaire {}: \n".format(i) + str(self.poids[i]) + '\n')
        print("Poids de couche de sortie: \n" + str(self.poids[len(self.poids)-1]) )

    def printBiais(self):
        print("Biais d'activation (toujours nuls): \n" + str(self.biais[0]) + '\n')
        for i in range(1, self.couchesIntermediaire+1):
            print("Biais de couche intermédiaire {}: \n".format(i) + str(self.biais[i]) + '\n')
        print("Biais de couche de sortie: \n" + str(self.biais[len(self.biais)-1]))

    def guess(self, X):
        for i in range(0, self.COUCHE_TOTAL):
            X = self.sigmoid(self.calculateNextLayer(i, X))
        return X

    def sigmoid(self, X):
        #f(x)=1/1+e-x
        s = lambda x : 1 / (1+math.exp(-x))
        X.applyFunc(s)
        return X
    
    def cost(self, X, Y, verbose = False):
        guessed = self.guess(X)
        ecart = guessed.add(Y.multiplyByReal(-1))
        if(verbose):
            print("Guessed: ", guessed)
        ecartSquared = ecart.applyFunc(lambda x : x**2)
        return ecartSquared.sum()

def main():
    reseau = Reseau(3, 2, 2, 1)
    reseau.show()
    print("\n")
    matriceA = Matrice(3,1,[0.4,0.32,0.876])
    matriceR = Matrice(1,1, [0.9])
    reseau.setTrainingData([matriceA], [matriceR])
    print("cout:", reseau.cost(matriceA, matriceR, True))

if __name__ == "__main__":
    main()