from copyreg import constructor
import matplotlib.pyplot as plt
import numpy as np
from matrice import *


class Reseau:
    def __init__(self, activations, couchesIntermediaire, nbParCoucheInter, resultats):
        self.activations = activations
        self.couchesIntermediaire = couchesIntermediaire
        self.nbParCoucheInter = nbParCoucheInter
        self.resultats = resultats

        self.COUCHE_TOTAL = couchesIntermediaire + 2

        # Training data
        self.X = []
        self.Y = []

        #Guessing data
        self.ValueToGuess = None

        #Listes des matrices
        self.poids = []
        self.biais = []

        self.poids.append(np.random.randn(self.activations, self.nbParCoucheInter))
        self.biais.append(np.random.randn(self.activations, 1))
        for i in range(1, self.COUCHE_TOTAL-1):
            self.poids.append(np.random.randn(self.nbParCoucheInter, self.nbParCoucheInter))
            self.biais.append(np.random.randn(self.nbParCoucheInter, 1))
        self.poids.append(np.random.randn(self.resultats, self.nbParCoucheInter))
        self.biais.append(np.random.randn(self.resultats, 1))
    
    def setTrainingData(self, X, Y):
        self.X = X
        self.Y = Y

    def calculateNextLayer(self, layer, X):
        return np.dot(X, self.poids[layer]) + self.biais[layer]

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
        for i in range(1, self.couchesIntermediaire+1):
            print("Poids de couche intermédiaire {}: \n".format(i) + str(self.poids[i]) + '\n')
        print("Poids de couche de sortie: \n" + str(self.poids[len(self.poids)-1]) )

    def printBiais(self):
        print("Biais d'activation: \n" + str(self.biais[0]) + '\n')
        for i in range(1, self.couchesIntermediaire+1):
            print("Biais de couche intermédiaire {}: \n".format(i) + str(self.biais[i]) + '\n')
        print("Biais de couche de sortie: \n" + str(self.biais[len(self.biais)-1]))

def main():
    reseau = Reseau(3, 2, 2, 1)
    reseau.show()

if __name__ == "__main__":
    main()