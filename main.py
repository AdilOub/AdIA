import matplotlib.pyplot as plt
import numpy as np
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
        self.X = []
        self.Y = []

        #Guessing data
        self.ValueToGuess = None

        #Listes des matrices
        self.poids = []
        self.biais = []

        self.poids.append(np.random.randn(self.nbParCoucheInter, self.activations)) #L'erreur est ici dans la formes des matrices
        self.biais.append(np.zeros((self.activations, 1)))
        for i in range(1, self.COUCHE_TOTAL-1):
            self.poids.append(np.random.randn(self.nbParCoucheInter, self.nbParCoucheInter))
            self.biais.append(np.random.randn(self.nbParCoucheInter, 1))
        self.poids.append(np.random.randn(self.resultats, self.nbParCoucheInter))
        self.biais.append(np.random.randn(self.resultats, 1))
    
    def setTrainingData(self, X, Y):
        self.X = X
        self.Y = Y

    def calculateNextLayer(self, layer, X):
        produit = np.dot(self.poids[layer], X)
        somme = produit
        if layer < self.COUCHE_TOTAL-1:
            somme = np.add(produit, self.biais[layer+1])
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
        for i in range(1, self.couchesIntermediaire+1):
            print("Poids de couche intermédiaire {}: \n".format(i) + str(self.poids[i]) + '\n')
        print("Poids de couche de sortie: \n" + str(self.poids[len(self.poids)-1]) )

    def printBiais(self):
        print("Biais d'activation (toujours nuls): \n" + str(self.biais[0]) + '\n')
        for i in range(1, self.couchesIntermediaire+1):
            print("Biais de couche intermédiaire {}: \n".format(i) + str(self.biais[i]) + '\n')
        print("Biais de couche de sortie: \n" + str(self.biais[len(self.biais)-1]))


    def subtractMatrice(self, A, B):
        n = 0
        result = []
        for cell in A.flatten():
            result.append(cell - B[n])
        return result

    def guess(self, X):
        for i in range(0, self.COUCHE_TOTAL):
            X = self.sigmoid(self.calculateNextLayer(i, X))
        return X

    def sigmoid(self, X):
        #f(x)=1/1+e-x
        s = lambda x : 1 / (1+math.exp(-x))
        sV = np.vectorize(s)
        return sV(X)
    
    def cost(self, X, Y):
        guessed = self.guess(X)
        print("guessed: " + str(guessed))
        print("Y: " + str(Y))
        ecart = self.subtractMatrice(guessed, Y)
        print("ecart: " + str(ecart))
        squareMatrice = np.vectorize(lambda x : x**2)
        ecartSquared = squareMatrice(ecart)
        return ecartSquared.sum()

def main():
    reseau = Reseau(3, 2, 2, 1)
    reseau.show()
    reseau.setTrainingData([[0.4],[0.32],[0.876]],[[0,3333]])
    print(reseau.cost([[0.4],[0.32],[0.876]],[[0,3333]]))
    matriceA = Matrice(2,2,[1,2,3,4])
    matriceB = Matrice(2, 1, [5,6])
    print(matriceA.multiply(matriceB))

if __name__ == "__main__":
    main()