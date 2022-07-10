from audioop import add
from venv import create
import matplotlib.pyplot as plt
from matrice import *
import math

__H__ = 10

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
        self.biais.append(createMatriceOfK(self.activations, 1, 0))
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
        self.layer = []
        self.layer.append(X)
        for i in range(0, self.COUCHE_TOTAL):
            X = self.sigmoid(self.calculateNextLayer(i, X))
            self.layer.append(X)
        return X

    def sigmoid(self, X):
        #f(x)=1/1+e-x
        s = lambda x : 1 / (1+math.exp(-x))
        X.applyFunc(s)
        return X
    
    def costTotal(self, verbose=False):
        if self.TrainingX == [] or self.TrainingY == []:
            print("Warning: Aucune donnée d'entrainement")
            return 0
        costTotal = 0
        for i in range(0, len(self.TrainingX)):
            costTotal += self.cost(self.TrainingX[i], self.TrainingY[i], verbose)
        return costTotal/len(self.TrainingX)

    def cost(self, X, Y, verbose = False):
        guessed = self.guess(X)
        ecart = guessed.add(Y.multiplyByReal(-1))
        if(verbose):
            print("Guessed: ", guessed)
        ecartSquared = ecart.applyFunc(lambda x : x**2)
        return ecartSquared.sum()
    
    def reseauToList(self):
        liste = []
        for i in range(0, len(self.poids)):
            for j in range(0, self.poids[i].lignes * self.poids[i].colones):
                liste.append(self.poids[i].valeurs[j])
            if i < len(self.poids):
                for k in range(0, self.biais[i+1].lignes * self.biais[i].colones):
                    liste.append(self.biais[i+1].valeurs[k])
        return liste

    def registerList(self, liste):
        self.poids = []
        self.biais = []
        n = 0

        self.biais.append(createMatriceOfK(self.activations, 1, 0))

        self.poids.append(Matrice(self.nbParCoucheInter, self.activations, liste[n:n+self.nbParCoucheInter*self.activations]))
        n+=self.nbParCoucheInter*self.activations
        self.biais.append(Matrice(self.nbParCoucheInter, 1, liste[n:n+self.nbParCoucheInter]))
        n+=self.nbParCoucheInter

        if(self.nbParCoucheInter > 1):
            for i in range(0,self.nbParCoucheInter-1):
                self.poids.append(Matrice(self.nbParCoucheInter, self.nbParCoucheInter, liste[n:n+self.nbParCoucheInter*self.nbParCoucheInter]))
                n+=self.nbParCoucheInter*self.nbParCoucheInter 
                self.biais.append(Matrice(self.nbParCoucheInter, 1, liste[n:n+self.nbParCoucheInter]))
                n+=self.nbParCoucheInter
        self.poids.append(Matrice(self.resultats, self.nbParCoucheInter, liste[n:n+self.resultats*self.nbParCoucheInter]))
        n+=self.resultats*self.nbParCoucheInter
        self.biais.append(Matrice(self.resultats, 1, liste[n:n+self.resultats]))
        n+=self.resultats

    def addHInReseau(self, index, h = __H__):
        liste = self.reseauToList()
        liste[index] += h
        self.registerList(liste)

    def calculateDerivate(self, index):
        reseauIntoListSave = list(self.reseauToList())
        self.addHInReseau(index)
        costWithH = self.costTotal()
        self.registerList(reseauIntoListSave)
        cost = self.costTotal()
        return (costWithH-cost)/__H__
    
    def getNbOfPoidsbiais(self):
        nbOfPoids = 0
        nbOfBiais = 0
        for poid in self.poids:
            nbOfPoids += poid.getLines() * poid.getColones()
        for i in range(1, len(self.biais)):
            nbOfBiais += self.biais[i].getLines() * self.biais[i].getColones()
        return nbOfPoids + nbOfBiais

    def createGradient(self):
        gradientVal = []
        gradient = Matrice(self.getNbOfPoidsbiais(), 1, gradientVal)
        for i in range(0, self.getNbOfPoidsbiais()):
            derivate = self.calculateDerivate(i)
            gradientVal.append(-derivate) #inverse de la dérivée
        self.show()
        return gradient
        
    def train(self, nbIteration, step):
        for i in range(0, nbIteration):
            matriceParam = Matrice(1, self.getNbOfPoidsbiais(), [])
            matriceParam.valeurs = self.reseauToList()
            gradient = self.createGradient()
            gradient = gradient.multiplyByReal(step)
            matriceParam.multiply(gradient)
            self.registerList(matriceParam.valeurs)
            print("Iteration {}: Cout {}".format(i, self.costTotal()))        

def main():
    print("Hello world!")
    reseau = Reseau(3, 2, 2, 1)
    reseau.show()
    print("\n")
    matriceA = Matrice(3,1,[0.4,0.32,0.876])
    matriceB = Matrice(3,1, [0.7,0.5,0.9,.0])
    matriceR = Matrice(1,1, [0.333])
    matriceR2 = Matrice(1,1, [0.999])
    reseau.setTrainingData([matriceA, matriceB], [matriceR, matriceR2])
    print("cout avant: ", reseau.costTotal(True))
    #reseau.train(1, 0.1)
    print("cout après: ", reseau.costTotal(True))
    grad = reseau.createGradient()
    print("Gradient: ", grad)
if __name__ == "__main__":
    main()