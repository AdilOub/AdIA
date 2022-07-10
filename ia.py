from audioop import add
from ensurepip import version
from venv import create
import matplotlib.pyplot as plt
from matrice import *
import math

__H__ = 0.0000000001

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
            self.layer.append(createMatriceOfK(couchesIntermediaire, 1, 0))
        self.layer.append(createMatriceOfK(resultats, 1, 0))

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
        self.layer[0] = X
        for i in range(1, self.COUCHE_TOTAL):
            matriceTemp = self.poids[i-1].multiply(self.layer[i-1])
            matriceTemp = matriceTemp.add(self.biais[i])
            self.layer[i] = self.sigmoid(matriceTemp)
        return self.layer[self.COUCHE_TOTAL-1]

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
        if verbose:
            print("On compare {} avec {}".format(X, Y))
        guessed = self.guess(X)
        tempY = Y.clone()
        tempY = tempY.multiplyByReal(-1)
        ecart = guessed.add(tempY)
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

    def registerList(self, liste): #TODO ERREUR ICI !
        self.poids = []
        self.biais = []
        n = 0

        self.biais.append(createMatriceOfK(self.activations, 1, 0))

        self.poids.append(Matrice(self.nbParCoucheInter, self.activations, liste[n:n+self.nbParCoucheInter*self.activations]))
        n+=self.nbParCoucheInter*self.activations
        self.biais.append(Matrice(self.nbParCoucheInter, 1, liste[n:n+self.nbParCoucheInter]))
        n+=self.nbParCoucheInter

        for i in range(0,self.couchesIntermediaire-1):
            self.poids.append(Matrice(self.nbParCoucheInter, self.nbParCoucheInter, liste[n:n+self.nbParCoucheInter*self.nbParCoucheInter]))
            n+=self.nbParCoucheInter*self.nbParCoucheInter 
            self.biais.append(Matrice(self.nbParCoucheInter, 1, liste[n:n+self.nbParCoucheInter]))
            n+=self.nbParCoucheInter
        self.poids.append(Matrice(self.resultats, self.nbParCoucheInter, liste[n:n+self.resultats*self.nbParCoucheInter]))
        n+=self.resultats*self.nbParCoucheInter
        self.biais.append(Matrice(self.resultats, 1, liste[n:n+self.resultats]))
        n+=self.resultats

    def addHInReseau(self, index, h = __H__):
        liste = list(self.reseauToList())
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

    def createInverseGradient(self):
        gradientVal = []
        gradient = Matrice(self.getNbOfPoidsbiais(), 1, gradientVal)
        for i in range(0, self.getNbOfPoidsbiais()):
            derivate = self.calculateDerivate(i)
            gradientVal.append(-derivate) #inverse de la dérivée
        return gradient

    def addTwoList(self, LA, LB):
        if(len(LA) != len(LB)):
            raise Exception("Les deux listes n'ont pas la même taille")
        LR = []
        for i in range(0, len(LA)):
            LR.append(LA[i] + LB[i])
        return LR
        
    def train(self, nbIteration, step, verbose = False):
        for i in range(0, nbIteration):
            reseauToList = list(self.reseauToList())
            gradient = self.createInverseGradient()
            gradient = gradient.multiplyByReal(step)
            self.registerList(self.addTwoList(reseauToList, gradient.valeurs))
            if verbose:
                print("Iteration {}: Cout {}".format(i, self.costTotal()))        
