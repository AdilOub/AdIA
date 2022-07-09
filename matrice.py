import random
from venv import create

'''
Une matrice (i,j) avec i le nombre de ligne et j le nombre de colonne
Soit Index
x = index%colonne
y = index/colonne

Donc:
index congru x modulo j
colonne * y + x  = index 
SSI CA COMMENCE EN (0,0)
'''
class Matrice:
    def __init__(self, lignes, colones, valeurs = []) -> None:
        self.lignes = lignes
        self.colones = colones
        self.valeurs = valeurs

    def multiply(self, B): 
        if not isinstance(B, Matrice):
            raise Exception("Erreur: impossible de multiplier avec un objet non matrice")
        if self.colones != B.getLines():
            raise Exception("Erreur: Les deux matrices ne peuvent pas être multipliées.")
        resultat = self.initList(self.lignes * B.getColones())
        matriceResultat = Matrice(self.lignes, B.getColones(), resultat)
        for h in range(0, self.lignes):
            for i in range(0, B.getColones()):
                sum = 0
                for j in range(0, self.colones):
                    sum += self.valeurs[j+h*self.colones] * B.getValue(i, j)
                matriceResultat.setValue(sum, i, h)           
        return matriceResultat

    def add(self, B):
        if not isinstance(B, Matrice):
            raise Exception("Erreur: impossible d'ajouter avec un objet non matrice")
        if self.colones != B.getColones() or self.lignes != B.getLines():
            raise Exception("Erreur: impossible d'ajouter les deux matrices")
        resultat = self.initList(self.lignes * self.colones)
        matriceResultat = Matrice(self.lignes, self.colones, resultat)
        for i in range(0, self.lignes * self.colones):
            resultat[i] = self.valeurs[i] + B.getValueByIndex(i)
        return matriceResultat
    
    def multiplyByReal(self, k):
        for i in range(0, len(self.valeurs)):
            self.valeurs[i] *= k
        return self

    def addReal(self, k):
        for i in range(0, len(self.valeurs)):
            self.valeurs[i] += k
        return self

    def sum(self):
        sum = 0
        for i in range(0, len(self.valeurs)):
            sum += self.valeurs[i]
        return sum

    def applyFunc(self, fun):
        for i in range(0, len(self.valeurs)):
            self.valeurs[i] = fun(self.valeurs[i])
        return self

    def initList(self, size):
        liste = []
        for i in range(0, size):
            liste.append(0)
        return liste

    def setValue(self, value, x, y):
        self.setValueByIndex(value, y * self.colones + x)

    def setValueByIndex(self, value, index):
        self.valeurs[index] = value

    def getValueByIndex(self, index):
        return self.valeurs[index]
    def getValue(self, x,y):
        return self.getValueByIndex(self.colones * y + x)

    def getCoordonate(self, index):
        return (index%self.colones, index//self.colones)

    def getLines(self):
        return self.lignes
    def getColones(self):
        return self.colones
    def getValList(self):
        return self.valeurs
        
    def __str__(self) -> str:
        return "Lignes: {} Colonnes {} Valeurs: ".format(self.lignes, self.colones) + str(self.valeurs)

def createMatrice(lignes, colones, valeurs = []):
    return Matrice(lignes, colones, valeurs)

def createMatriceOfK(lignes, colones, k):
    values = []
    for i in range(0, lignes*colones):
        values.append(k)
    return createMatrice(lignes, colones, values)

def createRandomMatrice(lignes, colones):
    valeurs = []
    for i in range(0, lignes * colones):
        valeurs.append(random.random())
    return Matrice(lignes, colones, valeurs)