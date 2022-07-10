from matrice import *
from ia import *

#input liste lde donné return liste de matrice
def helper(listeDeDonne):
    pass

def readLabel():
    try:
        listDesResultatsAttend = []
        with open("./train/train-labels.idx1-ubyte", 'rb') as file:
            secretNb = int.from_bytes(file.read(4), "big")
            print("secretNb: ", secretNb)
            nbOfLabel = int.from_bytes(file.read(4), "big")
            print("nbOfLabel: ", nbOfLabel)
            byte = b'00'
            
            valeurs = [0,0,0,0,0,0,0,0,0,0]
            n=0
            while byte and n < nbOfLabel:
                byte = file.read(1)
                resultat = int.from_bytes(byte, "big")
                
                valeursTemp = list(valeurs)
                valeurs[resultat] = 1
                matriceTemp = Matrice(10, 1, valeurs)

                listDesResultatsAttend.append(matriceTemp)
                n+=1
            return listDesResultatsAttend
    except IOError:
        print("Can't open file !")
        raise IOError

def readImage():
    try:
        listDesResultatsAttend = []
        with open("./train/train-images.idx3-ubyte", 'rb') as file:
            secretNb = int.from_bytes(file.read(4), "big")
            print("secretNb: ", secretNb)
            nbOfLabel = int.from_bytes(file.read(4), "big")
            print("nbOfLabel: ", nbOfLabel)
            byte = b'00'
            
            valeurs = [0,0,0,0,0,0,0,0,0,0]
            n=0
            while byte and n < nbOfLabel:
                byte = file.read(1)
                resultat = int.from_bytes(byte, "big")
                
                valeursTemp = list(valeurs)
                valeurs[resultat] = 1
                matriceTemp = Matrice(10, 1, valeurs)

                listDesResultatsAttend.append(matriceTemp)
                n+=1
            return listDesResultatsAttend
    except IOError:
        print("Can't open file !")
        raise IOError


def doIAStuff():
    print("Hello world!")
    reseau = Reseau(3, 2, 8, 2)
    reseau.show()
    print("\n")
    matriceA = Matrice(3,1,[0.4,0.32,0.876])
    matriceB = Matrice(3,1, [0.7,0.5,0.9,.0])
    matriceC = Matrice(3,1, [0.125,0.524,0.45,0.666])
    matriceD = Matrice(3,1, [0.851,0.0,0.0,0.0])
    matriceR = Matrice(2,1, [0.333, 0.999])
    matriceR2 = Matrice(2,1, [0.999, 0.333])
    matriceR3 = Matrice(2,1, [0.005, 0.5555])
    matriceR4 = Matrice(2,1, [0.999, 0.999])
    reseau.setTrainingData([matriceA, matriceB, matriceC, matriceD], [matriceR, matriceR2, matriceR3, matriceR4])
    costBefore = reseau.costTotal()
    print("cout avant: ", costBefore)
    reseau.train(5000, 0.99, True, 100)
    print("cout après: ", reseau.costTotal()," avant: ", costBefore)

def main():
    #doIAStuff()
    listeDesLabels = readLabel()

if __name__ == "__main__":
    main()


'''
step 0.1
3 2 8 2 cout après:  0.26916384611598976  avant:  0.5059493002041443
3 2 10 2 cout après:  0.26882216929571084  avant:  0.49941052818761555
3 2 16 2 cout après:  0.5192270542151065  avant:  0.5192321625968184
step 0.05 5000
3 2 8 2 cout après:  0.26888088134408594  avant:  0.4909566342513761
step 0.9 5000 cout après:  0.00043139097531157795  avant:  0.49720491734798766
3 2 8 2
'''