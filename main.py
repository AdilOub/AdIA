from email.mime import image
from matrice import *
from ia import *

#input liste lde donné return liste de matrice
def helper(listeDeDonne):
    pass

def readLabel(path):
    try:
        listDesResultatsAttend = []
        with open(path, 'rb') as file:
            secretNb = int.from_bytes(file.read(4), "big")
            print("secretNb Label: ", secretNb)
            nbOfLabel = int.from_bytes(file.read(4), "big")
            print("nbOfLabel: ", nbOfLabel)
            byte = b'00'
            
            valeurs = [0,0,0,0,0,0,0,0,0,0]
            n=0
            while byte and n < nbOfLabel:
                byte = file.read(1)
                resultat = int.from_bytes(byte, "big")
                
                valeursTemp = list(valeurs)
                valeursTemp[resultat] = 1
                matriceTemp = Matrice(10, 1, valeursTemp)

                listDesResultatsAttend.append(matriceTemp)
                n+=1
            return listDesResultatsAttend
    except IOError:
        print("Can't open file !")
        raise IOError

def readImage(path):
    try:
        listDesResultatsAttend = []
        with open(path, 'rb') as file:
            secretNb = int.from_bytes(file.read(4), "big")
            print("secretNb Image: ", secretNb)
            nbOfImgs = int.from_bytes(file.read(4), "big")
            print("nbOfimgs: ", nbOfImgs)
            byte = b'00'
            
            rows = int.from_bytes(file.read(4), "big")
            columns = int.from_bytes(file.read(4), "big")
            if( not (rows == columns and rows == 28)):
                raise Exception("Taille differente à 28x28")

            n=0
            valeurs = [0 for i in range(0, 28*28)]
            while byte and n < nbOfImgs:
                valeursTemp = list(valeurs)
                for i in range(0,28*28):
                    byte = file.read(1)
                    resultat = int.from_bytes(byte, "big")
                    valeursTemp[i] = resultat/255 #greyscale 0-255 -> 0-1

                matriceTemp = Matrice(28*28, 1, valeurs)
                listDesResultatsAttend.append(matriceTemp)
                n+=1
            return listDesResultatsAttend
    except IOError:
        print("Can't open file !")
        raise IOError


def doIAStuff(labels, images, test, label):
    print("Hello world (todo find {})!".format(label))
    reseau = Reseau(28*28, 2, 8, 10)
    
    reseau.describe()
    #reseau.show()
    
    print("\n")
    reseau.setTrainingData(images, labels)
    costBefore = reseau.costTotal()
    print("cout avant: ", costBefore)
    reseau.train(100, 0.99, True, 1)
    print("cout après: ", reseau.costTotal()," avant: ", costBefore)
    print("le réseau doit trouver: ", label)
    guessed = reseau.guess(test)
    print("GUESSed:", guessed)
    
    a = input("any key to leave")
    confirm = input("sure ? (Y/N)")
    while confirm != "Y":
        confirm = input("sure ? (Y/N)")
    for i in range(1,11):
        input("vraiment sur ({}/10)".format(i))

def main():
    #doIAStuff()

    listeDesLabels = readLabel("./train/train-labels.idx1-ubyte")
    listeDesImages = readImage("./train/train-images.idx3-ubyte")

    listeTestLabel = readLabel("./data/t10k-labels.idx1-ubyte")
    listeTestImage = readImage("./data/t10k-images.idx3-ubyte")
    

    '''
    #DEBUG
    listeDesLabels = [Matrice(10, 1, [0,0,0,0,0,0,0,0,0,1])]
    listeDesImages = [Matrice(28*28,1, [0 for i in range(28*28)])]

    listeTestImage = [None, Matrice(10, 1, [0,0,0,0,0,0,0,0,0,1])]
    listeTestLabel = [None, Matrice(28*28,1,[0 for i in range(28*28)])]
    '''

    if len(listeDesLabels) != len(listeDesImages):
        raise Exception("Les deux listes n'ont pas la même taille")
    print("Image chargée")
    doIAStuff(listeDesLabels[:200], listeDesImages[:200], listeTestImage[1], listeTestLabel[1])

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