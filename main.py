from matrice import *
from ia import *

#input liste lde donné return liste de matrice
def helper(listeDeDonne):



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
    costBefore = reseau.costTotal()
    print("cout avant: ", costBefore)
    reseau.train(10000, 0.25)
    print("cout après: ", reseau.costTotal()," avant: ", costBefore)
if __name__ == "__main__":
    main()