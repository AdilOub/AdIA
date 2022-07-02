@DeprecationWarning
class Matrice:
    def __init__(self, lignes, colones, valeurs = []) -> None:
        self.lignes = lignes
        self.colones = colones
        self.valeurs = valeurs



    def __str__(self) -> str:
        return str(self.valeurs)

    