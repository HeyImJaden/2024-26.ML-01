import math

def E1_Triangoli_ask():
    tmp = 0
    lati = []
    for i in range (3):
        while (tmp <= 0):
            try:
                tmp = int(input("Inserisci un lato del triangolo, (numero intero maggiore di Zero): "))
            except:
                print("Errore: tipo non valido")
            lati.append(tmp)
        tmp = -1
    return lati

def E1_Triangoli_classifier(lati):
    tmp = -1

    if ((lati[0] + lati[1] <= lati[2]) or (lati[0] + lati[2] <= lati[1]) or (lati[2] + lati[1] <= lati[0])):
        print("Il triangolo non si può formare")
        return "N/A"
    else:
        if (lati[0] == lati [1] == lati[2]):
            print("Il triangolo è equilatero")
            return "EQUILATERO"
        elif ((lati[0] == lati [1]) or (lati[0] == lati [2]) or (lati[2] == lati [1])):
            print("Il triangolo è isoscele")
            return "ISOSCELE"
        else:
            print("Il triangolo è sacleno")
            return "SCALENO"
       
def E1_Triangoli_perimetro(lati):
        if(E1_Triangoli_isValidTriangolo(lati)):
            perimetro = lati[0] + lati[1] + lati[2]
            return perimetro
        else:
            raise ValueError("Il triangolo non è valido, perimetro non calcolabile")

def E1_Triangoli_area(lati):
        
        perimetro = E1_Triangoli_perimetro(lati)
        
        area = math.sqrt(perimetro/2*(perimetro/2-lati[0])*(perimetro/2-lati[1])*(perimetro/2-lati[2]))
        return round(area, 2)

def E1_Triangoli_isValidTriangolo(lati):
    if ((lati[0] + lati[1] <= lati[2]) or (lati[0] + lati[2] <= lati[1]) or (lati[2] + lati[1] <= lati[0])):
        return False
    else:
        return True
        
def E3_ListaStringa(lista):
    stringa = ''.join(e + "_" for e in lista)[:-1]
    return stringa