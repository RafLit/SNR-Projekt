# SNR-Projekt
Głęboki klasyfikator neuronowy etykietujący dane na podstawie określonych reguł.
## Struktura
    .
    ├── docs                    # Dokumentacja
    ├── script                  # Skrypty w pythonie
    │   ├──generateTrainTest.py # Generuje zbiór treningowy i testowy ze zbiorów ogólnych 
    │   ├──createModel.py       # Zawiera funkcje pozwalajace stworzyc uzywane przez nas modele
    │   ├──trainTest.py             # Skrypt trenujacy model i testujacy na zbiorze testowym 
    │   ├──classifyUnlabeled.py # Skrypt trenujacy model i klasyfikujacy dane nieoznaczone
    ├── data                    # Zbiór danych
    │   ├──train                # Dane trenujace (generowane przez skrypt)
    │   │  ├──accept            # Dane trenujace - zaakceptowane deski
    │   │  ├──reject            # Dane trenujace - odrzucone deski
    │   ├──test                 # Dane testujace (generowane przez skrypt)
    │   │  ├──accept            # Dane testujace - zaakceptowane deski
    │   │  ├──reject            # Dane testujace - zaakceptowane deski
    │   ├──unlabeled            # Dane nieoznaczone
    │   │  ├──unlabeled         # Dane nieoznaczone do oznaczenia przez model
    │   ├──result               # Wyniki klasyfikacji danych nieoznaczonych
    │   │  ├──accept            # Dane sklasyfikowane jako zaakceptowane
    │   │  ├──reject            # Dane sklasyfikowane jako odrzucone
    │   ├──accept               # Wszystkie dane zaakceptowane
    │   ├──reject               # Wszystkie dane odrzucone
    │   
    └── README.md               # Ten plik
## Uzywanie
Zdjęcia desek zaakceptowanych powinny trafić do folderu
```
data/accept
```
Zdjęcia desek odrzuconych powinny trafić do folderu
```
data/reject
```
Zdjęcia desek nieoznaczonych powinny trafić do folderu
```
data/unlabeled
```

Najpierw, aby stworzyć zbiór trenujący i testujący należy uruchomić skrypt:
```
script/generateTestTrain.py
```
Zmieniając parametr trainTestSplit można zmieniać stosunek wielkości zbioru trenującego. Domyślnie jest ustawiony na 0.7.

Aby wytrenować model i przetestować na danych testujących należy uruchomić skrypt: 
```
script/trainTest.py
```
Aby wytrenować model i sklasyfikować dane nieoznaczone należy uruchomić skrypt należy uruchomić:
```
script/classifyUnlabeled.py
```

w skryptach można zmieniać stosowany model wybierając odpowiednią funkcję z pliku:
```
script/createModel.py
```
