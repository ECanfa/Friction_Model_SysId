Il presente lavoro si articola in 4 passi:
1. Generazione dei dataset nella directory DatasetGeneration/
2. Addestramento e test delle reti nelle directory CNN/ per le reti convolutive e LSTMId/ per le reti Long Short Term Memory
3. Generazione della traiettoria di pick and place
4. Analisi del comportamento

# Setup degli ambienti virtuali
Per evitare collisioni tra librerie consiglio di creare due ambienti virtuali. Il primo, usato per la quasi interezza, le cui dipendenze sono specificate nel file requirements.txt, e il secondo, usato esclusivamente per lo script genera\_pick\_and\_place.py, le cui dipendenze sono contenute in roboDK\_requirements.txt.

# Generazione dei dataset
Per generare i dataset di addestramento e test è necessario eseguire gli script presenti nella directory DatasetGeneration/ e crearsi una directory  Datasets/ dove verranno successivamente salvati.
Gli script da eseguire sono:
- python3 TrainingDataA.py
- python3 TrainingDataB.py
- python3 TestDataA.py
- python3 TestDataB.py
- python3 TestDataC.py

# Addestramento delle reti
Per quanto riguarda l'addestramento delle reti è sufficiente, una volta generati i dataset, spostarsi nelle cartelle CNN/ o LSTMId/, a seconda del tipo di rete che si vuole addestrare.
Per procedere all'addestramento è sufficiente eseguire:
- python3 CNN.py
- python3 LSTMid.py

Per addestramenti di giunti diversi è sufficiente modificare la variabile "giunto", con valore del giunto pari a x - 1 in cui x rappresenta il giunto preso in esame (es. se si vuole addestrare la rete del giunto 2, la variabile giunto dovrà essere uguale ad 1)

Nelle rispettive directory sono comunque già presenti i modelli per ogni giunto già addestrati.

# Test dei modelli
Una volta addestrati i modelli si può analizzare il comportamento principale delle reti su tre profili di velocità: velocità a gradino, velocità sinusoidali, velocità che si alternano tra molto lente, prossime a quella di Stribeck, fino a velocità più veloci, prossime ai limiti di funzionamento del giunto.

- python3 CNN\_test.py
- python3 LSTMid\_test.py

# Generazione di traiettorie di pick and place
Per generare le traiettorie di pick and place sono possibili due alternative.
L'alternativa più semplice è quella di eseguire lo script controllore.py e controllore_LSTM.py. Gli script generano in automatico la traiettoria di pick and place. È necessario eseguire gli script due volte, curandosi di modificare il paramentro di compesazione (una volta True una volta False) così da poter generare le traiettorie con e senza compensazione dell'attrito.
Gli script salveranno le traiettorie in file .csv nella directory dataset e anche file .mp4 che contengono la registrazione.

La seconda alternativa è quella di generare la traiettoria di pick and place con lo script genera\_pick\_and\_place.py. Per fare questo è necessario attivare il secondo ambiente virtuale per evitare conflitti con la libreria Pyside2. Sarà inoltre necessario aprire il programma roboDK e importare il file kukaiiwa\_14.rdk presente nella root directory del progetto.
- python3 genera\_pick\_and\_place.py

Una volta generata la traiettoria, tornando al primo ambiente virtuale si può eseguire lo script follow_test.py, una sola volta, poiché simula in sequenza le traiettorie con e senza compensazione in sequenza. Anch'esso salva automaticamente le traiettorie in file .csv e le registrazioni in file .mp4.

# Analisi del comportamento
Per verificare l'efficacia del lavoro svolto è possibile eseguire lo script control\_plot.py, nella cartella utils/,br che mostra la differenza tra la traiettoria desiderata (quella da inseguire) e quella effettivamente registrata dal sensore del giunto preso in esame, con e senza retroazione dell'attrito stimato.
È inoltre presente un secondo plot che mostra l'errore calcolato punto per punto sulla traiettoria.
Sarà necessario modificare i titoli dei .csv in base al comportamento che si vuole analizzare.

