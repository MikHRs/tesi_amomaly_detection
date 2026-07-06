# Domande per il relatore

## Obiettivo e annotazione

1. Confermiamo `anomalia` come unica label principale?
2. Gli attributi `tipo`, `visibilita` e `posizione` sono utili o vanno ridotti?
3. La ground truth minima deve essere una bounding box o servono maschere?
4. Quali oggetti sono certamente anomalie e quali dipendono dal contesto?
5. Come trattiamo oggetti molto piccoli, parziali o quasi invisibili?
6. Qual è la regione ferroviaria di interesse: solo tra le rotaie o anche le
   aree adiacenti?
7. Quanti video, eventi e frame è realistico annotare per il test finale?
8. Possiamo mostrare nella tesi o nelle slide alcuni frame anonimizzati?

## Strumenti e dati

9. SAM2 va usato dentro CVAT, se disponibile, oppure come strumento separato?
10. È autorizzato usare RunPod, Colab, CVAT Online o altri servizi cloud?
11. Se sì, quali dati possono essere caricati e con quali precauzioni?
12. Dove vanno conservati originali, copie di lavoro ed export?

## Esperimenti

13. Quali famiglie di prompt hanno priorità?
14. Valutiamo il compito come localizzazione class-agnostic di `anomalia`?
15. Sono sufficienti IoU 0.3 e 0.5, precision, recall e F1?
16. Il dataset termico pubblico di videosorveglianza può essere usato solo
    come confronto esterno, dichiarando il forte domain gap?
17. Il fine-tuning resta fuori dal lavoro salvo tempo residuo?
