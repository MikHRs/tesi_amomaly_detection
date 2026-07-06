# Protocollo E04 — baseline one-class di anomaly detection

> **Stato dopo l'incontro con il relatore:** protocollo storico, preparato ma
> non eseguito. Non è il protocollo finale perché ricava normalità e anomalie
> dagli stessi due video e usa la validation normale anche come classe
> negativa nelle metriche del notebook. La nuova valutazione usa il terzo
> video come sorgente normale separata e riserva normali indipendenti per il
> test.

## Obiettivo

Costruire un primo sistema che apprenda soltanto l'aspetto della scena
ferroviaria normale e assegni:

1. un anomaly score a ogni frame;
2. una heatmap che localizzi le regioni più diverse dalla normalità.

Non vengono usati prompt o nomi di oggetti per decidere l'anomalia.

## Split preparato

I frame sono campionati a 2 fps dai due video con annotazioni CVAT. Sono
esclusi 2 secondi attorno alle transizioni normale/anomalo per ridurre
l'effetto di confini temporali imprecisi.

| Split | Etichetta | Frame | Uso |
|---|---|---:|---|
| train | normale | 169 | costruzione della memoria della normalità |
| validation | normale | 42 | scelta della soglia senza vedere anomalie |
| test | anomalia | 895 | valutazione finale del pilot |

Le 3.473 box CVAT associate ai frame di test sono conservate soltanto per la
valutazione della localizzazione. Non entrano nel training.

## Metodo del pilot

La baseline è ispirata al principio di PatchCore:

1. un backbone ResNet18 pre-addestrato e congelato estrae feature locali;
2. le feature dei frame normali costituiscono una memoria della normalità;
3. per ogni patch di test si calcola la distanza dalla patch normale più
   vicina;
4. distanze alte producono una heatmap e un anomaly score alto.

Si tratta di una baseline one-class a feature patch, non di una riproduzione
integrale di PatchCore. Questa formulazione va mantenuta nella tesi.

## Valutazione prevista

- AUROC e average precision a livello di frame;
- precision, recall e F1 usando una soglia scelta sui soli frame normali di
  validation;
- esempi qualitativi delle heatmap;
- pointing game: verifica se il massimo della heatmap cade dentro almeno una
  box CVAT anomala.

AUROC, average precision e falsi positivi non devono essere riportati come
metriche finali usando i 42 frame di validation sia per la soglia sia come
negativi della valutazione. È necessario un test normale indipendente.

## Ruolo degli altri modelli

- Grounding DINO non decide se il frame è anomalo.
- SAM2 potrà ricevere in seguito un punto o una regione ricavati dalla heatmap
  per rifinire la segmentazione e seguire temporalmente l'anomalia.

## Limiti da dichiarare

- L'assenza di box CVAT viene interpretata come normalità e va controllata
  visivamente.
- Le annotazioni CVAT sono ancora da revisionare.
- Training e test provengono dalle stesse telecamere: il pilot valuta anomalie
  contestuali rispetto a ciascuna scena, non la generalizzazione a una nuova
  stazione.
- Il terzo video sarà tenuto fuori finché non verrà annotato e potrà diventare
  un test aggiuntivo.

L'ultimo punto è superato dalla decisione successiva del relatore: il terzo
video viene ora trattato, dopo controllo visivo, come sorgente della normalità
per AnomalyDINO.
