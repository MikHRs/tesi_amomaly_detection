# Protocollo di annotazione v0

Stato: bozza da validare con il relatore dopo un pilot di 1–2 clip.

## Obiettivo

La label principale è `anomalia`. La ground truth minima è una bounding box
per ogni oggetto anomalo visibile. Persone, pallet e scatole non sono label
principali separate: sono manifestazioni diverse dello stesso compito,
localizzare qualcosa che non dovrebbe trovarsi nella scena ferroviaria.
Separare subito le classi renderebbe il piccolo test set più frammentato e
sposterebbe il lavoro verso object detection supervisionata.

## Attributi proposti

- `tipo`: `persona`, `pallet`, `scatola`, `legno`,
  `oggetto_generico`, `altro`, `incerto`;
- `visibilita`: `chiaro`, `parziale`, `difficile`;
- `posizione`: `sui_binari`, `vicino_binari`, `fuori_roi`;
- usare le proprietà native CVAT `occluded` e `outside`.

Gli attributi descrivono i casi e aiutano l'analisi degli errori, ma non
cambiano la label principale.

## Cosa annotare

- persone sui binari o nell'area ferroviaria rilevante;
- pallet e scatole;
- assi, legno e ostacoli chiaramente visibili;
- oggetti grandi o estranei che possono interferire con il passaggio;
- altri casi indicati esplicitamente dal relatore.

La box deve contenere l'intero oggetto visibile con poco sfondo. Mantenere lo
stesso `track_id` finché si tratta dello stesso oggetto. Usare `occluded=1`
quando l'oggetto è presente ma parzialmente nascosto e `outside=1` quando non
è più presente: le box `outside=1` non sono ground truth attiva.

## Cosa non annotare

- binari, rotaie, massicciata e infrastruttura normale;
- sfondo e terreno;
- oggetti minuscoli quasi invisibili, salvo diversa indicazione;
- artefatti o riflessi termici non confermati come oggetti;
- elementi fuori dalla regione di interesse, salvo che servano come casi
  negativi concordati.

Non inferire un oggetto invisibile soltanto perché era presente nei frame
precedenti.

## Box, maschere e SAM2

Le bounding box sono il riferimento minimo e hanno priorità. Maschere o
poligoni si aggiungono soltanto se utili e sostenibili. SAM/SAM2 può proporre
una segmentazione o propagare un oggetto indicato con punti, box o maschera;
ogni risultato va controllato e corretto. SAM2 non decide cosa sia anomalo.

## Procedura e controllo qualità

1. Creare clip brevi e conservare il collegamento al video originale.
2. Annotare un pilot di 50–100 frame o alcuni eventi completi.
3. Controllare inizio/fine di ogni track, `outside`, occlusioni e box.
4. Esportare nel formato XML nativo CVAT e verificare il risultato con gli
   script del repository.
5. Rivedere tutti i casi dubbi con il relatore e aggiornare il protocollo.
6. Congelare una versione `v1` prima degli esperimenti finali.

Non distribuire frame adiacenti dello stesso evento tra insiemi sperimentali
diversi. Registrare anche clip o frame senza anomalie, necessari per misurare
i falsi positivi.
