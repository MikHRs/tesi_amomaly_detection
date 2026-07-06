# Appunti su SAM e SAM2

## SAM

Segment Anything Model è un modello di segmentazione *promptable*: dato un
prompt geometrico, per esempio uno o più punti o una bounding box, produce
una maschera dell'oggetto. Può trasferirsi zero-shot a immagini diverse da
quelle viste in addestramento, ma la qualità va verificata nel dominio
termico.

## SAM2

SAM2 estende l'idea a immagini e video. Una memoria temporale permette di
propagare l'identità e la maschera dell'oggetto nei frame successivi. Per
questo è interessante come supporto alla costruzione o correzione di track.

## Differenza operativa

- SAM lavora principalmente su immagini singole;
- SAM2 gestisce anche sequenze video e propagazione temporale;
- entrambi ricevono nativamente prompt geometrici, non una definizione
  autonoma di “anomalia”.

I prompt testuali elencati nel progetto non vengono passati direttamente a
SAM/SAM2. Servono a un eventuale detector open-vocabulary che propone box;
le box possono poi diventare prompt per SAM/SAM2.

## Uso proposto nella tesi

1. caricare video o frame;
2. scegliere un frame iniziale;
3. indicare un candidato con punto, box o maschera;
4. segmentarlo;
5. propagare la maschera nei frame successivi;
6. convertire maschere in box se necessario;
7. correggere manualmente drift, occlusioni ed errori;
8. salvare risultati e tempo di correzione.

SAM2 non decide cosa sia anomalo; serve come supporto per segmentare e
tracciare oggetti candidati. Per proporre automaticamente candidati occorre
un'altra strategia: un detector open-vocabulary, un operatore umano oppure
un metodo che modelli lo sfondo normale.

## Limiti nel termico

- domain shift rispetto alle immagini RGB di pre-training;
- bordi deboli e basso contrasto termico;
- oggetti piccoli o parzialmente occlusi;
- drift durante la propagazione;
- pseudo-colori che non corrispondono ai colori naturali;
- costo di memoria e necessità possibile di una GPU.

Questi limiti sono ipotesi da verificare qualitativamente e
quantitativamente, non risultati già dimostrati sui video della tesi.
