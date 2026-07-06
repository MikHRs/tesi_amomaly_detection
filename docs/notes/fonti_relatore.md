# Lettura critica delle fonti fornite dal relatore

Questa nota chiarisce come usare le fonti ricevute senza confondere semantic
segmentation, anomaly detection, OOD detection e segmentazione prompt-based.

## Kütük e Algan — survey sulla segmentazione termica

**Z. Kütük, G. Algan, “Semantic Segmentation for Thermal Images: A
Comparative Survey”, CVPR Workshops, 2022.**

[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/html/Kutuk_Semantic_Segmentation_for_Thermal_Images_A_Comparative_Survey_CVPRW_2022_paper.html)

Uso nella tesi:

- motivare vantaggi delle immagini termiche in condizioni di luce difficili;
- discutere bassa risoluzione, rumore, thermal crossover e bordi ambigui;
- presentare dataset RGB-T e solo termici.

Non è un lavoro di anomaly detection e non valuta SAM o foundation model.

## Li et al. — EC-CNN e SODA

Il riferimento IEEE con numero `9152142` è:

**C. Li et al., “Segmenting Objects in Day and Night: Edge-Conditioned CNN
for Thermal Image Semantic Segmentation”, IEEE TNNLS, 2021.**

[IEEE Xplore](https://ieeexplore.ieee.org/document/9152142)

È un metodo supervisionato di semantic segmentation termica. È utile per la
discussione sui bordi e sul dataset SODA, non come baseline zero-shot di
anomaly localization.

## Kanwal e Lee — SAM-OOD

**S. Kanwal, S.-I. Lee, “SAM-OOD: Foundation-Model-Guided Unknown Mining for
Object-Level Anomaly Detection”, CVPR Workshops, 2026.**

Il lavoro usa FastSAM offline durante il training per generare proposte
class-agnostic che non si sovrappongono alle box note. Le proposte diventano
una classe `unknown` per addestrare un Faster R-CNN K+1. In inferenza SAM non
viene usato.

Uso nella tesi:

- dimostrare che SAM può generare proposte o pseudo-label;
- discutere unknown mining e annotazioni incomplete;
- motivare una valutazione attenta delle soglie.

Non è una pipeline pronta AnomalyDINO→SAM e non è stata validata su immagini
termiche nel paper.

## Thermal Anomaly Detection Dataset

[Pagina Kaggle](https://www.kaggle.com/datasets/neelu1/thermal-anomaly-detection-dataset)

Il dataset deriva da *Seasons in Drift* e contiene 300 clip normali di
training, 29 clip anomale di test e annotazioni dei frame anomali. La licenza
dichiarata è CC BY 4.0.

È utile per video anomaly detection e per un controllo esterno a livello di
frame. Le annotazioni temporali non equivalgono a box o maschere spaziali:
per una valutazione della localizzazione servirebbe verificare i file ed
eventualmente creare annotazioni aggiuntive.

## SAM-Anomaly-Segmentation

[Repository GitHub](https://github.com/jinwwo/SAM-Anomaly-Segmentation)

Il repository valuta SAM su MVTec-AD con prompt derivati dalla ground truth e
con prompt automatici ricavati da mappe di similarità rispetto a immagini
normali. È un prototipo utile per la pipeline:

```text
anomaly map → soglia → box/punto → SAM
```

Non è una fonte peer-reviewed, non usa dati termici e segnala sensibilità
alla scelta dei riferimenti e della soglia.

## arXiv:2411.17217

Il link corrisponde a:

**H.-Y. Yang et al., “Promptable Anomaly Segmentation with SAM Through
Self-Perception Tuning”, AAAI 2025.**

[arXiv](https://arxiv.org/abs/2411.17217)

Il metodo introduce tuning parameter-efficient e adapter per adattare SAM
all'anomaly segmentation. Non è quindi la prova zero-shot leggera richiesta
come primo esperimento, ma può essere citato tra gli sviluppi futuri.

## Riferimenti corretti per AnomalyDINO e AnomalyVFM

**AnomalyDINO:** S. Damm et al., WACV 2025,
[paper](https://arxiv.org/abs/2405.14529) e
[codice ufficiale](https://github.com/dammsi/AnomalyDINO).

- patch matching con DINOv2;
- one/few-shot con immagini normali di riferimento;
- training-free;
- score di immagine e anomaly map.

**AnomalyVFM:** M. Fučka et al., CVPR 2026,
[paper](https://arxiv.org/abs/2601.20524) e
[codice ufficiale](https://github.com/MaticFuc/AnomalyVFM).

- trasforma foundation model visivi in detector zero-shot;
- usa pesi adattati su dati sintetici ausiliari;
- non richiede frame normali del dominio in inferenza;
- produce score e localizzazione.

Questa differenza determina due protocolli sperimentali separati e impedisce
di usare impropriamente la parola `zero-shot` per entrambi.
