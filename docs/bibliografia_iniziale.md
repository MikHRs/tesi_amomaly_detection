# Bibliografia ragionata iniziale

Questa è una mappa di lettura, non un elenco definitivo. I dettagli
bibliografici vanno importati nel gestore bibliografico e ricontrollati prima
della consegna.

## 1. SAM

**Kirillov et al., “Segment Anything”, ICCV, 2023,
[arXiv:2304.02643](https://arxiv.org/abs/2304.02643).**

- Serve per definire la segmentazione promptable e i prompt punto/box.
- Supporta il capitolo sulle tecnologie e il flusso di annotazione assistita.
- Limite: non è un rilevatore autonomo di anomalie e nasce soprattutto da
  immagini non termiche.

## 2. SAM2

**Ravi et al., “SAM 2: Segment Anything in Images and Videos”, 2024,
[arXiv:2408.00714](https://arxiv.org/abs/2408.00714).**

- Serve per memoria video, propagazione temporale e correzioni interattive.
- Supporta il design del tracking/annotazione semi-automatica.
- Limite: un prompt iniziale resta necessario; prestazioni e costo nel
  dominio termico ferroviario devono essere misurati.

## 3. SAM per anomaly detection

Qui vanno cercati lavori che usano SAM come segmentatore di regioni candidate
o generatore di maschere, distinguendo sempre il modulo che assegna
l'anomaly score dal modulo che segmenta.

**Roth et al., “Towards Total Recall in Industrial Anomaly Detection”,
CVPR, 2022,
[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html).**

- PatchCore è un riferimento per localizzazione one-class basata su feature
  di immagini normali.
- Supporta il confronto concettuale con l'anomaly detection classica.
- Limite: nasce per ispezione industriale statica, non per ostacoli
  semantici in video ferroviari.

## 4. Zero-shot, open-vocabulary e vision-language

**Radford et al., “Learning Transferable Visual Models From Natural
Language Supervision”, ICML, 2021,
[PMLR](https://proceedings.mlr.press/v139/radford21a.html).**

- Introduce il riferimento CLIP per collegare testo e immagini.
- Supporta la motivazione dei prompt testuali e del trasferimento zero-shot.
- Limite: similarità immagine-testo non equivale a box o maschere precise.

**Liu et al., “Grounding DINO: Marrying DINO with Grounded Pre-Training for
Open-Set Object Detection”, 2023/2024,
[arXiv:2303.05499](https://arxiv.org/abs/2303.05499).**

- Fornisce un esempio di detector che trasforma espressioni testuali in box.
- Supporta la strategia testo → box → SAM/SAM2.
- Limite: il trasferimento da RGB a termico e prompt generici come
  `anomaly` non sono garantiti.

## 5. Anomaly detection classica

**Chandola, Banerjee e Kumar, “Anomaly Detection: A Survey”, ACM Computing
Surveys, 2009, [DOI](https://doi.org/10.1145/1541880.1541882).**

- Serve per definizioni, tassonomia e assunzioni sui dati normali/anomali.
- Supporta il capitolo del problema.
- Limite: precede i foundation model visivi e non tratta il caso specifico.

PatchCore, citato sopra, è una baseline moderna da discutere se si aggiunge
un esperimento one-class. Usare immagini normali del dominio significa però
adattarsi al dominio: non va chiamato zero-shot in senso stretto.

## 6. Immagini termiche e dominio ferroviario

**Li et al., “Segmenting Objects in Day and Night: Edge-Conditioned CNN for
Thermal Image Semantic Segmentation”, IEEE TNNLS, vol. 32, n. 7,
pp. 3069–3082, 2021 (early access 2020), DOI
[10.1109/TNNLS.2020.3009373](https://doi.org/10.1109/TNNLS.2020.3009373).**

- Il paper fornito introduce EC-CNN e il benchmark SODA, con 7.168 immagini
  termiche e 20 label semantiche.
- Supporta la discussione su segmentazione termica, bordi e domain gap.
- Limite: semantic segmentation supervisionata, non anomaly detection
  ferroviaria zero-shot.

**Nikolov et al., “Seasons in Drift: A Long Term Thermal Imaging Dataset for
Studying Concept Drift”, NeurIPS Datasets and Benchmarks, 2021,
[pagina proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/c45147dee729311ef5b5c3003946c48f-Abstract-round2.html).**

- Documenta la sorgente dei video termici da cui deriva il dataset esterno.
- Supporta la discussione su stagioni, drift e videosorveglianza termica.
- Limite: scenario fisso non ferroviario.

**Madan et al., “Self-Supervised Masked Convolutional Transformer Block for
Anomaly Detection”, 2022,
[arXiv:2209.12148](https://arxiv.org/abs/2209.12148).**

- È associato al *Thermal Anomaly Detection Dataset* pubblicato su
  [Kaggle](https://www.kaggle.com/datasets/neelu1/thermal-anomaly-detection-dataset):
  train normale e test anomalo per video anomaly detection open-set.
- Può supportare un piccolo controllo esterno della pipeline o una sezione
  sui benchmark termici.
- Limite: videosorveglianza non ferroviaria; la ground truth e il compito non
  vanno assunti compatibili con le box della tesi senza ispezionare i file.

Per lavori specifici su thermal object detection e railway obstacle
detection si farà una ricerca sistematica per query, anno e scenario,
registrando sensore, dataset, task, supervisione e disponibilità del codice.

## 7. Foundation model per anomaly localization

**Damm et al., “AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection
with DINOv2”, WACV, 2025,
[arXiv:2405.14529](https://arxiv.org/abs/2405.14529),
[codice ufficiale](https://github.com/dammsi/AnomalyDINO).**

- Usa patch feature DINOv2 e riferimenti normali per detection e
  localizzazione.
- È training-free ma, quando usa frame normali del dominio, è
  reference-based/few-shot e non zero-shot puro.
- Nasce su benchmark industriali RGB; il trasferimento a scene termiche
  complesse deve essere misurato.

**Fučka, Zavrtanik e Skočaj, “AnomalyVFM — Transforming Vision Foundation
Models into Zero-Shot Anomaly Detectors”, CVPR, 2026,
[arXiv:2601.20524](https://arxiv.org/abs/2601.20524),
[codice ufficiale](https://github.com/MaticFuc/AnomalyVFM).**

- Adatta foundation model visivi su dati ausiliari sintetici e fornisce
  checkpoint per anomaly detection/localization zero-shot.
- Consente un confronto senza costruire una memoria dai frame ferroviari.
- Il paper non dimostra prestazioni sul dominio termico della tesi.

## 8. SAM e anomaly/OOD detection

**Kanwal e Lee, “SAM-OOD: Foundation-Model-Guided Unknown Mining for
Object-Level Anomaly Detection”, CVPR Workshops, 2026.**

- Usa proposte FastSAM offline per creare box `unknown` e addestrare un
  detector K+1.
- Supporta il ruolo di SAM come generatore di proposte/pseudo-label.
- In inferenza SAM non viene usato; il metodo non coincide con la pipeline
  AnomalyDINO/AnomalyVFM → SAM e non è validato su termico.

**Yang et al., “Promptable Anomaly Segmentation with SAM Through
Self-Perception Tuning”, AAAI, 2025,
[arXiv:2411.17217](https://arxiv.org/abs/2411.17217).**

- Adatta SAM mediante parameter-efficient tuning e un adapter relazionale.
- È pertinente come sviluppo futuro per il domain shift.
- Non è un metodo zero-shot senza adattamento e non rientra nel nucleo
  leggero iniziale.

Il repository
[*SAM-Anomaly-Segmentation*](https://github.com/jinwwo/SAM-Anomaly-Segmentation)
è un prototipo non peer-reviewed che converte anomaly map reference-based in
prompt per SAM su MVTec-AD. È utile come esempio implementativo, non come
evidenza sul dominio termico.

## 9. Survey sulla segmentazione termica

**Kütük e Algan, “Semantic Segmentation for Thermal Images: A Comparative
Survey”, CVPR Workshops, 2022,
[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/html/Kutuk_Semantic_Segmentation_for_Thermal_Images_A_Comparative_Survey_CVPRW_2022_paper.html).**

- Riassume dataset e metodi RGB-T e thermal-only.
- Motiva thermal crossover, rumore, bassa risoluzione e bordi ambigui.
- È una survey di semantic segmentation, non di anomaly detection.
