# 📅 Plan de proiect — PulmoGnosis

Ghidul complet pentru terminarea proiectului: ce s-a făcut, ce urmează, cum și de ce.
Este scris astfel încât **o singură persoană** (eventual cu un asistent AI) să poată duce proiectul
până la capăt doar din acest fișier și din [README.md](README.md).

**Responsabil pentru etapele 2–6:** `TODO (numele colegului)`.
Săptămânile sunt numerotate de la **S1 = 26 oct. – 1 nov. 2026**.

> **Terminologie:** **„Etapa 1–6"** = etapele de evaluare de la laborator (Lab 2 – Lab 7).
> **„Etapa 0 / 1 / 2 a pipeline-ului"** = părțile sistemului din README: preprocesare /
> localizarea tumorii (U-Net) / clasificatorul. Nu le confundați.

---

## Cuprins

0. [Cum se folosește acest plan](#0-cum-se-folosește-acest-plan)
1. [Stadiul curent și termenele](#1-stadiul-curent-și-termenele)
2. [Contextul proiectului](#2-contextul-proiectului)
3. [Datele](#3-datele)
4. [Reguli de lucru (obligatorii)](#4-reguli-de-lucru-obligatorii)
5. [Arhitectura codului](#5-arhitectura-codului)
6. [Etapele 1–6 în detaliu](#6-etapele-16-în-detaliu)
7. [Resurse de calcul](#7-resurse-de-calcul)
8. [Riscuri și planuri de rezervă](#8-riscuri-și-planuri-de-rezervă)
9. [Comenzi utile](#9-comenzi-utile)

---

## 0. Cum se folosește acest plan

1. Citiți **în ordine**: această secțiune → secțiunea 1 (stadiu) → README.md (arhitectura) → secțiunile 2–5 → etapa curentă din secțiunea 6.
2. Lucrați etapele **în ordine**. Fiecare etapă are: obiectiv, sarcini numerotate, ce apare în aplicație,
   ce intră în raport, criteriul „Gata când" și riscuri.
3. După fiecare etapă: bifați „Gata când", actualizați **secțiunea 1** (stadiul), faceți *tag* git (`etapa-N`).
4. **Dacă folosiți un asistent AI:** dați-i acest fișier și README.md ca punct de plecare și cereți-i să
   respecte secțiunea 4. Cereți-i să vă întrebe înainte de descărcări mari, conturi/chei API, ștergeri
   și înainte de `push`. Verificați voi rezultatele — un agent poate greși cifre.
5. Când o decizie din plan se schimbă, **actualizați planul** în același commit cu codul.

---

## 1. Stadiul curent și termenele

> **Ipoteze despre date (de verificat):** laboratoarele sunt la 2 săptămâni, cu Lab 2 în S1. Lab 6 ar cădea
> în vacanța de iarnă, deci planul presupune Lab 6 și Lab 7 în ianuarie. Corectați coloana *Termen*.

| Etapa | Termen | Aplicație | Pct. | Raport | Pct. | Stare |
|---|---|---|---|---|---|---|
| **1** | **Lab 2** — S1 (26 oct. – 1 nov.) | Flow-ul de bază | 10 | Funcționalitățile aplicației; descrierea plastică și formală a problemei; *related work*, instrumente și tehnologii | 5 | 🟡 Aproape gata |
| **2** | **Lab 3** — S3 (9–15 nov.) | Algoritm inteligent + date puține | 5 | Algoritmul, metodologia experimentală, rezultate | 5 | ⬜ |
| **3** | **Lab 4** — S5 (23–29 nov.) | Algoritm inteligent + date reale / *benchmark* | 5 | Metodologie, rezultate, **comparație cu SOTA** (cele mai bune rezultate publicate) | 5 | ⬜ |
| **4** | **Lab 5** — S7 (7–13 dec.) | Îmbunătățirea algoritmului | 10 | Îmbunătățirile: metodologie, rezultate, analize comparative | 10 | ⬜ |
| **5** | **Lab 6** — ~S11 (4–10 ian.) | Integrarea modelului în aplicație (5) + noi îmbunătățiri (10) + **teaser video** (5) | 20 | Analiza statistică a rezultatelor (10) + slide-uri, draft (10) | 20 | ⬜ |
| **6** | **Lab 7** — ~S13 (18–24 ian.) | **Demo cu clienți** = pitch + demo aplicație | 10 | Analiza SWOT și concluzii + slide-uri, versiune finală | 10 | ⬜ |
| | | **Total aplicație** | **60** | **Total raport** | **55** | |

**Ce cântărește cel mai mult:** etapele 4 și 5 (60 din 115 puncte).

### Ce există deja în repository
- **Aplicația demo** (`app/`, Streamlit): flow complet în 5 pași — încărcare DICOM → vizualizare CT + PET la
  rezoluția nativă → date clinice → localizarea tumorii (euristică + corecție manuală) → panou de rezultate
  cu valori *placeholder* marcate vizibil „DEMO".
- **Pachetul `src/pulmognosis/`**: citire DICOM, conversii HU/SUV, alinierea PET pe CT, extragerea cubului
  de intrare al modelului (**128³ voxeli de 0,75 mm**), interfața `Predictor` pentru modele, o fantomă
  sintetică CT + PET scrisă ca DICOM (pentru demo și teste).
- **21 de teste** (`pytest`).
- **Raportul** (`raport_biopsie_virtuala.tex`) cu secțiunile Etapei 1 și cu tabelele de rezultate goale.

### Ce NU există în repository (se obține local)
Datele (`data/`), metadatele TCIA, mediul Python (`.venv`), greutățile modelelor. Vezi secțiunile 3 și 9.

---

## 2. Contextul proiectului

### 2.1 Problema
PulmoGnosis este un sistem de suport al deciziei pentru **cancerul pulmonar fără celule mici (NSCLC)**.
Din **CT** + **PET** (opțional) + **vârstă, gen, statut de fumător**, sistemul:
1. **localizează tumora** (Etapa 1 a pipeline-ului);
2. estimează **tipul histologic**: adenocarcinom / carcinom scuamos / carcinom cu celule mici;
3. estimează probabilitatea mutațiilor **EGFR** și **KRAS**;
4. arată **harta Grad-CAM** (ce regiuni au influențat predicția).

**Ce NU prezice:** ALK (doar 2 cazuri pozitive în date), ROS1 și PD-L1 (nu există în datele publice).

### 2.2 Pipeline-ul
```
DICOM CT (+PET) ──► Etapa 0: preprocesare ──► Etapa 1: localizare ──► cub 128³ ──► Etapa 2: clasificator
                    HU, SUV, PET aliniat        (U-Net; azi:               (0,75 mm,     3D ResNet-18 + MLP clinic
                    pe grila CT nativă           euristică)                 96 mm)        → histologie, EGFR, KRAS
```
Detaliile arhitecturii sunt în README.md, secțiunile 1–4.

### 2.3 Decizii luate (nu le schimbați fără motiv documentat)

| Decizie | Valoare | De ce |
|---|---|---|
| Cubul de intrare al clasificatorului | **128³ voxeli de 0,75 mm** (96 mm latură), re-eșantionat direct din grilele native CT și PET | Aproape de rezoluția originală a CT-ului; 96 mm cuprinde și tumorile mari. Constante: `MODEL_PATCH_SIZE`, `MODEL_SPACING_MM` în `pipeline.py` |
| Afișarea în aplicație | CT la rezoluția nativă, PET aliniat pe grila CT | Nu se pierde detaliu |
| Encoder imagine | 3D ResNet-18, inițializat din **MedicalNet** | 3D ResNet-50 de la zero ar supra-învăța pe ~200 de pacienți |
| Fuziune | *Late fusion*: vector imagine (512) + MLP clinic (32) → straturi comune → 3 ieșiri | Fiecare tip de date procesat separat |
| Ieșiri | Histologie (*softmax*), EGFR și KRAS (*sigmoid*) | Multi-task: histologia are mult mai multe etichete și ajută mutațiile |
| Etichete lipsă | **Pierdere mascată** (eticheta lipsă nu contribuie) | Nu se aruncă pacienți și nu se presupune „negativ" |
| PET lipsă | Canal PET = 0 + *modality dropout* la antrenare | Modelul trebuie să meargă și doar cu CT |
| Evaluare | **Validare încrucișată în 5 fold-uri, stratificată, la nivel de pacient**; AUC cu IÎ 95% bootstrap | Prea puține date pentru o împărțire fixă 70/15/15 |
| Localizare (pipeline Etapa 1) | 3D U-Net pre-antrenat pe LIDC-IDRI, ajustat pe măștile NSCLC | Etichete multe pentru localizare; tumorile NSCLC sunt mai mari decât nodulii LIDC |

---

## 3. Datele

Toate seturile sunt publice, anonimizate, pe **TCIA** (The Cancer Imaging Archive); aceleași date sunt și în
**NCI Imaging Data Commons (IDC)**, de unde se pot descărca selectiv cu pachetul Python `idc-index`
(identificatorii colecțiilor se găsesc interogând indexul — probabil `nsclc_radiogenomics`,
`lung_pet_ct_dx`, `lidc_idri`; de verificat).

### 3.1 Rezumat

| Set | Pacienți | Mărime | Modalități | Etichete | Rol | Etapa |
|---|---|---|---|---|---|---|
| **NSCLC-Radiogenomics** | 211 | 98 GB | CT, PET (201), SEG (144) | EGFR, KRAS, histologie, vârstă, gen, fumat | Singura sursă pentru mutații | 2, 3, 4, 5 |
| **Lung-PET-CT-Dx** | 355 | 127 GB | CT, PET (parțial) | Histologie, casete în jurul tumorii, vârstă, gen, fumat | Histologie (mai multe date), validare externă | 3, 4, 5 |
| **LIDC-IDRI** | 1010 | 133 GB | CT | Contururi de noduli (până la 4 radiologi) | Antrenarea U-Net (localizare) | 4, 5 |
| Kaggle *Chest CT-Scan images* | ~1000 imagini 2D | 125 MB | JPG/PNG | 4 clase (3 tipuri + normal) | Doar baseline 2D | 2 |

### 3.2 NSCLC-Radiogenomics — detalii verificate

- **Metadate** (de descărcat în `data/metadata/`):
  - date clinice + etichete: <https://www.cancerimagingarchive.net/wp-content/uploads/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv>
  - inventarul seriilor: <https://www.cancerimagingarchive.net/wp-content/uploads/NSCLC_Radiogenomics-6-1-21-Version-4-nbia-digest.xlsx>
- **Coloane utile** în CSV: `Case ID`, `Age at Histological Diagnosis`, `Gender` (Male/Female),
  `Smoking status` (Former/Nonsmoker/Current), **`Histology `** (atenție: spațiu la final),
  `EGFR mutation status` / `KRAS mutation status` (Mutant / Wildtype / Unknown / Not collected),
  `ALK translocation status`.
- **Doar Mutant și Wildtype sunt etichete cunoscute.** Unknown și Not collected = lipsă (pierdere mascată), niciodată „negativ".

| | Total | Cu PET | Cu mască tumorală (SEG) |
|---|---|---|---|
| Pacienți | 211 (162 `R01-*`, 49 `AMC-*`) | 201 | **144** (doar `R01-*`) |
| EGFR cunoscut (mutant) | 172 (43) | 163 (39) | 117 (23) |
| KRAS cunoscut (mutant) | 171 (38) | 162 (36) | 115 (27) |
| ALK cunoscut (translocat) | 157 (**2**) | 152 (2) | 111 (2) |
| Histologie | 172 adeno / 35 scuamos / 4 NOS | | 112 adeno / 29 scuamos / 3 NOS |

> ⚠️ **Capcana 1 — PET-ul NU e aliniat cu masca.** La **toți cei 144** de pacienți cu SEG, PET-ul este
> într-un **alt studiu** (altă zi, altă poziționare) decât CT-ul diagnostic pe care e desenată masca.
> Alinierea prin coordonate fizice (`resample_to_reference`) **nu funcționează** între studii diferite.
> **Soluție:** 143 din 144 de studii PET conțin propriul CT (low-dose). Se face **registrarea** acestui CT
> pe CT-ul diagnostic (SimpleITK, rigid apoi eventual afin, metrică *Mattes Mutual Information*) și se
> aplică aceeași transformare PET-ului. Verificare vizuală obligatorie.

> ⚠️ **Capcana 2 — două serii PET.** Fiecare studiu PET are `PET_BODY_CTAC` (corectat de atenuare — **cel
> corect**) și `PET_BODY_NO_AC` (necorectat — **nu se poate folosi pentru SUV**). `pick_series` alege deja
> seria corectă; folosiți-l.

### 3.3 Lung-PET-CT-Dx
- Metadate clinice: <https://www.cancerimagingarchive.net/wp-content/uploads/statistics-clinical-20201221.xlsx>
  (coloane: `NewPatientID`, `Sex`, `Age`, `weight (kg)`, `T-Stage`, `N-Stage`, `M-Stage`,
  `Histopathological grading`, `Smoking History` 0/1).
- **Histologia e codată în ID-ul pacientului:** `A` = adenocarcinom (251), `B` = celule mici (38),
  `E` = celule mari (5), `G` = scuamos (61).
- **Adnotări:** casete (*bounding boxes*) în format PASCAL VOC (XML), câte un fișier pe secțiune; descărcare
  separată de pe pagina TCIA a setului. De verificat cum se leagă fișierele de secțiuni (probabil prin
  `SOPInstanceUID`). Din casetele de pe mai multe secțiuni rezultă o casetă 3D → centrul tumorii.
- PET doar pentru o parte din pacienți (adăugat ulterior pentru 140). De verificat dacă CT și PET sunt în
  același *frame of reference* (de obicei da, fiind același examen PET/CT).
- ⚠️ **Capcana 3:** carcinomul cu celule mici există **doar** aici ⇒ un model poate învăța „spitalul" în loc
  de clasă. Evaluați această clasă separat; comparația adenocarcinom vs. scuamos se face și între seturi.

### 3.4 LIDC-IDRI
- 1010 pacienți CT; contururile nodulilor în XML, citite cu **`pylidc`** (necesită un fișier de configurare
  care indică folderul DICOM). Alternativ: subsetul **LUNA16** (888 scanări, pregătit pentru detecție).
- Folosit doar pentru localizare (U-Net). Nu are PET, histologie sau mutații.

### 3.5 Kaggle *Chest CT-Scan images* (mohamedhanyyy)
- <https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images> — necesită cont Kaggle.
- Imagini 2D JPG/PNG, fără HU, **fără ID de pacient** ⇒ risc de *data leakage*. Doar baseline 2D, raportat ca atare.

### 3.6 Spațiu pe disc
Un pacient NSCLC-Radiogenomics ≈ **450 MB** (CT + PET). Strategia: descărcare **pe loturi** (ex. 10 pacienți),
procesare, păstrarea doar a cuburilor (`float16`, ~8 MB/pacient) în `data/processed/`, ștergerea DICOM-urilor
brute. Verificați spațiul liber înainte.

---

## 4. Reguli de lucru (obligatorii)

### 4.1 Mediu și repository
- **Python 3.12**, mediu în `.venv`:
  `py -3.12 -m venv .venv` → `.venv\Scripts\pip install -r requirements.txt -e .` → `pytest`.
- **Repository-ul nu are `.gitignore` (intenționat).** Fiecare membru pune regulile **local**, în
  `.git/info/exclude`, înainte să creeze date:
  ```
  /data/
  /checkpoints/
  /runs/
  /wandb/
  /mlruns/
  .venv/
  __pycache__/
  *.py[cod]
  .pytest_cache/
  *.egg-info/
  .env
  *.dcm
  *.nii
  *.nii.gz
  *.npz
  *.pt
  *.pth
  *.ckpt
  ```
  Regulile pentru foldere încep cu `/` — altfel `data/` ar ascunde și `src/pulmognosis/data/`.
  Verificați cu `git status --ignored`.
- Bibliotecile noi se adaugă în `requirements.txt` (cu versiune minimă).

### 4.2 Git
- O ramură pentru fiecare sarcină (`feature/...`), commit-uri mici, `pytest` trebuie să treacă.
- Unire în `main` prin *pull request*. **Tag** pentru versiunea prezentată la fiecare laborator: `etapa-N`.
- **Nu se pun niciodată în git:** date, DICOM, cuburi `.npz`, greutăți de modele, `.venv`, nimic din `/data/`.
- **Nu se pun în git fișiere de configurare ale asistenților AI** (`CLAUDE.md`, `AGENTS.md`, `.claude/`,
  `.cursor/` etc.) și **nu se adaugă linii `Co-Authored-By`** sau alte atribuiri AI în commit-uri/PR-uri.
- Greutățile modelelor folosite de aplicație se distribuie prin **GitHub Releases** (fișiere de până la 2 GB),
  cu un script care le descarcă în `checkpoints/`.

### 4.3 Limbă
- Cod, comentarii, identificatori, mesaje de commit: **engleză**.
- README, PLAN, raport, texte din aplicație: **română**.

### 4.4 Învățare automată
- **Împărțire la nivel de pacient**, stratificată. Fold-urile se generează **o singură dată** (seed fix),
  se salvează ca CSV mic în `configs/splits/` și sunt folosite de **toate** modelele.
- Statisticile de normalizare (media/deviația vârstei, intensități) se calculează **doar pe fold-ul de antrenare**.
- Etichete lipsă → **pierdere mascată**. Niciodată „lipsă = negativ".
- Raportați pentru fiecare ieșire: **ROC-AUC cu IÎ 95% (bootstrap, ≥ 1000 re-eșantionări)**, F1, sensibilitate,
  specificitate și **numărul de pacienți și de pozitivi** din fiecare fold. Semnalați etichetele cu prea puțini pozitivi.
- *Seed* fix și fișierul de configurare salvat lângă rezultatele fiecărei rulări (`runs/<nume>/`).
- Preprocesarea (HU, SUV, orientare, centrare, registrare) se testează cu teste unitare **și** vizual — aceste erori nu dau mesaje.

### 4.5 Onestitate
- **Nicio cifră în raport sau în aplicație care nu provine dintr-o rulare reală.** Ce nu e măsurat → `TODO`.
- **Fiecare referință bibliografică trebuie să fie reală și verificabilă** (DOI / PubMed / arXiv), citită de voi.
- Aplicația marchează vizibil orice ieșire care nu vine dintr-un model antrenat.

---

## 5. Arhitectura codului

### 5.1 Ce există

| Fișier | Rol |
|---|---|
| `src/pulmognosis/data/dicom.py` | `scan_folder` (grupează fișierele pe serii), `pick_series` (preferă PET corectat), `load_series` (volum 3D SimpleITK; CT iese în HU), `read_header` |
| `src/pulmognosis/data/synthetic.py` | Fantoma sintetică CT + PET (DICOM); `DEMO` (512², 0,7 mm) și `TEST` (mică, pentru teste) |
| `src/pulmognosis/preprocessing/intensity.py` | Ferestre CT, normalizare HU, `suv_scale_factor` (cu corecția dezintegrării, tag-ul Philips, trecerea peste miezul nopții) |
| `src/pulmognosis/preprocessing/spatial.py` | `resample_isotropic`, `resample_to_reference` (aliniere prin coordonate fizice), `resample_around` (cub centrat pe un punct în mm), `to_array` |
| `src/pulmognosis/pipeline.py` | `prepare_case` → `Case` (CT nativ în HU, PET în SUV pe grila CT și pe grila proprie); `extract_model_patch` → cubul 128³ CT și PET; `ClinicalData`; constantele cubului |
| `src/pulmognosis/inference.py` | Protocolul **`Predictor`** (`localise`, `predict`), `Localisation`, `Prediction`, `PlaceholderPredictor` |
| `app/streamlit_app.py`, `app/rendering.py` | Aplicația; vorbește cu modelele **doar** prin `get_predictor()` |
| `tests/` | Teste pentru intensități, geometrie, serii DICOM, pipeline-ul complet pe fantomă |

### 5.2 Ce trebuie adăugat (unde și cu ce interfață)

| Modul | Conținut | Etapa |
|---|---|---|
| `src/pulmognosis/data/cohort.py` + `scripts/build_cohort.py` | Tabelul de cohortă (CSV în `data/`): pacient × modalități × etichete | 2 |
| `src/pulmognosis/data/download.py` + `scripts/download_batch.py` | Descărcare selectivă (`idc-index`) pe loturi | 2 |
| `src/pulmognosis/data/seg.py` | Citirea DICOM SEG (ex. `highdicom`) → mască 3D → centrul tumorii în mm | 2 |
| `src/pulmognosis/preprocessing/registration.py` | Registrarea CT (studiul PET) → CT diagnostic; aplicarea transformării pe PET | 3 |
| `src/pulmognosis/data/lung_pet_ct_dx.py` | Casetele PASCAL VOC → caseta 3D → centrul tumorii | 3 |
| `scripts/build_cubes.py` | Pacient → `data/processed/<set>/<pacient>.npz` cu `ct` (float16, HU), `pet` (float16, SUV sau lipsă), `center_mm`, `spacing_mm` | 2–3 |
| `src/pulmognosis/evaluation/` | `splits.py` (fold-uri), `metrics.py` (AUC, F1, sens./spec.), `bootstrap.py` (IÎ), `delong.py` (test DeLong), `report.py` (tabele LaTeX) | 2, 5 |
| `src/pulmognosis/classification/` | `clinical.py` (model A), `cnn2d.py` (model B), `model3d.py` (3D ResNet-18 + MLP + capete), `dataset.py` (citirea cuburilor + augmentări), `train.py` | 2–4 |
| `src/pulmognosis/segmentation/` | U-Net 3D (antrenare, inferență cu fereastră glisantă) | 4–5 |
| `src/pulmognosis/explain.py` | Grad-CAM pe clasificator | 5 |
| `configs/` | YAML pentru experimente + `configs/splits/*.csv` | 2+ |
| `scripts/download_weights.py` | Descarcă greutățile din GitHub Releases în `checkpoints/` | 3+ |

### 5.3 Cum intră un model în aplicație
Aplicația apelează doar `get_predictor()` din `app/streamlit_app.py`, care întoarce un obiect cu:
- `localise(case) -> Localisation` (centrul tumorii, ca index `(z, y, x)` în CT-ul nativ, + metoda);
- `predict(case, location, clinical) -> Prediction` (histologie, EGFR, KRAS, `is_placeholder`, notă).

Pentru un model nou: scrieți o clasă cu aceste două metode în `src/pulmognosis/` și schimbați linia din
`get_predictor()`. **Aplicația nu se restructurează.** Fiecare ieșire trebuie să spună clar de unde vine
(ex. „model clinic, antrenat pe 172 de pacienți"); ce e încă *placeholder* rămâne marcat.

---

## 6. Etapele 1–6 în detaliu

### Etapa 1 → Lab 2 (S1) — 15 puncte

| Sarcină | Stare |
|---|---|
| Structura repository-ului, `requirements.txt`, `pyproject.toml`, Python 3.12 | ✅ |
| Aplicația v0 (flow complet în 5 pași, *placeholder* marcat) | ✅ |
| Preprocesare (Etapa 0 a pipeline-ului) cu teste | ✅ |
| Raport: funcționalitățile aplicației, problema (plastic + formal), instrumente și tehnologii | ✅ |
| **Raport: *related work*** — 3–4 lucrări reale (predicția EGFR din CT; histologie din CT/PET; radiomică pe NSCLC-Radiogenomics; detecție/segmentare pe LIDC/LUNA16), fiecare cu *Date / Algoritm / Rezultate (cifre exacte)* | ⬜ |
| **Raport: capturi de ecran** ale celor 5 pași (cazul demonstrativ) | ⬜ |
| Compilarea raportului (Overleaf, compilator **XeLaTeX**) | ⬜ |
| Tag `etapa-1` | ⬜ |

**Gata când:** aplicația rulează cap-coadă ✅; raportul are funcționalitățile, problema, *related work* real, instrumentele.

---

### Etapa 2 → Lab 3 (S2–S3) — 10 puncte: „algoritm inteligent + small data"

**Obiectiv:** trei modele reale pe date puține; unul dintre ele intră în aplicație.

| # | Algoritm | Date | Prezice | În aplicație? |
|---|---|---|---|---|
| **A** | Regresie logistică pe vârstă, gen, fumat | Tabelul clinic NSCLC-Radiogenomics (~172 cu EGFR cunoscut) | EGFR, KRAS | ✅ |
| **B** | CNN 2D (ResNet-18 pre-antrenat ImageNet, ajustat fin) | Kaggle, ~1000 imagini | Histologie (3 + normal) | ❌ (doar raport) |
| **C** | 3D ResNet-18 (MedicalNet) pe cubul 128³ | **~40 de pacienți** NSCLC-Radiogenomics cu SEG, aleși stratificat după histologie | Adenocarcinom vs. scuamos | ❌ (doar raport) |

**Sarcini, în ordine:**
1. Mediul, `.git/info/exclude`, metadatele în `data/metadata/` (secțiunile 3.2, 3.3, 4.1).
2. **Cohorta** (`build_cohort.py`): verificați că obțineți cifrele din secțiunea 3.2 (211 / 144 / 43 din 172).
3. **Evaluarea** (`evaluation/`): fold-uri stratificate pe pacient, salvate în `configs/splits/`; AUC + bootstrap; metrici.
4. **Modelul A**: standardizare în interiorul *pipeline*-ului de validare; coeficienții salvați ca JSON mic
   (pot fi în git — nu sunt date de pacient). Rezultate pe 5 fold-uri cu IÎ.
5. **Modelul A în aplicație**: clasă nouă `Predictor` — EGFR/KRAS reale, etichetate „model clinic (regresie
   logistică), antrenat pe N pacienți"; histologia rămâne *placeholder* marcat; localizarea rămâne euristică.
   Teste pentru noua clasă.
6. **Date pentru modelul C**: descărcare pe loturi de 10 a ~40 de pacienți (CT diagnostic + SEG; PET-ul doar
   dacă rămâne timp — vezi Capcana 1); masca SEG → centrul tumorii în mm → `extract_model_patch` → `.npz`;
   notebook de verificare vizuală (tumora e în centrul fiecărui cub).
7. **Modelul B** pe Kaggle (cu limitările scrise explicit).
8. **Modelul C**, **doar CT** (PET-ul aliniat corect vine în Etapa 3): precizie mixtă, batch 4, aceleași fold-uri.
9. **Raport** (secțiunea „Metodologia experimentală" + o subsecțiune „Etapa 2: rezultate pe date puține"):
   descrierea algoritmilor A, B, C, a datelor, a fold-urilor și a metricilor; tabelul de rezultate cu IÎ și
   numărul de pacienți/pozitivi.
10. Tag `etapa-2`.

**Aplicația după Etapa 2:** EGFR/KRAS reale (doar din date clinice); histologie *placeholder*; localizare euristică.

**De aprobat înainte:** PyTorch cu CUDA (~2,5 GB), ~18 GB de scanări (pe loturi), greutățile MedicalNet (~100–200 MB), cont Kaggle.

**Așteptări:** modelul C va fi slab, cu IÎ largi — normal pentru „small data". Comparația importantă: imaginile vs. modelul A.

**Gata când:**
- [ ] 3 modele antrenate, cu rezultate reale (IÎ 95%, nr. pacienți și pozitivi);
- [ ] aplicația afișează predicția reală a modelului A, alături de *placeholder*-ele marcate;
- [ ] raportul descrie algoritmii, metodologia și rezultatele;
- [ ] tag `etapa-2`.

---

### Etapa 3 → Lab 4 (S4–S5) — 10 puncte: „algoritm inteligent + date reale + comparație cu SOTA"

**Obiectiv:** clasificatorul antrenat pe **toate** datele disponibile, cu PET aliniat corect, comparat cu
baseline-urile și cu literatura; histologia reală intră în aplicație.

**Sarcini, în ordine:**
1. **Registrarea PET** (`preprocessing/registration.py`, Capcana 1): CT-ul din studiul PET → CT-ul diagnostic
   (SimpleITK: inițializare pe centrul geometric, rigid, *Mattes MI*, multi-rezoluție; apoi eventual afin);
   aceeași transformare aplicată PET-ului; scorul de registrare salvat; **verificare vizuală pe toți
   pacienții** (hotspot-ul PET trebuie să cadă pe masca tumorii). Pacienții la care registrarea eșuează
   rămân doar cu CT (canal PET = 0) și se raportează câți sunt.
2. **Toți cei 144 de pacienți NSCLC-Radiogenomics cu SEG** → cuburi CT + PET (pe loturi, ștergând brutele).
3. **Lung-PET-CT-Dx**: descărcare pe loturi; casetele → centrul tumorii → cuburi; histologia din ID.
   Verificați *frame of reference* CT–PET (Capcana 3 la evaluare).
4. **Baseline radiomic**: PyRadiomics pe masca SEG (NSCLC-Radiogenomics) + regresie logistică, aceleași fold-uri.
5. **Clasificatorul** (3D ResNet-18, CT+PET, MedicalNet) pe datele complete:
   - histologie adeno vs. scuamos pe ambele seturi (~520 de pacienți); celule mici raportat separat;
   - EGFR și KRAS pe NSCLC-Radiogenomics (pierdere mascată).
6. **Tabelul SOTA**: rezultatele noastre vs. baseline-uri vs. cifrele publicate în lucrările din *related work*
   (aceeași sarcină, același set dacă se poate; diferențele de protocol menționate).
7. **Aplicația v1**: `TrainedPredictor.predict` folosește clasificatorul pentru histologie (și EGFR/KRAS dacă
   bate modelul A; altfel rămâne modelul A, cu explicație). Greutățile → GitHub Releases + `download_weights.py`.
   Localizarea rămâne euristică/manuală ⇒ în aplicație se spune că cubul e centrat de utilizator.
8. **Raport**: datele complete, registrarea (cu câți pacienți au reușit), rezultatele cu IÎ, tabelul SOTA.
9. Tag `etapa-3`.

**Aplicația după Etapa 3:** histologie reală (imagine); EGFR/KRAS din cel mai bun model; localizare euristică + manuală.

**Gata când:**
- [ ] PET aliniat și verificat vizual; numărul de eșecuri raportat;
- [ ] rezultate pe seturile complete, cu IÎ;
- [ ] tabel: modelul nostru vs. baseline-uri (clinic, radiomic, 2D) vs. lucrări publicate;
- [ ] aplicația folosește clasificatorul real pentru histologie;
- [ ] tag `etapa-3`.

---

### Etapa 4 → Lab 5 (S6–S7) — 20 de puncte: „îmbunătățirea algoritmului"

**Obiectiv:** îmbunătățiri măsurate față de versiunea din Etapa 3, fiecare cu o **ablație** (se adaugă/elimină
o singură componentă, restul identic: aceleași fold-uri, același seed, aceleași epoci).

**Experimente, în ordine** (fiecare = un rând în tabelul de ablație):

| # | Variantă | Întrebare |
|---|---|---|
| 0 | Versiunea din Etapa 3 (referința) | — |
| 1 | Inițializare aleatoare vs. **MedicalNet** | Ajută pre-antrenarea? |
| 2 | Doar CT vs. **CT + PET** (+ *modality dropout*) | Ajută PET-ul? |
| 3 | Doar imagine vs. **+ MLP clinic** (fuziune târzie) | Ajută datele clinice? |
| 4 | Ieșiri separate vs. **multi-task** (histologie + EGFR + KRAS, pierdere mascată) | Ajută histologia mutațiile? |
| 5 | BCE simplă vs. **pierdere ponderată / *focal loss*** | Ajută la dezechilibrul claselor? |
| 6 | Fără vs. **cu augmentări 3D** (rotații mici, oglindiri, zgomot, intensitate) | Reduce supra-învățarea? |

**În paralel — pregătirea localizării (necesară în Etapa 5):**
1. Subset **LIDC-IDRI** (~100–200 pacienți, pe loturi) sau LUNA16; măști 3D cu `pylidc` (consens ≥ 50% din radiologi).
2. **3D U-Net** (MONAI) sau **nnU-Net**: CT re-eșantionat la 1 mm, patch-uri 96³, precizie mixtă, batch 2.
3. Evaluare pe LIDC: **Dice** și sensibilitatea detecției.

**Raport:** pentru fiecare îmbunătățire — motivație, metodologie, rezultat, comparație; tabelul de ablație și
grafice (curbe ROC, AUC per variantă cu IÎ).

**Gata când:**
- [ ] cel puțin 3 îmbunătățiri măsurate față de referință, în tabelul de ablație, cu IÎ;
- [ ] cea mai bună configurație aleasă pe baza fold-urilor (nu a unui set de test privit de mai multe ori);
- [ ] U-Net antrenat pe LIDC, cu Dice raportat;
- [ ] tag `etapa-4`.

---

### Etapa 5 → Lab 6 (S8–S11, include vacanța) — 40 de puncte

**Obiectiv:** pipeline-ul complet în aplicație (fără adnotări manuale), îmbunătățiri noi, analiză statistică,
teaser video, slide-uri draft.

**Sarcini, în ordine:**
1. **Ajustarea fină a U-Net** pe cele 144 de măști NSCLC-Radiogenomics (+ casetele Lung-PET-CT-Dx ca
   supervizare slabă, opțional); Dice pe pacienți ținuți deoparte.
2. **Integrarea** (`TrainedPredictor`):
   - `localise`: U-Net pe CT-ul nativ (re-eșantionat intern la 1 mm, fereastră glisantă) → cea mai mare
     componentă conexă (sau cea cu SUV maxim) → centrul în mm → index CT;
   - `predict`: cubul 128³ în jurul acestui centru → clasificatorul → histologie, EGFR, KRAS;
   - utilizatorul poate în continuare corecta manual centrul.
3. **Noi îmbunătățiri (10 p):**
   - **validare externă** histologie: antrenare pe Lung-PET-CT-Dx → test pe NSCLC-Radiogenomics (și invers);
   - impactul localizării automate: cuburi din măștile prezise vs. din cele adnotate (cât scade AUC-ul?);
   - **Grad-CAM** (`explain.py`, ex. MONAI `GradCAM`) afișat în aplicație peste CT.
4. **Analiza statistică (10 p):** IÎ bootstrap pentru toate metricile; **test DeLong** între variante; curbe
   ROC; matrici de confuzie; eventual calibrare (curbă de fiabilitate); discuție despre puterea statistică
   (câți pozitivi avem). Tabelele generate automat din `evaluation/report.py`.
5. **Aplicația finală:** pipeline complet + Grad-CAM; textul „DEMO" dispare doar pentru ieșirile care vin
   acum din modele reale; mențiunea „instrument de cercetare, nu dispozitiv medical" rămâne.
6. **Teaser video (5 p), 1–2 min:** problema (biopsia invazivă) → ideea (biopsie virtuală) → demo rapid în
   aplicație → rezultatul principal → echipa. Înregistrare de ecran + voce/subtitrări.
7. **Slide-uri draft (10 p):** problemă, soluție, date, arhitectură, rezultate (tabel + ROC), demo, limitări.
8. **Raport:** analiza statistică, îmbunătățirile noi, validarea externă.
9. Tag `etapa-5`.

**Gata când:**
- [ ] aplicația rulează pipeline-ul complet pe un pacient nou, fără adnotări (5p);
- [ ] îmbunătățiri noi documentate: localizare automată, validare externă, Grad-CAM (10p);
- [ ] teaser video (5p);
- [ ] analiza statistică în raport (10p); slide-uri draft (10p);
- [ ] tag `etapa-5`.

---

### Etapa 6 → Lab 7 (S12–S13) — 20 de puncte: demo cu clienți + SWOT

**Sarcini:**
1. **Pitch** (5–7 min): problemă → soluție → demo → rezultate → impact → limitări. Repetat de cel puțin 2 ori.
2. **Scenariul demo-ului live** cu un pacient real pre-procesat + **pacient de rezervă** + **înregistrare video**
   a demo-ului, în caz că ceva eșuează.
3. **Analiza SWOT** în raport:
   - *Puncte tari*: multimodal (CT + PET + clinic), pipeline complet, evaluare riguroasă, explicabilitate;
   - *Puncte slabe*: date puține pentru mutații, fără ALK/ROS1/PD-L1, validare doar retrospectivă;
   - *Oportunități*: date de la spitale partenere, alte mutații, integrare PACS;
   - *Amenințări*: *domain shift* între scanere, reglementarea dispozitivelor medicale, încrederea medicilor.
   (Completați cu ce reiese din rezultatele voastre reale.)
4. **Concluziile** raportului, pe baza rezultatelor reale.
5. **Slide-urile finale**; curățarea repository-ului și a README-ului; tag `etapa-6`.

**Gata când:** pitch și demo repetate; SWOT și concluzii în raport; slide-uri finale; tag `etapa-6`.

---

### Calendar pe scurt

```
S1  │ S2   S3 │ S4   S5 │ S6   S7 │ S8   S9   S10  S11 │ S12  S13
E1  │ Etapa 2 │ Etapa 3 │ Etapa 4 │ Etapa 5 (vacanța) │ Etapa 6
    │ A, B, C │ PET reg.│ ablații │ integrare, stat., │ pitch,
    │ ~40 pac.│ date    │ + U-Net │ video, slide-uri  │ demo, SWOT
    │         │ complete│  LIDC   │                   │
```

---

## 7. Resurse de calcul

| Resursă | Folosire |
|---|---|
| GPU local (ex. RTX 3060, 6 GB) | Preprocesare, clasificatorul (cuburi 128³: batch 4–8 cu precizie mixtă), teste U-Net, aplicația |
| Google Colab / Kaggle Notebooks (GPU gratuit) | U-Net pe LIDC dacă memoria nu ajunge; rulări paralele ale fold-urilor |
| Disc | Descărcare pe loturi; se păstrează doar cuburile (~8 MB/pacient) |
| Server GPU al facultății | `TODO`: de întrebat coordonatorul |

**Pentru 6 GB VRAM:** precizie mixtă (`torch.autocast`), batch mic, *gradient accumulation*, patch-uri 96³ pentru U-Net.

---

## 8. Riscuri și planuri de rezervă

| Risc | Semnal | Plan B |
|---|---|---|
| Descărcarea e lentă / nu e spațiu | Un lot durează > 1 zi | Doar seriile necesare; Colab/Kaggle; disc extern |
| Registrarea CT–PET eșuează | Hotspot-ul PET nu cade pe tumoră | Rigid → afin; inițializare manuală; pacientul rămâne doar CT (raportat) |
| U-Net nu converge / nu încape în memorie | Dice < 0,5 pe LIDC | nnU-Net *low-res*; Colab; în aplicație localizare euristică + corecție manuală, menționat clar |
| AUC pentru mutații ≈ baseline clinic | Rezultatele Etapei 3 | Se raportează onest; histologia, ablațiile și pipeline-ul complet rămân contribuțiile principale |
| Modelul învață spitalul, nu histologia | Intern mare, extern mic | Evaluare separată pe sursă; normalizare per set; fără clasa „celule mici" la comparația între seturi |
| Rezultate prea bune ca să fie adevărate | AUC > 0,95 pe date puține | Căutați *leakage*: același pacient în două fold-uri, normalizare pe tot setul, eticheta în imagine |
| Demo-ul live eșuează | — | Pacient de rezervă pre-procesat + înregistrare video |

---

## 9. Comenzi utile

```bash
# Mediul (o singură dată)
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt -e .

# Teste
.venv\Scripts\pytest

# Aplicația (http://localhost:8501)
.venv\Scripts\streamlit run app/streamlit_app.py

# Ce ignoră git (verificare după ce ați completat .git/info/exclude)
git status --ignored

# Tag pentru o etapă
git tag etapa-2
git push origin etapa-2
```
