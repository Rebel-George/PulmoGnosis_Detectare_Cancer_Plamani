# 📅 Plan de proiect — PulmoGnosis

Planul de lucru pentru implementarea pipeline-ului descris în [README.md](README.md),
organizat după cele 6 etape de evaluare ale laboratorului.
Săptămânile sunt numerotate de la **S1 = 26 oct. – 1 nov. 2026**.

> **Ipoteze despre date (de verificat):** laboratoarele sunt la 2 săptămâni, cu Lab 2 în S1.
> Lab 6 ar cădea în vacanța de iarnă, așa că planul presupune că Lab 6 și Lab 7 se mută în ianuarie.
> Corectați coloana *Termen* dacă datele reale diferă.

> **Terminologie:** în acest plan, **„Etapa 1–6"** înseamnă etapele de evaluare de la laborator.
> Părțile pipeline-ului din README se numesc **„Etapa 0 / 1 / 2 a pipeline-ului"** (preprocesare /
> localizarea tumorii / clasificatorul).

---

## 📍 0. Stadiul curent (27 oct. 2026)

| Etapa | Stare | Ce mai rămâne |
|---|---|---|
| **1 → Lab 2** | 🟡 Aproape gata | *Related work* cu 3–4 lucrări reale; capturi de ecran în raport |
| **2 → Lab 3** | ⬜ Neînceput — **responsabil: `TODO`** | Totul (vezi secțiunea Etapa 2) |
| **3–6** | ⬜ Neînceput | — |

**Ce există deja în repository:**
- aplicația demo Streamlit (`app/`) cu flow-ul complet în 5 pași și un caz demonstrativ sintetic (fantomă CT + PET în DICOM);
- pachetul `src/pulmognosis/`: citire DICOM, conversii HU/SUV, alinierea PET pe CT la rezoluția nativă,
  extragerea cubului pentru model (**128³ voxeli de 0,75 mm**, `extract_model_patch`), interfața `Predictor` pentru modele;
- 19 teste automate (`pytest`);
- raportul cu secțiunile Etapei 1 (funcționalități, problema, instrumente).

**Ce NU există în repository (se obține local):** datele (`data/`), metadatele TCIA, mediul Python (`.venv`).

---

## 🎯 1. Etapele de evaluare și punctajul

| Etapa | Termen | Aplicație | Pct. | Raport | Pct. |
|---|---|---|---|---|---|
| **1** | **Lab 2** — S1 (26 oct. – 1 nov.) | Flow-ul de bază | 10 | Funcționalitățile aplicației; descrierea plastică și formală a problemei; *related work*, instrumente și tehnologii | 5 |
| **2** | **Lab 3** — S3 (9–15 nov.) | Algoritm inteligent + date puține (*small data*) | 5 | Algoritmul inteligent, metodologia experimentală, rezultate | 5 |
| **3** | **Lab 4** — S5 (23–29 nov.) | Algoritm inteligent + date reale / *benchmark* | 5 | Metodologie, rezultate, **comparație cu SOTA** (*state of the art* — cele mai bune rezultate publicate) | 5 |
| **4** | **Lab 5** — S7 (7–13 dec.) | Îmbunătățirea algoritmului | 10 | Îmbunătățirile: metodologie, rezultate, analize comparative | 10 |
| **5** | **Lab 6** — ~S11 (4–10 ian.) | Integrarea modelului în aplicație (5) + noi îmbunătățiri (10) + **teaser video** (5) | 20 | Analiza statistică a rezultatelor (10) + slide-uri, draft (10) | 20 |
| **6** | **Lab 7** — ~S13 (18–24 ian.) | **Demo cu clienți** = pitch + demo aplicație | 10 | Analiza SWOT și concluzii + slide-uri, versiune finală | 10 |
| | | **Total aplicație** | **60** | **Total raport** | **55** |

**Ce cântărește cel mai mult:** etapele 4 și 5 (50 din 115 puncte). Planul pune îmbunătățirile cu
impact mare (multi-task, PET, segmentarea automată, statistica) exact acolo.

### Cum se potrivește pipeline-ul pe etape

| Etapa | Ce parte din PulmoGnosis se predă |
|---|---|
| 1 | Aplicația demo cu flow-ul complet (încărcare → vizualizare → panou de rezultate), cu un model *placeholder* marcat clar |
| 2 | Primele modele pe date puține: CNN 2D pe Kaggle, regresie logistică clinică, primul model 3D pe un lot mic din NSCLC-Radiogenomics |
| 3 | Clasificatorul (Etapa 2 a pipeline-ului: histologie + EGFR/KRAS) pe seturile complete, cu baseline-uri și comparație cu literatura |
| 4 | Îmbunătățiri: pre-antrenare MedicalNet, PET + date clinice (fuziune), multi-task, augmentare — cu ablații |
| 5 | Localizarea automată (Etapa 1 a pipeline-ului, 3D U-Net) în aplicație → pipeline complet; validare externă; Grad-CAM; statistică (bootstrap, DeLong); video |
| 6 | Pitch + demo live, SWOT, concluzii |

---

## 👥 2. Roluri

| Rol | Responsabil | Zonă de cod | Ce livrează |
|---|---|---|---|
| **R1 — Date și preprocesare** | `TODO` | `src/pulmognosis/data/`, `src/pulmognosis/preprocessing/` | Tabelul de cohortă, descărcarea, preprocesarea (Etapa 0 a pipeline-ului), cuburile 3D, integrarea pipeline-ului |
| **R2 — Modele de clasificare și evaluare** | `TODO` | `src/pulmognosis/classification/`, `src/pulmognosis/evaluation/` | Baseline-uri, clasificatorul (Etapa 2 a pipeline-ului), validare încrucișată, ablații, statistică |
| **R3 — Aplicație și segmentare** | `TODO` | `app/`, `src/pulmognosis/segmentation/` | Aplicația demo (din S1), localizarea (Etapa 1 a pipeline-ului, 3D U-Net), Grad-CAM, video |

**Dacă echipa are 2 membri:** R1 preia segmentarea (din S5), R3 rămâne pe aplicație, Grad-CAM și video,
iar R2 rămâne pe modele.

**Comun:** fiecare scrie în raport secțiunile propriei zone; revizuirea codului colegilor; sincronizare
săptămânală de 30 de minute, plus o sesiune comună cu 2 zile înainte de fiecare laborator.

---

## 🗓️ 3. Calendar pe etape

```
        S1 │ S2   S3 │ S4   S5 │ S6   S7 │ S8  S9  S10  S11 │ S12  S13
R1  cohortă│ preproc │ Lung-Dx │ PET lipsă│ integrare pipeline│ suport
R2  clinic │ 2D + 3D │ complet │ multi-tk │ statistică+extern │ SWOT
R3  app v0 │ U-Net   │ app v1  │ U-Net ft │ app final + video │ demo
       E1 ▲      E2 ▲      E3 ▲      E4 ▲   (vacanță)   E5 ▲       E6 ▲
```

### Etapa 1 → Lab 2 (S1) — 15 puncte

| Sarcină | Stare |
|---|---|
| Structura repository-ului, `requirements.txt`, `pyproject.toml`, mediul Python 3.12 | ✅ |
| **Aplicația v0 (Streamlit):** încărcare DICOM → vizualizare CT + PET la rezoluția nativă → date clinice → localizare (euristică + corecție manuală) → panou de rezultate *placeholder*, marcat „DEMO – fără model antrenat" | ✅ |
| Preprocesare (Etapa 0 a pipeline-ului) cu teste unitare | ✅ |
| **Raport:** funcționalitățile aplicației, problema (plastic + formal), instrumente și tehnologii | ✅ |
| **Raport:** *related work* cu 3–4 lucrări reale (radiogenomică EGFR, histologie din CT/PET, radiomică, detecție pe LIDC) | ⬜ |
| **Raport:** capturi de ecran ale celor 5 pași ai aplicației | ⬜ |

*(Scriptul `build_cohort.py` a fost mutat în Etapa 2, unde e folosit prima dată.)*

**Gata când:**
- [x] aplicația rulează local cu un pacient de test, de la încărcare până la panoul de rezultate;
- [ ] raportul conține funcționalitățile, problema (plastic + formal), *related work* cu referințe reale, instrumentele.

### Etapa 2 → Lab 3 (S2–S3, 2–15 nov.) — 10 puncte

**Ideea:** trei modele reale antrenate pe date puține, iar unul dintre ele integrat în aplicație — prima
predicție reală afișată, alături de valorile *placeholder* marcate ca atare.

| # | Algoritm | Date („small data") | Prezice | De ce |
|---|---|---|---|---|
| **A** | **Regresie logistică** pe vârstă, gen, fumat | Tabelul clinic NSCLC-Radiogenomics, 211 pacienți — **fără imagini, poate începe din prima zi** | EGFR, KRAS | Baseline-ul pe care orice model ulterior trebuie să-l bată; primul model real **în aplicație** |
| **B** | **CNN 2D** (ResNet-18 pre-antrenat pe ImageNet, ajustat fin) | Kaggle, ~1000 de imagini 2D | Histologie (3 tipuri + normal) | Model rapid pe imagini; limitările setului (fără ID de pacient, risc de *leakage*) se documentează explicit |
| **C** | **3D ResNet-18** (greutăți MedicalNet) pe cubul CT+PET de 128³ la 0,75 mm | **~40 de pacienți** NSCLC-Radiogenomics, cubul centrat pe masca tumorală oficială (SEG) | Adenocarcinom vs. scuamos | Prima versiune a clasificatorului (Etapa 2 a pipeline-ului) |

**Datele disponibile (numărate din metadate):** 144 de pacienți au CT + PET + mască tumorală
(112 adenocarcinom, 29 scuamos; 23 EGFR mutant din 117 cunoscuți). Lotul de ~40 se alege stratificat
după histologie. **Disc:** ~450 MB/pacient (CT + PET) ⇒ ~18 GB pentru 40 de pacienți, față de ~29 GB liberi
⇒ descărcare în **loturi de câte 10**, se păstrează doar cuburile (~8 MB/pacient, `float16`), iar DICOM-urile
brute se șterg după procesare.

| Zile | R1 — Date | R2 — Modele și evaluare | R3 — Aplicație |
|---|---|---|---|
| **1–2** | `scripts/build_cohort.py`: tabel pacient × (CT, PET, SEG, histologie, EGFR, KRAS, vârstă, gen, fumat); alegerea celor ~40 de pacienți | **Modelul A** + cadrul comun de evaluare: validare încrucișată în 5 fold-uri, stratificată la nivel de pacient (generată o singură dată), AUC cu interval de încredere bootstrap | Instalarea PyTorch (CUDA) și MONAI; descărcarea setului Kaggle |
| **3–5** | Descărcare selectivă cu `idc-index` (loturi de 10); citirea măștii **DICOM SEG**; centrul tumorii → cub 128³ → salvat ca `.npz` în `data/processed/` | Script de evaluare comun, folosit de toate modelele (aceleași fold-uri, aceleași metrici) | **Modelul B** (CNN 2D pe Kaggle) |
| **6–8** | Notebook de verificare vizuală: cubul conține tumora la fiecare pacient | **Modelul C**: întâi doar CT, apoi CT+PET; precizie mixtă, batch 4 | **Modelul A în aplicație**: EGFR/KRAS reale din datele clinice, etichetate „model clinic, antrenat pe 211 pacienți"; histologia rămâne *placeholder* marcat |
| **9–10** | Raport: subsetul de date și preprocesarea | Raport: tabelul de rezultate (cu IÎ 95%, nr. pacienți și pozitivi) și discuția | Raport: descrierea algoritmilor A, B, C + capturi de ecran; *tag* git `etapa-2` |

**În paralel, dacă rămâne timp (pregătire pentru Etapa 4):** subset LIDC-IDRI (~100 pacienți), măști 3D cu
`pylidc` și un prim U-Net de test pentru a verifica memoria GPU.

**De pregătit / aprobat înainte de start:**
- metadatele TCIA (3 fișiere mici, nu sunt în repository), în `data/metadata/`:
  - <https://www.cancerimagingarchive.net/wp-content/uploads/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv> (date clinice și etichete)
  - <https://www.cancerimagingarchive.net/wp-content/uploads/NSCLC_Radiogenomics-6-1-21-Version-4-nbia-digest.xlsx> (inventarul seriilor)
  - <https://www.cancerimagingarchive.net/wp-content/uploads/statistics-clinical-20201221.xlsx> (Lung-PET-CT-Dx, pentru etapele următoare)
- descărcări: ~18 GB de scanări (pe loturi), greutățile MedicalNet (~100–200 MB), PyTorch cu CUDA (~2,5 GB);
- un cont Kaggle (sau descărcarea manuală a setului) pentru modelul B;
- spațiul liber pe disc al celui care rulează (cifrele de mai sus presupun ~29 GB).

**Unde intră codul:** `src/pulmognosis/data/` (cohortă, descărcare, SEG), `src/pulmognosis/evaluation/`
(fold-uri, metrici, bootstrap), `src/pulmognosis/classification/` (modelele A, B, C), `scripts/` (puncte de
intrare), `tests/`. Modelul A intră în aplicație printr-o clasă nouă care respectă interfața `Predictor`
din `src/pulmognosis/inference.py` — aplicația nu trebuie restructurată.

**Așteptări realiste:** cu ~40 de pacienți, modelul C va avea probabil performanță modestă și intervale de
încredere largi — normal pentru „small data". Comparația importantă este dacă modelul pe imagini bate
baseline-ul clinic; Etapa 3 (toți cei 144+ pacienți) și Etapa 4 (îmbunătățiri) pornesc de aici.

**Gata când:**
- [ ] 3 modele antrenate, cu rezultate reale (IÎ 95%, numărul de pacienți și de cazuri pozitive);
- [ ] aplicația afișează predicția reală a modelului A, alături de valorile *placeholder* marcate;
- [ ] raportul descrie algoritmii, metodologia (date, fold-uri, metrici) și rezultatele;
- [ ] versiunea prezentată e înghețată cu *tag*-ul `etapa-2`.

### Etapa 3 → Lab 4 (S4–S5) — 10 puncte

| Săpt. | R1 | R2 | R3 |
|---|---|---|---|
| **S4** | Restul NSCLC-Radiogenomics (pe loturi, păstrând doar cuburile) | Clasificatorul pe NSCLC-Radiogenomics complet: histologie + EGFR + KRAS (pierdere mascată) | Antrenarea U-Net pe LIDC (precizie mixtă, patch-uri 96³) |
| **S5** | Descărcare și preprocesare **Lung-PET-CT-Dx** (casete → cuburi) | **Baseline radiomic** (PyRadiomics + regresie logistică). Tabelul de comparație cu SOTA (AUC-urile publicate în lucrările din *related work*) | **Aplicația v1:** modelul real înlocuiește *placeholder*-ul (cu cuburi centrate pe adnotări) |

**Raport:** metodologia pe date reale, rezultate cu intervale de încredere, comparație cu SOTA.

**Gata când:**
- [ ] rezultate pe seturile complete, cu intervale de încredere;
- [ ] tabel de comparație: modelul nostru vs. baseline-uri vs. lucrări publicate.

### Etapa 4 → Lab 5 (S6–S7) — 20 de puncte

Fiecare îmbunătățire se măsoară față de versiunea din Etapa 3 (**ablație** — se adaugă sau se elimină câte o componentă).

| Săpt. | R1 | R2 | R3 |
|---|---|---|---|
| **S6** | PET lipsă (canal zero + *modality dropout*), augmentări 3D | **Îmbunătățiri:** greutăți pre-antrenate MedicalNet; CT → CT+PET; + date clinice (fuziune târzie) | Evaluarea U-Net pe LIDC (Dice, sensibilitate) |
| **S7** | Histologie pe ambele seturi (~560 pacienți) | **Multi-task** (histologie ajută mutațiile); *focal loss*; tabelul de ablații | **Ajustare fină** a U-Net pe măștile NSCLC-Radiogenomics + casetele Lung-PET-CT-Dx |

**Raport:** fiecare îmbunătățire cu motivație, metodologie, rezultat și comparație (tabel de ablație + grafice).

**Gata când:**
- [ ] cel puțin 3 îmbunătățiri măsurate față de versiunea de bază, cu tabel comparativ.

### Etapa 5 → Lab 6 (S8–S11, include vacanța de iarnă) — 40 de puncte

| Săpt. | R1 | R2 | R3 |
|---|---|---|---|
| **S8** | **Integrare:** script pacient nou DICOM → preprocesare → localizare (U-Net) → cub → clasificator | **Noi îmbunătățiri:** **validare externă** (antrenare Lung-PET-CT-Dx → test NSCLC-Radiogenomics); cuburi din măștile prezise vs. adnotate | Grad-CAM pe clasificator |
| **S9** | Testarea pipeline-ului pe pacienți ținuți deoparte | **Analiza statistică:** bootstrap, test DeLong între variante, curbe ROC, matrici de confuzie | **Aplicația finală:** pipeline complet integrat (fără adnotări manuale) + Grad-CAM |
| **S10** | *(vacanță — doar rezervă)* | *(vacanță — doar rezervă)* | *(vacanță — doar rezervă)* |
| **S11** | Curățarea codului | Secțiunea de analiză statistică din raport | **Teaser video** (1–2 min: problema, soluția, demo); **slide-uri draft** (toată echipa) |

**Gata când:**
- [ ] aplicația rulează pipeline-ul complet pe un pacient nou (5p);
- [ ] noi îmbunătățiri documentate: segmentarea automată, validarea externă, Grad-CAM (10p);
- [ ] teaser video (5p);
- [ ] analiza statistică în raport (10p) și slide-urile draft (10p).

### Etapa 6 → Lab 7 (S12–S13) — 20 de puncte

| Toată echipa |
|---|
| **Pitch** (problemă → soluție → rezultate → impact) și scenariul demo-ului live, repetate de cel puțin 2 ori. Un pacient de rezervă pre-procesat, în caz că încărcarea live eșuează |
| **Analiza SWOT** (*Strengths, Weaknesses, Opportunities, Threats* — puncte tari, puncte slabe, oportunități, amenințări) și concluziile raportului |
| Slide-urile, versiunea finală |

---

## ⚙️ 4. Reguli de lucru în echipă

- **Git:** câte o ramură pentru fiecare sarcină (ex. `feature/hu-conversion`), *pull request* revizuit de un coleg înainte de unire în `main`.
- **Date:** nu se pun niciodată în git (`data/`, greutăți, rezultate, `.venv`). Repository-ul nu are `.gitignore`; fiecare membru pune regulile de ignorare local, în `.git/info/exclude` (`/data/`, `/checkpoints/`, `/runs/`, `.venv/`, `__pycache__/`, `*.egg-info/`, `*.dcm`, `*.nii.gz`, `*.pt`, `*.pth`). Fiecare membru descarcă datele local cu aceleași scripturi.
- **Limbă:** codul, comentariile și mesajele de commit în engleză; README, PLAN și raportul în română.
- **Reproductibilitate:** fiecare experiment are un fișier de configurare în `configs/`, *seed* fix și rezultatele salvate împreună cu configurația. Împărțirea în fold-uri se generează **o singură dată** și e folosită de toți.
- **Urmărirea experimentelor:** un proiect comun Weights & Biases (sau MLflow), ca toată echipa să vadă toate rulările.
- **Rezultate în raport:** doar cifre care provin din rulări reale, cu link către rularea respectivă.
- **Înainte de fiecare laborator:** se îngheață o versiune (*tag* git, ex. `etapa-2`), ca demo-ul să nu fie stricat de modificări de ultim moment.

---

## 💻 5. Resurse de calcul

| Resursă | Folosire |
|---|---|
| Laptop cu RTX 3060 (6 GB VRAM), 16 GB RAM | Preprocesare, Etapa 2 (cuburi 128³: batch 4–8 cu precizie mixtă), teste U-Net, aplicația |
| Google Colab / Kaggle Notebooks (GPU gratuit) | Antrenarea U-Net pe LIDC dacă 6 GB nu ajung; rulări paralele ale fold-urilor |
| Spațiu pe disc | ~29 GB liberi: se descarcă pe loturi, se păstrează doar cuburile procesate (~8 MB/pacient) |
| Server GPU al facultății | `TODO`: de întrebat coordonatorul dacă există |

**Pentru 6 GB VRAM:** precizie mixtă (`torch.autocast`), batch mic (2–4) pentru U-Net, patch-uri de 96³
în loc de volume întregi, *gradient accumulation* dacă e nevoie.

---

## ⚠️ 6. Riscuri și planuri de rezervă

| Risc | Semnal de alarmă | Plan B |
|---|---|---|
| Etapa 1 e chiar săptămâna aceasta | Aplicația nu e gata cu 2 zile înainte | Aplicația v0 poate folosi un pacient de exemplu pre-încărcat; vizualizarea și panoul contează mai mult decât încărcarea generală |
| Descărcarea e prea lentă sau nu e spațiu | Lotul 1 durează > 1 zi | Doar seriile strict necesare; Colab/Kaggle cu datele în cloud; disc extern |
| Co-registrarea CT–PET e greșită | Hotspot-ul PET nu cade pe tumoră în verificarea vizuală | Seria PET-CT asociată (același spațiu) sau doar CT pentru pacienții problematici |
| U-Net nu converge sau nu încape în memorie | Dice < 0,5 după S7 | nnU-Net în varianta *low-res*, antrenare pe Colab, sau patch-uri din adnotări în aplicație cu mențiune clară |
| AUC pentru mutații ≈ baseline clinic | Rezultatele Etapei 3 | Se raportează onest; histologia, îmbunătățirile măsurate și pipeline-ul complet rămân contribuțiile principale |
| Modelul învață spitalul, nu histologia | Performanță mare intern, mică la validarea externă | Evaluare separată pe sursă, normalizare de intensitate per set, eliminarea clasei „celule mici" din comparația între seturi |
| Demo-ul live eșuează la Lab 7 | — | Pacient pre-procesat de rezervă și înregistrare video a demo-ului |
| Un membru rămâne în urmă | Sarcina săptămânii nu e gata la sincronizare | Vacanța (S10) și S12 sunt rezerve; interfețe clare între module, ca sarcinile să poată fi preluate |
