# 📅 Plan de proiect — PulmoGnosis

Planul de lucru pentru implementarea pipeline-ului descris în [README.md](README.md),
organizat după cele 6 etape de evaluare ale laboratorului.
Săptămânile sunt numerotate de la **S1 = 26 oct. – 1 nov. 2026**.

> **Ipoteze despre date (de verificat):** laboratoarele sunt la 2 săptămâni, cu Lab 2 în S1.
> Lab 6 ar cădea în vacanța de iarnă, așa că planul presupune că Lab 6 și Lab 7 se mută în ianuarie.
> Corectați coloana *Termen* dacă datele reale diferă.

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
| 3 | Etapa 2 (histologie + EGFR/KRAS) pe seturile complete, cu baseline-uri și comparație cu literatura |
| 4 | Îmbunătățiri: pre-antrenare MedicalNet, PET + date clinice (fuziune), multi-task, augmentare — cu ablații |
| 5 | Etapa 1 (3D U-Net) în aplicație → pipeline complet; validare externă; Grad-CAM; statistică (bootstrap, DeLong); video |
| 6 | Pitch + demo live, SWOT, concluzii |

---

## 👥 2. Roluri

| Rol | Responsabil | Zonă de cod | Ce livrează |
|---|---|---|---|
| **R1 — Date și preprocesare** | `TODO` | `src/pulmognosis/data/`, `src/pulmognosis/preprocessing/` | Tabelul de cohortă, descărcarea, Etapa 0, patch-urile 3D, integrarea pipeline-ului |
| **R2 — Modele de clasificare și evaluare** | `TODO` | `src/pulmognosis/classification/`, `src/pulmognosis/evaluation/` | Baseline-uri, Etapa 2, validare încrucișată, ablații, statistică |
| **R3 — Aplicație și segmentare** | `TODO` | `app/`, `src/pulmognosis/segmentation/` | Aplicația demo (din S1), Etapa 1 (3D U-Net), Grad-CAM, video |

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

### Etapa 1 → Lab 2 (S1, **săptămâna aceasta**) — 15 puncte

| R1 | R2 | R3 |
|---|---|---|
| Structura repository-ului, `requirements.txt`, mediul Python. Script `build_cohort.py` din metadatele deja descărcate | **Raport:** *related work* cu 3–4 lucrări reale (radiogenomică EGFR, histologie din CT/PET, radiomică, detecție pe LIDC) + secțiunea de instrumente și tehnologii | **Aplicația v0 (Streamlit):** încărcare DICOM (CT + PET opțional) → vizualizare secțiuni CT cu PET suprapus → formular clinic → panou de rezultate generat de un model *placeholder*, marcat vizibil „DEMO – fără model antrenat" |

**Toată echipa:** secțiunea *Funcționalitățile aplicației* din raport (cu capturi de ecran din aplicația v0);
recitirea descrierii plastice și formale (există deja).

**Gata când:**
- [ ] aplicația rulează local cu un pacient de test, de la încărcare până la panoul de rezultate;
- [ ] raportul conține funcționalitățile, problema (plastic + formal), *related work* cu referințe reale, instrumentele.

### Etapa 2 → Lab 3 (S2–S3) — 10 puncte

| Săpt. | R1 | R2 | R3 |
|---|---|---|---|
| **S2** | Descărcarea lotului 1 NSCLC-Radiogenomics (~30 pacienți: CT + PET + SEG). Conversie HU/SUV cu teste unitare | **Regresie logistică clinică** (vârstă, gen, fumat → EGFR/KRAS) pe toți cei 211 pacienți — nu are nevoie de imagini. Cadrul de evaluare: fold-uri stratificate la nivel de pacient, AUC + bootstrap | **CNN 2D pe Kaggle** (histologie, *small data*), cu limitările documentate. Descărcarea unui subset LIDC (~100 pacienți) |
| **S3** | Resampling, co-registrare CT–PET, decupare patch 64³ pentru lotul 1. Notebook de verificare vizuală | **Primul model 3D** (3D ResNet-18, doar CT, histologie) pe lotul mic → rezultate „small data" | Măști 3D din contururile LIDC (`pylidc`); primul U-Net de test (verifică memoria GPU) |

**Raport:** descrierea algoritmului (CNN 2D, ResNet 3D, regresie logistică), metodologia, rezultatele pe date puține.

**Gata când:**
- [ ] cel puțin un algoritm inteligent antrenat și evaluat, cu rezultate reale pe date puține;
- [ ] raportul descrie algoritmul, metodologia și rezultatele.

### Etapa 3 → Lab 4 (S4–S5) — 10 puncte

| Săpt. | R1 | R2 | R3 |
|---|---|---|---|
| **S4** | Restul NSCLC-Radiogenomics (pe loturi, păstrând doar patch-urile) | Etapa 2 pe NSCLC-Radiogenomics complet: histologie + EGFR + KRAS (pierdere mascată) | Antrenarea U-Net pe LIDC (precizie mixtă, patch-uri 96³) |
| **S5** | Descărcare și preprocesare **Lung-PET-CT-Dx** (casete → patch-uri) | **Baseline radiomic** (PyRadiomics + regresie logistică). Tabelul de comparație cu SOTA (AUC-urile publicate în lucrările din *related work*) | **Aplicația v1:** modelul real înlocuiește *placeholder*-ul (cu patch-uri din adnotări) |

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
| **S8** | **Integrare:** script pacient nou DICOM → Etapa 0 → Etapa 1 → patch → Etapa 2 | **Noi îmbunătățiri:** **validare externă** (antrenare Lung-PET-CT-Dx → test NSCLC-Radiogenomics); patch-uri din măștile prezise vs. adnotate | Grad-CAM pe modelul Etapei 2 |
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
- **Date:** nu se pun niciodată în git (`data/`, greutăți, rezultate). Fiecare membru descarcă local cu aceleași scripturi.
- **Reproductibilitate:** fiecare experiment are un fișier de configurare în `configs/`, *seed* fix și rezultatele salvate împreună cu configurația. Împărțirea în fold-uri se generează **o singură dată** și e folosită de toți.
- **Urmărirea experimentelor:** un proiect comun Weights & Biases (sau MLflow), ca toată echipa să vadă toate rulările.
- **Rezultate în raport:** doar cifre care provin din rulări reale, cu link către rularea respectivă.
- **Înainte de fiecare laborator:** se îngheață o versiune (*tag* git, ex. `etapa-2`), ca demo-ul să nu fie stricat de modificări de ultim moment.

---

## 💻 5. Resurse de calcul

| Resursă | Folosire |
|---|---|
| Laptop cu RTX 3060 (6 GB VRAM), 16 GB RAM | Preprocesare, Etapa 2 (patch-uri 64³ încap ușor), teste U-Net, aplicația |
| Google Colab / Kaggle Notebooks (GPU gratuit) | Antrenarea U-Net pe LIDC dacă 6 GB nu ajung; rulări paralele ale fold-urilor |
| Spațiu pe disc | ~29 GB liberi: se descarcă pe loturi, se păstrează doar patch-urile procesate (< 5 GB în total) |
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
