# Predare — cum continui PulmoGnosis

Acest fișier este punctul de plecare pentru cine preia proiectul. Spune ce există acum, cum pornești,
ce am învățat pe parcurs și ce urmează. Detaliile sunt în [PLAN.md](PLAN.md) (plan complet, reguli,
procedura pe server), [README.md](README.md) (arhitectura) și în raportul tehnic
([PDF](raport_biopsie_virtuala.pdf), sursa [`raport_biopsie_virtuala.tex`](raport_biopsie_virtuala.tex)).

Stare la 26.11.2026: etapele de laborator 1–4 prezentate (etapele 3 și 4 parțial); aplicația rulează
cap-coadă pe pacienți reali, cu modele antrenate.

---

## 1. Ce există acum

| Ce | Unde | Stare |
|---|---|---|
| Aplicația (5 pași: încărcare → vizualizare → date clinice → localizare → rezultate) | `app/streamlit_app.py` | funcționează pe date reale |
| **Modelul A** — regresie logistică pe vârstă, gen, fumat → EGFR, KRAS | `models/clinical_lr.json` | EGFR AUC 0,72 (0,62–0,81); KRAS 0,55 (0,45–0,65) |
| **Modelul B** — ResNet-18 2D pe setul Kaggle (4 clase) | doar pe server | acuratețe 0,90 pe imaginile de test fără copii; doar comparație |
| **Modelul C** — 3D ResNet-18 MedicalNet fix + regresie logistică → adenocarcinom vs. scuamos | `models/histology_probe.json` | AUC 0,64 (0,53–0,74); EGFR/KRAS din imagine ≈ 0,5 |
| Preprocesare (HU, SUV, cuburi 128³ de 0,75 mm) + 39 de teste | `src/pulmognosis/`, `tests/` | toate testele trec |
| Notebook-ul de pe server (descărcare, cuburi, antrenare, tabel) | `notebooks/etapa2_jupyterhub.ipynb` | rulat cap-coadă |
| Raportul tehnic, organizat pe etapele de laborator 2, 3, 4 | `raport_biopsie_virtuala.tex` / `.pdf` | 17 pagini, se compilează curat |

Toate AUC-urile vin din validare încrucișată în 5 fold-uri la nivel de pacient (aceleași fold-uri
pentru toate modelele, în `configs/splits/`), cu interval de încredere 95% prin bootstrap.

**Ramura de lucru este `feature/etapa-2`** (a fost îmbinată și în `main`). Tag-urile `etapa-2`,
`etapa-3`, `etapa-4` marchează versiunile prezentate la laborator.

---

## 2. Pornire rapidă (laptop, Windows)

Necesită Python 3.12.

```powershell
git clone https://github.com/Rebel-George/PulmoGnosis_Detectare_Cancer_Plamani.git
cd PulmoGnosis_Detectare_Cancer_Plamani
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt -e .
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m streamlit run app/streamlit_app.py
```

`requirements.txt` include PyTorch (CPU ajunge pentru aplicație). La prima predicție de histologie,
MONAI descarcă automat greutățile MedicalNet (este nevoie de internet).

### Cazurile de test reale

Doi pacienți NSCLC-Radiogenomics au fost aleși aleator și **scoși din antrenarea finală** a
modelelor A și C, ca aplicația să fie testată pe pacienți nevăzuți: **R01-080** (adenocarcinom) și
**R01-007** (carcinom scuamos). Datele lor (CT + masca radiologului, ~320 MB, publice) nu sunt în
git; le descarci cu:

```powershell
.\.venv\Scripts\python -m pip install idc-index
.\.venv\Scripts\python scripts/download_demo_cases.py
```

Apoi, în aplicație: pasul 1 → „Folder local cu fișiere DICOM" → calea completă către
`data\demo\R01-080`. Aplicația găsește masca (SEG) și pornește pasul 4 din centrul tumorii, exact ca
la antrenare. La pasul 3 introduci vârsta, genul și fumatul pacientului (din tabelul de cohortă,
`data/cohort/nsclc_radiogenomics.csv`, construit de `scripts/build_cohort.py` din metadatele TCIA).
Pentru R01-080, modelul C dă ~97% adenocarcinom.

Cazul demonstrativ (fantoma sintetică) arată doar că aplicația funcționează: fantoma nu seamănă cu o
tumoră reală, deci scorul ei de histologie (100% adenocarcinom) nu înseamnă nimic.

### Raportul PDF

Se compilează cu LuaLaTeX (folosește `fontspec` și fontul Noto Sans). Pe Windows: MiKTeX
(`winget install MiKTeX.MiKTeX`), cu instalarea automată a pachetelor activată și pachetul `noto`.
Din rădăcina repository-ului, de două ori (a doua oară pentru referințe):

```powershell
lualatex -interaction=nonstopmode raport_biopsie_virtuala.tex
```

---

## 3. Serverul (JupyterHub-ul facultății)

Antrenarea rulează pe https://www.cs.ubbcluj.ro/apps/mlhub/ (GPU A100), **fiecare în contul
propriu**: datele și rezultatele de pe contul inițial nu sunt accesibile altcuiva. Procedura completă
este în [PLAN.md, secțiunea 10](PLAN.md); pe scurt:

1. Terminal pe server: `git clone` (ramura de lucru), apoi deschizi `notebooks/etapa2_jupyterhub.ipynb`.
2. **Kernel-ul trebuie să fie `venv`**, creat de primele celule; kernel-ul implicit (Python-ul
   sistemului) nu permite instalarea pachetelor.
3. Rulezi celulele în ordine: descărcarea NSCLC-Radiogenomics (~68 GB, ~15 min), cuburile (~50 min),
   modelul A, modelul C (sonda liniară, ~1 min), modelul B (are nevoie de o cheie API Kaggle salvată de
   tine în `~/.kaggle/access_token`, niciodată în repository), tabelul de rezultate.
4. Codul se schimbă **doar pe laptop**: laptop → GitHub → `git pull` pe server. Înainte de `git pull`,
   notebook-ul de pe server (care are rezultate salvate în el) trebuie pus deoparte cu
   `git stash push -- notebooks/etapa2_jupyterhub.ipynb`.
5. Discul `/bigdata` este **comun tuturor studenților** (~240 GB liberi, fără cotă personală):
   descarci fiecare set o singură dată și ștergi ce nu mai folosești.

Exportul modelului C pentru aplicație se face pe server (are nevoie de trăsăturile celor 141 de
pacienți): `python scripts/probe_model3d.py --cubes <cuburi> --out runs/model_c_probe --export <fișier>
--exclude R01-080,R01-007`. Modelul A se reantrenează pe laptop: `python scripts/train_clinical.py
--exclude R01-080,R01-007`.

---

## 4. Ce am învățat (capcane deja rezolvate)

- **PET-ul nu e aliniat cu CT-ul.** În NSCLC-Radiogenomics, masca tumorii este desenată pe un CT de
  diagnostic, iar PET-ul este într-un alt studiu, cu alt sistem de coordonate. Fără o registrare
  CT–CT, canalul PET nu poate fi folosit; de aceea toate modelele actuale folosesc doar CT.
- **IDC conține și măști generate de AI** („AIMI …", `analysis_result_id` nenul). Selecția păstrează
  doar cele 144 de măști originale ale radiologilor.
- **Ajustarea fină a rețelei 3D pe ~140 de pacienți nu învață** (AUC 0,48): *early stopping* pe ~17
  pacienți de validare oprește antrenarea înainte să înceapă învățarea. Un test de supra-ajustare a
  arătat că implementarea e corectă. Soluția folosită: trăsături fixe MedicalNet + regresie logistică.
- **Regularizarea aleasă separat pe fiecare fold strică AUC-ul cumulat** (probabilități la scări
  diferite): C este fixat dinainte la 1,0, ca la modelul A.
- **Pragul de decizie trebuie să fie 0,5** când clasele sunt ponderate în pierdere (prevalența ca prag
  clasifica aproape totul drept scuamos).
- **Setul Kaggle are 16% din imaginile de test copiate din antrenare** (49 din 54 de imagini „normal"):
  rezultatele modelului B se raportează pe imaginile fără copii (`scripts/check_kaggle_leakage.py`).
- **Ablația modelului C:** pre-antrenarea MedicalNet și normalizarea intrării nu contează; contextul din
  jurul tumorii contează (decupajul de 48 mm coboară AUC-ul la nivelul întâmplării).
- **Varianta principală a fost aleasă înainte de rezultate**; restul variantelor sunt raportate ca
  exploratorii. Păstrează regula: nu alege varianta după ce vezi scorurile.

---

## 5. Ce urmează (în ordine)

1. **Registrarea PET pe CT** (SimpleITK, rigid, Mattes MI), verificată vizual pe toți pacienții; apoi
   cuburi CT + PET și modelul C pe două canale.
2. **Lung-PET-CT-Dx** (355 de pacienți cu histologie): mai multe date pentru histologie și validare
   externă (antrenare pe un set, test pe celălalt). Atenție: carcinomul cu celule mici există doar
   aici, deci un model poate învăța spitalul în loc de clasă.
3. **Baseline-uri lipsă:** radiomică (PyRadiomics pe masca SEG + regresie logistică) și un model
   clinic pentru histologie, pe aceiași pacienți și aceleași fold-uri.
4. **Fuziunea A + C** (imagine + date clinice) și, ulterior, antrenarea multi-task.
5. **Localizarea automată:** U-Net 3D antrenat pe LIDC-IDRI, ca aplicația să nu mai depindă de masca
   radiologului.
6. **Grad-CAM** în aplicație, **testul DeLong** între variante, apoi etapele 5–6 (video, slide-uri,
   SWOT, concluzii). Bifele pe etape sunt în PLAN.md.

Regulile de lucru (împărțire pe pacienți, statistici doar pe fold-ul de antrenare, intervale de
încredere, ablații, seed fix) sunt în [PLAN.md, secțiunea 4](PLAN.md). Numerele din raport trebuie
să vină numai din rulări reale din acest repository.
