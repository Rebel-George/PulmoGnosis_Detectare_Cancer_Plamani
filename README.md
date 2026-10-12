# 🫁 Virtual Biopsy: CDSS pentru Cancerul Pulmonar (NSCLC)

## 📄 Rezumat Executiv
Proiectul vizează dezvoltarea unui sistem inteligent de tip **CDSS (Clinical Decision Support System)** pentru diagnosticarea avansată a cancerului pulmonar (Non-Small Cell Lung Cancer - NSCLC). Scopul aplicației este de a prezice profilul molecular al tumorii (biomarkeri) direct din imagistica medicală non-invazivă, scutind pacienții de biopsii fizice riscante și accelerând decizia terapeutică.

Aplicația folosește o arhitectură AI multimodală de tip *Late-Fusion* care analizează simultan scanări CT (structură), scanări PET (metabolism) și date demografice (Vârstă, Gen) pentru a genera probabilități de tratament țintit.

---

## 🏗️ 1. Arhitectura Sistemului: Date de Intrare și Ieșire

Pentru ca modelul să facă o predicție pe un pacient nou în spital, aplicația necesită următoarele intrări și generează următoarele ieșiri:

### Date de Intrare (Inputs)
Aplicația preia trei tipuri de date în momentul evaluării pacientului:
1. **Scanarea CT (Computed Tomography) 3D:** Oferă rezoluție anatomică înaltă, arătând forma, marginile (spiculațiile) și densitatea fizică a nodulului pulmonar.
2. **Scanarea PET (Positron Emission Tomography) 3D:** Arată activitatea metabolică. Deoarece celulele canceroase consumă glucoză rapid, PET-ul evidențiază "hotspot-urile" de creștere agresivă.
3. **Date Tabulare (Clinice):** Vârsta (normalizată) și Genul (one-hot encoded), extrase automat din dosarul electronic al pacientului (EHR).

> *Notă: Modelul aliniază automat (co-registrează) scanarea CT cu cea PET, astfel încât pixelii anatomici să corespundă perfect cu cei metabolici.*

### Date de Ieșire (Outputs)
Modelul nu oferă un diagnostic binar ("Cancer: Da/Nu"), ci un panou multi-label bazat pe activarea `Sigmoid`. Ieșirea este un vector de probabilități (0-100%) pentru prezența următoarelor ținte terapeutice:

| Biomarker Prezis | Semnificație Clinică (Ce indică) | Tratament Asociat |
| :--- | :--- | :--- |
| **EGFR** | Mutație genetică structurală (frecventă la nefumători). | Terapie Țintită (TKI) |
| **KRAS** | Mutație de semnalizare celulară (frecventă la fumători). | Terapie Țintită (Inhibitori KRAS) |
| **ALK** | Fuziune genetică (frecventă la tineri). | Terapie Țintită (Inhibitori ALK) |
| **ROS1** | Rearanjare genetică rară. | Terapie Țintită |
| **PD-L1** | Expresie proteică de evaziune imunitară (Scutul tumorii). | Imunoterapie (Inhibitori Checkpoint) |

---

## 🖥️ 2. Fluxul de Lucru și Funcționalitățile Aplicației UI

Interfața medicului este concepută pentru a genera încredere (*trust*) și transparență. Fluxul aplicației arată astfel:

### Pasul 1: Încărcarea și Procesarea (Ingestia)
Medicul selectează pacientul din sistemul PACS. Aplicația extrage automat fișierele DICOM pentru CT și PET, convertește pixelii în Unități Hounsfield și rulează un model secundar (ex. 3D U-Net) pentru a decupa doar bounding-box-ul nodulului pulmonar.

### Pasul 2: Vizualizarea 3D Suprapusă (CT + PET Fusion)
Înainte de a afișa rezultatele AI-ului, aplicația prezintă medicului o randare 3D a tumorii decupate. Aici, structura gri, anatomică a plămânului (din CT) este **suprapusă cu o hartă de culori termale (din PET)**. Această vizualizare fuzionată permite medicului să confirme vizual că AI-ul a identificat corect centrul de greutate metabolic al tumorii și limitele ei fizice.


### Pasul 3: Tabloul de Bord Predictiv (Rezultatele)
Sub vizualizarea 3D, interfața afișează panoul de ieșire generat de rețeaua neurală (Late-Fusion). Medicul va vedea un card clar:
*   🟢 **EGFR Mutation:** 85% (Înaltă)
*   🔴 **KRAS Mutation:** 12% (Scăzută)
*   🟢 **PD-L1 Expression >50%:** 78% (Înaltă)
*   📝 *Mesaj de sistem:* "Profil radiomic sugestiv pentru mutație EGFR pozitivă și expresie PD-L1 ridicată. Se recomandă prioritizarea terapiilor țintite."

---

## 🧠 3. Descrierea Problemei Rezolvate (Cum funcționează)

**Problema:** Determinarea profilului molecular al unei tumori necesită în prezent o biopsie fizică invazivă. Aceasta implică riscuri majore (ex. colaps pulmonar), durează săptămâni pentru secvențierea ADN-ului și poate rata mutația dacă acul extrage țesut dintr-o zonă nereprezentativă a tumorii.

**Soluția AI (Lupa Matematică):** 
Rezolvăm această problemă traducând fenotipurile macroscopice în genotipuri microscopice. O tumoare condusă de mutația EGFR își construiește rețeaua de vase de sânge și își organizează celulele diferit față de o tumoare condusă de KRAS. 

Ochiul uman nu poate detecta aceste diferențe texturale pe o secțiune CT. Rețeaua noastră 3D CNN acționează ca o lupă matematică: scanează sute de mii de pixeli și extrage modele biometrice de formă, densitate și captare a glucozei (PET). Prin fuzionarea (Late Fusion) acestor texturi matematice de înaltă dimensionalitate cu contextul demografic simplu (Vârsta și Genul influențează masiv probabilitățile de bază ale anumitor mutații), AI-ul prezice biologia intimă a tumorii direct din imagini, instantaneu.

---

## 🛠️ 4. Tehnologii și Stack Recomandat

*   **Manipulare Date Medicale:** `pydicom`, `NiBabel` (pentru co-registrarea CT și PET în același spațiu geometric).
*   **Deep Learning Engine:** `PyTorch`
*   **Medical AI Framework:** **`MONAI`** (Medical Open Network for AI).
*   **Arhitectură Model (Late-Fusion):** 
    *   *Branch 1 (Imagine):* 3D ResNet cu 2 canale de intrare (Canalul 1: CT în Hounsfield Units, Canalul 2: PET în SUV - Standardized Uptake Value).
    *   *Branch 2 (Tabular):* Multi-Layer Perceptron (MLP) cu Batch Normalization.
*   **Interfață Utilizator (UI):** React / Vue.js integrat cu vizualizatoare DICOM precum `Cornerstone.js` sau `VTK.js` pentru afișarea fuziunii CT/PET direct în browser.