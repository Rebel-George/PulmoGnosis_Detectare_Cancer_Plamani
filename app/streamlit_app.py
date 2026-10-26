"""PulmoGnosis demo app: upload → view CT+PET → clinical data → localise → results.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import streamlit as st

from pulmognosis.data.dicom import SeriesInfo, pick_series, scan_folder
from pulmognosis.data.synthetic import write_demo_case
from pulmognosis.inference import PlaceholderPredictor, Predictor
from pulmognosis.pipeline import (
    MODEL_PATCH_MM,
    MODEL_PATCH_SIZE,
    MODEL_SPACING_MM,
    SEX_CHOICES,
    SMOKING_CHOICES,
    ClinicalData,
    extract_model_patch,
    prepare_case,
)
from pulmognosis.preprocessing.intensity import WINDOWS
from pulmognosis.preprocessing.spatial import to_array
from rendering import PLANES, patch_figure, slice_figure

STEPS = (
    "1 · Încărcare",
    "2 · Vizualizare CT + PET",
    "3 · Date clinice",
    "4 · Localizarea tumorii",
    "5 · Rezultate",
)
WINDOW_LABELS = {"Plămân": WINDOWS["lung"], "Mediastin": WINDOWS["mediastinum"]}

st.set_page_config(page_title="PulmoGnosis", page_icon="🫁", layout="wide")


# ---------------------------------------------------------------- state ----

def get_predictor() -> Predictor:
    # Replaced by the trained Stage 1 + Stage 2 models in a later stage.
    return PlaceholderPredictor()


def reset_from(step: str) -> None:
    """Clear everything computed at or after `step`, so stale results never show."""
    keys = {
        "case": ["case", "ct", "pet", "dicom_header", "clinical", "location", "patch_cache"],
        "clinical": ["clinical"],
        "location": ["location"],
    }[step]
    for key in keys:
        st.session_state.pop(key, None)


@st.cache_resource
def demo_case_folder() -> str:
    return str(write_demo_case(Path(tempfile.mkdtemp(prefix="pulmognosis_demo_"))))


@st.cache_data(show_spinner=False)
def scan(folder: str) -> list[SeriesInfo]:
    return scan_folder(folder)


def model_patch(center_zyx):
    """The classifier's input cube, cached per centre so slider moves elsewhere stay fast."""
    cache = st.session_state.setdefault("patch_cache", {})
    key = tuple(int(c) for c in center_zyx)
    if key not in cache:
        cache.clear()
        cache[key] = extract_model_patch(st.session_state.case, key)
    return cache[key]


def extract_zip(uploaded) -> str:
    target = Path(tempfile.mkdtemp(prefix="pulmognosis_upload_"))
    with zipfile.ZipFile(uploaded) as zf:
        zf.extractall(target)
    return str(target)


# ---------------------------------------------------------------- steps ----

def step_upload() -> None:
    st.header("1 · Încărcarea scanărilor")
    st.write(
        "Selectați un set de fișiere DICOM care conține un **CT toracic** și, opțional, un **PET**. "
        "Puteți folosi și cazul demonstrativ (o fantomă sintetică, nu un pacient real)."
    )
    source = st.radio(
        "Sursa datelor",
        ("Caz demonstrativ (fantomă sintetică)", "Folder local cu fișiere DICOM", "Arhivă ZIP"),
        horizontal=True,
    )

    folder = None
    if source.startswith("Caz demonstrativ"):
        folder = demo_case_folder()
    elif source.startswith("Folder"):
        path = st.text_input("Calea către folder", placeholder=r"D:\date\NSCLC-Radiogenomics\R01-001")
        if path:
            if Path(path).is_dir():
                folder = path
            else:
                st.error("Folderul nu există.")
    else:
        uploaded = st.file_uploader("Arhivă .zip cu seriile DICOM", type="zip")
        if uploaded is not None:
            if st.session_state.get("zip_name") != uploaded.name:
                st.session_state.zip_folder = extract_zip(uploaded)
                st.session_state.zip_name = uploaded.name
            folder = st.session_state.zip_folder

    if folder is None:
        return

    with st.spinner("Se caută seriile DICOM…"):
        series = scan(folder)
    if not series:
        st.error("Nu am găsit nicio serie DICOM cu cel puțin 2 secțiuni.")
        return

    st.dataframe(
        pd.DataFrame(
            [{"Modalitate": s.modality, "Descriere": s.description, "Secțiuni": s.num_slices} for s in series]
        ),
        hide_index=True,
        width="stretch",
    )

    ct_options = [s for s in series if s.modality == "CT"]
    pet_options = [s for s in series if s.modality == "PT"]
    if not ct_options:
        st.error("Nu există nicio serie CT.")
        return

    def label(s: SeriesInfo) -> str:
        return f"{s.modality} · {s.description or 'fără descriere'} · {s.num_slices} secțiuni"

    col1, col2 = st.columns(2)
    default_ct = pick_series(series, "CT")
    ct = col1.selectbox("Seria CT", ct_options, index=ct_options.index(default_ct), format_func=label)
    default_pet = pick_series(series, "PT")
    pet = col2.selectbox("Seria PET (opțional)", [None, *pet_options],
                         index=pet_options.index(default_pet) + 1 if default_pet else 0,
                         format_func=lambda s: "fără PET" if s is None else label(s))

    if st.button("Pregătește cazul", type="primary"):
        reset_from("case")
        with st.spinner("Conversie HU / SUV și alinierea PET pe CT…"):
            case = prepare_case(ct, pet)
        st.session_state.case = case
        st.session_state.ct = to_array(case.ct)
        st.session_state.pet = to_array(case.pet) if case.has_pet else None
        st.session_state.dicom_header = pydicom.dcmread(ct.files[0], stop_before_pixels=True)

    if "case" in st.session_state:
        case = st.session_state.case
        z, y, x = st.session_state.ct.shape
        sx, sy, sz = case.spacing_xyz
        st.success(
            f"Caz pregătit: CT {x}×{y}×{z} voxeli la rezoluția originală ({sx:.2f} × {sy:.2f} × {sz:.2f} mm), "
            + ("cu PET aliniat." if case.has_pet else "fără PET.")
            + " Continuați cu pasul 2."
        )
        if case.pet_warning:
            st.warning(case.pet_warning)


def view_controls(key: str, volume_shape, default_index=None):
    c1, c2 = st.columns([1, 2])
    plane = c1.selectbox("Plan", list(PLANES), key=f"{key}_plane")
    n = volume_shape[PLANES[plane]]
    index = c2.slider("Secțiune", 0, n - 1, n // 2 if default_index is None else int(default_index), key=f"{key}_{plane}_idx")
    return plane, index


def step_view() -> None:
    st.header("2 · Vizualizare CT + PET")
    ct, pet = st.session_state.ct, st.session_state.pet

    left, right = st.columns([3, 1])
    with right:
        win_label = st.radio("Fereastră CT", list(WINDOW_LABELS))
        if pet is not None:
            show_pet = st.toggle("Suprapune PET", value=True)
            opacity = st.slider("Opacitate PET", 0.0, 1.0, 0.5, 0.05) if show_pet else 0.0
            suv_max = max(float(pet.max()), 1.0)
            suv_min, suv_top = st.slider("Interval SUV afișat", 0.0, suv_max, (min(2.0, suv_max / 2), suv_max), 0.1)
        else:
            opacity, suv_min, suv_top = 0.0, 0.0, 1.0
            st.info("Acest caz nu are PET.")
        st.metric("SUV maxim", f"{pet.max():.1f}" if pet is not None else "—")
        sx, sy, sz = st.session_state.case.spacing_xyz
        st.caption(f"Rezoluția originală a CT-ului: {sx:.2f} × {sy:.2f} mm în plan, secțiuni de {sz:.2f} mm.")

    with left:
        plane, index = view_controls("view", ct.shape)
        fig = slice_figure(ct, plane, index, WINDOW_LABELS[win_label], pet, opacity, suv_min, suv_top,
                           spacing_xyz=st.session_state.case.spacing_xyz, size=6.5)
        st.pyplot(fig, width="content")
        plt.close(fig)


def step_clinical() -> None:
    st.header("3 · Date clinice")
    st.write("Datele clinice intră în model alături de imagini. Statutul de fumător este un predictor puternic pentru EGFR și KRAS.")

    header = st.session_state.get("dicom_header")
    age_default, sex_default = 60, 0
    if header is not None:
        age_text = str(getattr(header, "PatientAge", ""))
        if age_text[:3].isdigit():
            age_default = int(age_text[:3])
        sex_default = 1 if str(getattr(header, "PatientSex", "")).upper() == "F" else 0

    previous = st.session_state.get("clinical")
    with st.form("clinical_form"):
        age = st.number_input("Vârstă (ani)", 18, 100, previous.age if previous else age_default)
        sex = st.radio("Gen", SEX_CHOICES, index=SEX_CHOICES.index(previous.sex) if previous else sex_default, horizontal=True)
        smoking = st.selectbox("Statut de fumător", SMOKING_CHOICES,
                               index=SMOKING_CHOICES.index(previous.smoking) if previous else 1)
        if st.form_submit_button("Salvează", type="primary"):
            st.session_state.clinical = ClinicalData(int(age), sex, smoking)
            reset_from("location")
    if header is not None:
        st.caption("Vârsta și genul au fost pre-completate din antetul DICOM, când erau disponibile.")
    if "clinical" in st.session_state:
        st.success("Date clinice salvate. Continuați cu pasul 4.")


def step_localise() -> None:
    st.header("4 · Localizarea tumorii")
    ct, pet = st.session_state.ct, st.session_state.pet
    case = st.session_state.case
    predictor = get_predictor()

    if "location" not in st.session_state:
        st.session_state.location = predictor.localise(st.session_state.case)
    loc = st.session_state.location

    st.write(
        f"Metodă: **{loc.method}**. În versiunea finală, această etapă va fi făcută de modelul 3D U-Net (Etapa 1 a pipeline-ului). "
        f"Puteți ajusta manual centrul; caseta arată cubul de {MODEL_PATCH_MM:.0f} mm trimis clasificatorului "
        f"({MODEL_PATCH_SIZE}³ voxeli de {MODEL_SPACING_MM} mm)."
    )

    z0, y0, x0 = loc.center_zyx
    c1, c2, c3 = st.columns(3)
    x = c1.slider("x (dreapta → stânga pacientului)", 0, ct.shape[2] - 1, int(x0))
    y = c2.slider("y (anterior → posterior)", 0, ct.shape[1] - 1, int(y0))
    z = c3.slider("z (inferior → superior)", 0, ct.shape[0] - 1, int(z0))
    center = (z, y, x)
    if center != tuple(loc.center_zyx):
        loc = type(loc)(center, "ajustat manual")
        st.session_state.location = loc

    cols = st.columns(3)
    for col, plane in zip(cols, PLANES):
        with col:
            fig = slice_figure(ct, plane, center[PLANES[plane]], WINDOWS["lung"], pet, 0.45, 2.0,
                               max(float(pet.max()), 2.1) if pet is not None else 10.0,
                               spacing_xyz=case.spacing_xyz, center_zyx=center, box_mm=MODEL_PATCH_MM,
                               title=plane, size=4)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

    st.subheader(f"Intrarea modelului: {MODEL_PATCH_SIZE}³ voxeli de {MODEL_SPACING_MM} mm")
    ct_patch, pet_patch = model_patch(center)
    fig = patch_figure(ct_patch, pet_patch, WINDOWS["lung"])
    st.pyplot(fig, width="content")
    plt.close(fig)


def bar_row(label: str, value: float) -> None:
    c1, c2 = st.columns([2, 3])
    c1.write(f"{label}: **{value:.0%}**")
    c2.progress(float(value))


def step_results() -> None:
    st.header("5 · Rezultate")
    predictor = get_predictor()
    prediction = predictor.predict(st.session_state.case, st.session_state.location, st.session_state.clinical)

    if prediction.is_placeholder:
        st.error("**DEMO – fără model antrenat.** " + prediction.note, icon="⚠️")

    left, right = st.columns(2)
    with left:
        st.subheader("Tip histologic")
        for name, p in sorted(prediction.histology.items(), key=lambda kv: -kv[1]):
            bar_row(name.capitalize(), p)
    with right:
        st.subheader("Mutații")
        m1, m2 = st.columns(2)
        m1.metric("EGFR mutant", f"{prediction.egfr:.0%}")
        m2.metric("KRAS mutant", f"{prediction.kras:.0%}")
        st.caption("EGFR: candidat pentru terapie țintită cu inhibitori de tirozin-kinază. KRAS: candidat pentru inhibitori KRAS.")

    st.subheader("Hartă de atenție (Grad-CAM)")
    st.info("Va fi disponibilă după antrenarea clasificatorului (Etapa 5 a proiectului).")

    clinical = st.session_state.clinical
    with st.expander("Datele folosite"):
        st.write(f"- Vârstă: {clinical.age} ani · Gen: {clinical.sex} · Fumat: {clinical.smoking}")
        st.write(f"- PET: {'da' if st.session_state.case.has_pet else 'nu'}")
        st.write(f"- Centrul regiunii (z, y, x): {tuple(st.session_state.location.center_zyx)} · {st.session_state.location.method}")

    st.caption("PulmoGnosis este un instrument de cercetare, nu un dispozitiv medical. Rezultatele trebuie confirmate prin biopsie și testare moleculară.")


# ---------------------------------------------------------------- layout ----

def requirements_met(step_index: int) -> str | None:
    """Message explaining what is missing before `step_index` can run, or None."""
    if step_index >= 1 and "case" not in st.session_state:
        return "Încărcați și pregătiți mai întâi un caz (pasul 1)."
    if step_index >= 4 and "clinical" not in st.session_state:
        return "Completați mai întâi datele clinice (pasul 3)."
    if step_index >= 4 and "location" not in st.session_state:
        return "Verificați mai întâi localizarea tumorii (pasul 4)."
    return None


with st.sidebar:
    st.title("🫁 PulmoGnosis")
    st.caption("Biopsie virtuală pentru cancerul pulmonar (NSCLC)")
    step = st.radio("Pași", STEPS, label_visibility="collapsed")
    st.divider()
    progress = st.container()
    st.divider()
    st.caption("Versiune demonstrativă (Etapa 1): flow-ul complet, fără model antrenat.")

step_index = STEPS.index(step)
missing = requirements_met(step_index)
if missing:
    st.info(missing)
else:
    (step_upload, step_view, step_clinical, step_localise, step_results)[step_index]()

# Drawn after the step runs, so it reflects what this run just completed.
with progress:
    done = [
        "case" in st.session_state,
        "case" in st.session_state,
        "clinical" in st.session_state,
        "location" in st.session_state,
    ]
    for name, ok in zip(STEPS[:4], done):
        st.write(("✅ " if ok else "⬜ ") + name)
