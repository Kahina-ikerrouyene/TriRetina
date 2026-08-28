import streamlit as st
import numpy as np
import cv2
import keras
import os
import time
import datetime
import tempfile
from fpdf import FPDF

st.set_page_config(
    page_title="TriRetina — Analyse Oculaire IA",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS COMPLET ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&display=swap');

/* Base */
.main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1280px; }
#MainMenu, footer { visibility: hidden; }
.stApp { background: #F0F2F6; }

/* ── HEADER ── */
.triretina-header {
    background: linear-gradient(135deg, #F5A623 0%, #F07B45 45%, #E85D4A 100%);
    padding: 0.8rem 1.8rem;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.2rem;
    box-shadow: 0 6px 24px rgba(232,93,74,0.22);
}
.triretina-header-title { color: white; font-size: 1.35rem; font-weight: 800;
    font-family: 'Sora', sans-serif; letter-spacing: -0.3px;
    margin: 0; text-shadow: 0 1px 3px rgba(0,0,0,0.15); }
.triretina-header-sub { color: rgba(255,255,255,0.88); margin: 0.2rem 0 0 0;
    font-size: 0.79rem; line-height: 1.5; max-width: 560px; }
.triretina-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.triretina-badge {
    background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.4);
    color: white; padding: 0.3rem 0.9rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px); white-space: nowrap;
}

/* ── KPI CARDS ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: white; border-radius: 12px; padding: 1.2rem 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #EAECF0;
    text-align: center; border-top: 3px solid;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 6px 24px rgba(0,0,0,0.1); }
.kpi-value { font-size: 2rem; font-weight: 800; margin: 0; line-height: 1; }
.kpi-label { color: #6B7280; font-size: 0.72rem; margin: 0.4rem 0 0 0;
    text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600; }
.kpi-sub { color: #9CA3AF; font-size: 0.68rem; margin: 0.2rem 0 0 0; }

/* ── UPLOAD DnD natif Streamlit ── */
[data-testid="stFileUploader"] { margin-bottom: 1.2rem; }
[data-testid="stFileUploaderDropzone"] {
    background: white !important;
    border: 2px dashed #F07B45 !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
    padding: 2rem 2rem 1.5rem 2rem !important;
    display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important;
    cursor: pointer !important; transition: border-color 0.25s !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #F07B45 !important; }
[data-testid="stFileUploaderDropzone"]::before {
    content: '';
    display: block; width: 44px; height: 44px; margin-bottom: 0.9rem;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='44' height='44' viewBox='0 0 24 24' fill='none' stroke='%23D1D5DB' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='16 16 12 12 8 16'%3E%3C/polyline%3E%3Cline x1='12' y1='12' x2='12' y2='21'%3E%3C/line%3E%3Cpath d='M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3'%3E%3C/path%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: center; background-size: contain;
}
[data-testid="stFileUploaderDropzoneInstructions"] { text-align: center !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child {
    font-size: 0.95rem !important; font-weight: 700 !important; color: #374151 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    font-size: 0.74rem !important; color: #9CA3AF !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: transparent !important; border: 1px solid #E5E7EB !important;
    color: #6B7280 !important; border-radius: 8px !important;
    font-size: 0.78rem !important; padding: 0.3rem 1rem !important; margin-top: 0.4rem !important;
    box-shadow: none !important;
}

/* ── DISEASE INFO CARDS ── */
.disease-cards-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.disease-card {
    background: white; border-radius: 12px; padding: 1rem 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #EAECF0;
    transition: transform 0.2s;
}
.disease-card:hover { transform: translateY(-2px); }
.disease-card h4 { font-size: 0.9rem; font-weight: 700; margin: 0 0 0.4rem 0; }
.disease-card .auc-badge {
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 10px;
    font-size: 0.7rem; font-weight: 700; margin-bottom: 0.5rem;
}
.disease-card p { font-size: 0.78rem; color: #6B7280; margin: 0; line-height: 1.5; }

/* ── SECTION RESULTATS ── */
.section-card {
    background: white; border-radius: 16px; padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #EAECF0;
    height: 100%;
}
.section-title {
    font-size: 0.85rem; font-weight: 700; color: #6B7280;
    text-transform: uppercase; letter-spacing: 0.6px;
    margin: 0 0 1rem 0; padding-bottom: 0.8rem; border-bottom: 2px solid #F3F4F6;
    display: flex; align-items: center; gap: 0.4rem;
}

/* ── GAUGE ── */
.gauge-row {
    display: flex; align-items: center; gap: 1rem;
    padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem;
    border: 1px solid #F3F4F6; background: #FAFBFC; transition: all 0.2s;
}
.gauge-row:hover { background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
.gauge-donut {
    width: 76px; height: 76px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    position: relative;
}
.gauge-center {
    width: 56px; height: 56px; border-radius: 50%; background: #FAFBFC;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; font-weight: 800; color: #111827; position: relative; z-index: 1;
}
.gauge-info { flex: 1; min-width: 0; }
.gauge-name { font-size: 0.95rem; font-weight: 700; color: #111827; margin: 0 0 0.15rem 0; }
.gauge-desc { font-size: 0.72rem; color: #9CA3AF; margin: 0 0 0.5rem 0; }

.badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.2rem 0.65rem; border-radius: 20px; font-size: 0.72rem; font-weight: 700;
}
.badge-pos { background: #FEE2E2; color: #B91C1C; }
.badge-neg { background: #DCFCE7; color: #166534; }
.badge-lim { background: #FEF9C3; color: #854D0E; }

.progress-bar { height: 5px; border-radius: 3px; background: #E5E7EB; margin-top: 0.4rem; overflow: hidden; }
.progress-fill { height: 100%; border-radius: 3px; transition: width 0.6s cubic-bezier(.4,0,.2,1); }

/* ── SYNTHESIS ── */
.synth-ok    { background: linear-gradient(135deg,#DCFCE7,#BBF7D0); border:1px solid #86EFAC; border-radius:12px; padding:1.2rem 1.5rem; display:flex; gap:1rem; align-items:center; }
.synth-warn  { background: linear-gradient(135deg,#FEF9C3,#FDE68A); border:1px solid #FCD34D; border-radius:12px; padding:1.2rem 1.5rem; display:flex; gap:1rem; align-items:center; }
.synth-alert { background: linear-gradient(135deg,#FEE2E2,#FECACA); border:1px solid #FCA5A5; border-radius:12px; padding:1.2rem 1.5rem; display:flex; gap:1rem; align-items:center; }
.synth-icon { font-size:2.2rem; flex-shrink:0; }
.synth-body h3 { margin:0 0 0.2rem 0; font-size:1rem; font-weight:700; }
.synth-body p  { margin:0; font-size:0.82rem; }

/* ── TABLE ── */
.recap-table { width:100%; border-collapse:collapse; font-size:0.82rem; margin-top:1rem; }
.recap-table th { background:#F9FAFB; padding:0.65rem 1rem; text-align:left;
    color:#6B7280; font-weight:600; font-size:0.7rem; text-transform:uppercase;
    letter-spacing:0.5px; border-bottom:2px solid #E5E7EB; }
.recap-table td { padding:0.75rem 1rem; border-bottom:1px solid #F3F4F6; color:#374151; }
.recap-table tr:last-child td { border-bottom:none; }
.recap-table tr:hover td { background:#F9FAFB; }

/* ── DISCLAIMER ── */
.disclaimer {
    background:#FFF7ED; border:1px solid #FED7AA; border-radius:8px;
    padding:0.7rem 1rem; margin-top:1rem; font-size:0.74rem; color:#92400E;
    display:flex; gap:0.5rem; align-items:flex-start; line-height:1.5;
}

/* ── BOUTON PDF ── */
.stDownloadButton > button {
    background: linear-gradient(135deg,#F5A623,#E85D4A) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 0.88rem !important;
    padding: 0.65rem 1.5rem !important; width: 100% !important; margin-top: 1rem !important;
    box-shadow: 0 4px 14px rgba(232,93,74,0.3) !important;
    transition: opacity 0.2s !important;
}
.stDownloadButton > button:hover { opacity: 0.88 !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] > div { background:#F8F9FC; }
section[data-testid="stSidebar"] { border-right:1px solid #E5E7EB; }
.sb-header {
    background: linear-gradient(135deg,#F5A623,#E85D4A);
    padding:1rem 1.2rem; border-radius:12px; margin-bottom:1rem; text-align:center;
}
.sb-header h2 { color:white; margin:0; font-size:1.1rem; font-weight:800; font-family:'Sora',sans-serif; }
.sb-header p  { color:rgba(255,255,255,0.85); margin:0.2rem 0 0 0; font-size:0.75rem; }
.sb-section { background:white; border-radius:10px; padding:1rem; margin-bottom:0.8rem;
    box-shadow:0 1px 4px rgba(0,0,0,0.05); border:1px solid #EAECF0; }
.sb-section h4 { color:#374151; font-size:0.78rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.5px; margin:0 0 0.8rem 0; padding-bottom:0.5rem; border-bottom:1px solid #F3F4F6; }
.model-stat { display:flex; justify-content:space-between; align-items:center;
    padding:0.35rem 0; border-bottom:1px solid #F9FAFB; font-size:0.8rem; }
.model-stat:last-child { border-bottom:none; }
.model-stat span:first-child { color:#6B7280; }
.model-stat span:last-child  { color:#111827; font-weight:600; }

/* Image label */
.img-label { font-size:0.75rem; font-weight:700; color:#6B7280; text-transform:uppercase;
    letter-spacing:0.5px; margin:0 0 0.5rem 0; text-align:center; }
.img-meta { font-size:0.67rem; color:#9CA3AF; margin:0.4rem 0 0 0;
    text-align:center; letter-spacing:0.2px; }
</style>
""", unsafe_allow_html=True)

# ── CONFIG MALADIES ──────────────────────────────────────────────────────────
DISEASES = {
    "RD": {
        "label":   "Rétinopathie Diabétique",
        "short":   "Lésions vasculaires rétiniennes dues au diabète",
        "thresh":  0.40,
        "color":   "#E74C3C",
        "auc":     "0.9186",
    },
    "Glaucome": {
        "label":   "Glaucome",
        "short":   "Neuropathie du nerf optique — évolution silencieuse",
        "thresh":  0.35,
        "color":   "#3B82F6",
        "auc":     "0.9266",
    },
    "DMLA": {
        "label":   "DMLA",
        "short":   "Dégénérescence Maculaire Liée à l'Âge",
        "thresh":  0.45,
        "color":   "#F5A623",
        "auc":     "0.9175",
    },
}

# ── MODEL ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "EfficientNetB3_best.keras")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH, compile=False)

def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = img_rgb.copy()
    for c in range(3):
        out[:, :, c] = clahe.apply(img_rgb[:, :, c])
    return out

def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    img = apply_clahe(img_rgb)
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return np.expand_dims(img, 0)

# ── GÉNÉRATION PDF ─────────────────────────────────────────────────────────────
def generate_pdf_report(filename, img_rgb, img_clahe, probas, keys, detected,
                        inference_time, h_orig, w_orig, show_clahe, diseases_cfg,
                        patient_ref="", examiner=""):

    MODEL_VERSION = "v1.0"
    now       = datetime.datetime.now()
    report_id = f"TR-{now.strftime('%Y%m%d-%H%M%S')}"
    M         = 13
    CW        = 210 - 2 * M   # 184 mm

    CLR_NAV = (23, 46, 90)         # navy — séparateurs titres
    CLR_ACC = (240, 123, 69)       # orange app — titre TriRetina
    CLR_PRI = (17, 24, 39)         # quasi-noir — texte principal
    CLR_SEC = (75, 85, 99)         # gris foncé — texte secondaire
    CLR_MUT = (130, 140, 155)      # gris moyen — labels, notes
    CLR_LNE = (195, 200, 210)      # gris clair — séparateurs
    CLR_RED = (185, 28, 28)        # rouge — alertes uniquement

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    FD = "C:/Windows/Fonts/"
    try:
        pdf.add_font("F",       fname=FD+"times.ttf")
        pdf.add_font("F", "B",  fname=FD+"timesbd.ttf")
        pdf.add_font("F", "I",  fname=FD+"timesi.ttf")
        pdf.add_font("F", "BI", fname=FD+"timesbi.ttf")
        F = "F"
    except Exception:
        F = "Helvetica"
    pdf.add_page()

    def line(y):
        pdf.set_draw_color(*CLR_LNE); pdf.set_line_width(0.2)
        pdf.line(M, y, 210 - M, y)

    def sec(label):
        y = pdf.get_y()
        pdf.set_xy(M, y)
        pdf.set_font(F, "B", 9); pdf.set_text_color(*CLR_PRI)
        pdf.cell(0, 7, label, ln=True)
        line(pdf.get_y())
        pdf.ln(3)

    def save_img(arr):
        t = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(t.name, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        t.close()
        return t.name

    def place_contain(img_path, box_x, box_y, box_w, box_h):
        aspect  = w_orig / max(h_orig, 1)
        box_asp = box_w / max(box_h, 0.001)
        if aspect > box_asp:
            dw, dh = box_w, box_w / aspect
        else:
            dw, dh = box_h * aspect, box_h
        dx = box_x + (box_w - dw) / 2
        dy = box_y + (box_h - dh) / 2
        pdf.image(img_path, x=dx, y=dy, w=dw, h=dh)

    # ── Textes cliniques ───────────────────────────
    RECO = {
        "RD": {
            "signal": "Signal d'alerte : Rétinopathie Diabétique",
            "reco":   ("Consultation ophtalmologique recommandée pour évaluation et classification "
                       "selon la grille HAS (stades 0 à 4 de la rétinopathie diabétique). "
                       "Contrôle de l'HbA1c et de la pression artérielle recommandé."),
        },
        "Glaucome": {
            "signal": "Signal d'alerte : Glaucome",
            "reco":   ("Consultation ophtalmologique pour mesure de la pression intraoculaire "
                       "(tonomètre de Goldmann), champ visuel automatisé (périmètre Humphrey) "
                       "et pachymétrie. La pression intraoculaire n'est pas mesurée par cet outil. "
                       "Évolution silencieuse : avis spécialisé recommandé même sans symptômes."),
        },
        "DMLA": {
            "signal": "Signal d'alerte : DMLA",
            "reco":   ("Consultation ophtalmologique pour OCT maculaire et rétinographie. "
                       "En cas de métamorphopsies ou de baisse d'acuité rapide, "
                       "prise en charge inférieure à 1 semaine (forme exsudative à exclure)."),
        },
    }
    RECO_SHORT = {
        "RD":       "RD : évaluation ophtalmo et classification selon grille HAS. Contrôle HbA1c recommandé.",
        "Glaucome": "Glaucome : tonométrie (Goldmann), champ visuel et pachymétrie. Avis spécialisé urgent.",
        "DMLA":     "DMLA : OCT maculaire. Si BAV rapide ou métamorphopsies : prise en charge < 1 semaine.",
    }

    if not detected:
        bar_c     = (22, 101, 52)
        titre_res = "Aucun signal d'alerte identifié pour les pathologies ciblées"
        sous_res  = ("Aucun signe évocateur de rétinopathie diabétique, de glaucome ou de DMLA "
                     "n'a été identifié à l'analyse automatisée.")
        reco_txt  = ("Maintenir un suivi ophtalmologique adapté aux facteurs de risque du patient. "
                     "Ce résultat ne dispense pas d'un examen clinique complet en cas de symptômes visuels.")
    elif len(detected) == 1:
        bar_c     = CLR_RED
        titre_res = RECO[detected[0]]["signal"]
        score_d   = round(float(probas[keys.index(detected[0])]) * 100)
        sous_res  = (f"Score d'alerte : {score_d}/100. "
                     "Confirmation clinique requise avant toute décision thérapeutique.")
        reco_txt  = RECO[detected[0]]["reco"]
    else:
        bar_c     = CLR_RED
        lbls_det  = " et ".join(RECO[d]["signal"].replace("Signal d'alerte : ", "") for d in detected)
        titre_res = f"Signaux d'alerte multiples : {lbls_det}"
        sous_res  = "Scores supérieurs aux seuils pour plusieurs pathologies. Avis ophtalmologique prioritaire."
        reco_txt  = "\n".join(RECO_SHORT[d] for d in detected)

    pat_display  = patient_ref.strip()[:32] if patient_ref and patient_ref.strip() else ""
    exam_display = examiner.strip()[:32]    if examiner and examiner.strip()        else ""

    # ══════════════════════════════════════════════════════
    # 1. BANDE ORANGE  (0 → 2.5 mm)
    # ══════════════════════════════════════════════════════
    pdf.set_fill_color(*CLR_ACC)
    pdf.rect(0, 0, 210, 2.5, "F")

    # ══════════════════════════════════════════════════════
    # 2. NOM + RÉFÉRENCE  (2.5 → 19 mm)
    # ══════════════════════════════════════════════════════
    pdf.set_xy(M, 5)
    pdf.set_font(F, "B", 19); pdf.set_text_color(*CLR_ACC)
    pdf.cell(CW * 0.5, 9, "TriRetina", ln=False)

    pdf.set_xy(M + CW * 0.5, 5.5)
    pdf.set_font(F, "", 7.5); pdf.set_text_color(*CLR_MUT)
    pdf.cell(CW * 0.5, 4.5, f"Réf. {report_id}", align="R", ln=False)
    pdf.set_xy(M + CW * 0.5, 10.5)
    pdf.set_font(F, "I", 6.5); pdf.set_text_color(*CLR_MUT)
    pdf.cell(CW * 0.5, 4, f"Système : TriRetina {MODEL_VERSION} (EfficientNetB3)", align="R")

    pdf.set_draw_color(*CLR_ACC); pdf.set_line_width(0.5)
    pdf.line(M, 18.5, 210 - M, 18.5)

    # ══════════════════════════════════════════════════════
    # 3. BLOC IDENTIFICATION  (19 → 46 mm)
    # ══════════════════════════════════════════════════════
    # — Gauche : outil + patient + examinateur + fichier
    pdf.set_xy(M, 22)
    pdf.set_font(F, "I", 7); pdf.set_text_color(*CLR_MUT)
    pdf.cell(CW * 0.5, 4.5, "Outil d'aide au dépistage rétinien par intelligence artificielle")

    pdf.set_xy(M, 27)
    pdf.set_font(F, "", 7.5); pdf.set_text_color(*CLR_MUT)
    pdf.cell(24, 4.5, "Patient : ", ln=False)
    pdf.set_font(F, "B", 7.5); pdf.set_text_color(*CLR_PRI)
    pdf.cell(0, 4.5, pat_display if pat_display else "—")

    pdf.set_xy(M, 32)
    pdf.set_font(F, "", 7.5); pdf.set_text_color(*CLR_MUT)
    pdf.cell(24, 4.5, "Examinateur : ", ln=False)
    pdf.set_font(F, "B", 7.5); pdf.set_text_color(*CLR_PRI)
    pdf.cell(0, 4.5, exam_display if exam_display else "—")

    fname_short = filename[:22] + ("…" if len(filename) > 22 else "")
    pdf.set_xy(M, 37)
    pdf.set_font(F, "", 7); pdf.set_text_color(*CLR_MUT)
    pdf.cell(CW * 0.5, 4, f"Fichier : {fname_short}")

    # — Droite : date
    pdf.set_xy(M + CW * 0.5, 22)
    pdf.set_font(F, "", 8.5); pdf.set_text_color(*CLR_PRI)
    pdf.cell(CW * 0.5, 5, f"Le {now.strftime('%d/%m/%Y')}", align="R")

    line(42)

    # ══════════════════════════════════════════════════════
    # 4. TITRE DU DOCUMENT  (42 → 53 mm)
    # ══════════════════════════════════════════════════════
    pdf.set_xy(M, 44)
    pdf.set_font(F, "B", 11); pdf.set_text_color(*CLR_PRI)
    pdf.cell(CW, 6, "Rapport d'analyse rétinienne automatisée", align="C")
    line(52)

    # ══════════════════════════════════════════════════════
    # 5. IMAGERIE  (54 → ~114 mm)
    # ══════════════════════════════════════════════════════
    pdf.set_xy(M, 54)
    sec("Imagerie du fond d'œil")

    IMG_W, IMG_H = 65, 50
    GAP = CW - 2 * IMG_W
    y_img = pdf.get_y()

    p_orig  = save_img(img_rgb)
    p_clahe = save_img(img_clahe)

    pdf.set_font(F, "I", 6.5); pdf.set_text_color(*CLR_MUT)
    pdf.set_xy(M, y_img)
    pdf.cell(IMG_W, 4, "Image originale", align="C", ln=False)
    pdf.set_xy(M + IMG_W + GAP, y_img)
    pdf.cell(IMG_W, 4, "Image améliorée (contraste CLAHE)", align="C")

    place_contain(p_orig,  M,               y_img + 4.5, IMG_W, IMG_H)
    place_contain(p_clahe, M + IMG_W + GAP, y_img + 4.5, IMG_W, IMG_H)

    for p in (p_orig, p_clahe):
        try: os.unlink(p)
        except Exception: pass

    y_cap = y_img + IMG_H + 5.5
    pdf.set_font(F, "I", 5.5); pdf.set_text_color(*CLR_MUT)
    pdf.set_xy(M, y_cap)
    pdf.cell(IMG_W, 3.5, f"Sans traitement  ·  {w_orig} × {h_orig} px", align="C", ln=False)
    pdf.set_xy(M + IMG_W + GAP, y_cap)
    pdf.cell(IMG_W, 3.5, "Rehaussement adaptatif du contraste", align="C")
    pdf.set_xy(M, y_cap + 3.5)

    # ══════════════════════════════════════════════════════
    # 4. RÉSULTATS  (~117 → ~157 mm)
    # ══════════════════════════════════════════════════════
    pdf.ln(10)
    sec("Résultats par pathologie")

    # Tableau style compte rendu de biologie
    cw_t = [CW * 0.45, CW * 0.20, CW * 0.35]
    pdf.set_draw_color(*CLR_LNE); pdf.set_line_width(0.2)
    pdf.set_font(F, "BI", 6.5); pdf.set_text_color(*CLR_MUT)
    pdf.set_x(M)
    for w_c, h_c in zip(cw_t, ["Pathologie", "Score", "Interprétation"]):
        pdf.cell(w_c, 5.5, h_c, border="B", align="L")
    pdf.ln()

    for key, proba in zip(keys, probas):
        cfg   = diseases_cfg[key]
        score = round(min((float(proba) / cfg["thresh"]) * 50, 100))
        pos   = float(proba) >= cfg["thresh"]
        pdf.set_x(M)
        pdf.set_font(F, "B", 8); pdf.set_text_color(*CLR_PRI)
        pdf.cell(cw_t[0], 7, cfg["label"], border="B")
        pdf.set_font(F, "B" if pos else "", 8); pdf.set_text_color(*CLR_PRI)
        pdf.cell(cw_t[1], 7, f"{score} %", border="B", align="C")
        if pos:
            pdf.set_font(F, "B", 8); pdf.set_text_color(*CLR_RED)
            concl = "Signal d'alerte"
        else:
            pdf.set_font(F, "I", 7.5); pdf.set_text_color(*CLR_SEC)
            concl = "Normal"
        pdf.cell(cw_t[2], 7, concl, border="B")
        pdf.ln()

    pdf.ln(2); pdf.set_x(M)
    pdf.set_font(F, "I", 5.5); pdf.set_text_color(*CLR_MUT)
    pdf.multi_cell(CW, 3.5,
        "Score normalisé : > 50 % = signal d'alerte, < 50 % = normal.  "
        "Un score élevé requiert une confirmation clinique.")

    # ══════════════════════════════════════════════════════
    # 5. CONCLUSION
    # ══════════════════════════════════════════════════════
    pdf.ln(8)
    sec("Conclusion")

    y_res = pdf.get_y()
    pdf.set_fill_color(*bar_c)
    pdf.rect(M, y_res, 2, 13, "F")
    pdf.set_xy(M + 5, y_res + 1.5)
    pdf.set_font(F, "B", 8.5); pdf.set_text_color(*CLR_PRI)
    pdf.cell(CW - 6, 5.5, titre_res, ln=True)
    pdf.set_xy(M + 5, pdf.get_y())
    pdf.set_font(F, "I", 7.5); pdf.set_text_color(*CLR_SEC)
    pdf.cell(CW - 6, 4.5, sous_res, ln=True)
    pdf.set_xy(M, y_res + 13)

    # ══════════════════════════════════════════════════════
    # 6. CONDUITE À TENIR
    # ══════════════════════════════════════════════════════
    pdf.ln(8)
    sec("Conduite à tenir recommandée")

    if not detected:
        pdf.set_font(F, "", 8); pdf.set_text_color(*CLR_PRI)
        pdf.set_x(M)
        pdf.multi_cell(CW, 5.5, reco_txt)
    else:
        for d in detected:
            pdf.set_font(F, "B", 8.5); pdf.set_text_color(*CLR_PRI)
            pdf.set_x(M)
            pdf.cell(CW, 5.5, diseases_cfg[d]["label"], ln=True)
            line(pdf.get_y())
            pdf.ln(2)
            pdf.set_font(F, "", 8); pdf.set_text_color(*CLR_SEC)
            pdf.set_x(M)
            reco_content = RECO[d]["reco"] if len(detected) == 1 else RECO_SHORT[d]
            pdf.multi_cell(CW, 5.5, reco_content)
            pdf.ln(2)

    # ══════════════════════════════════════════════════════
    # PIED DE PAGE  (276 mm)
    # ══════════════════════════════════════════════════════
    pdf.set_y(272)
    line(272)
    pdf.set_xy(M, 274)
    pdf.set_font(F, "I", 5.5); pdf.set_text_color(*CLR_MUT)
    pdf.multi_cell(CW, 3.8,
        "Ce document ne constitue pas un acte médical et ne remplace pas un examen ophtalmologique. "
        "Pathologies non couvertes : rétinopathie hypertensive, occlusions vasculaires, décollement de rétine, "
        "neuropathies non glaucomateuses, pathologies du segment antérieur. "
        "Toute décision thérapeutique relève exclusivement d'un médecin qualifié.")
    line(285)
    pdf.set_xy(M, 286.5)
    pdf.set_font(F, "", 6); pdf.set_text_color(*CLR_MUT)
    pdf.cell(CW * 0.5, 3.5, f"Réf. {report_id}  ·  TriRetina {MODEL_VERSION} (EfficientNetB3)", align="L")
    pdf.cell(CW * 0.5, 3.5, f"Le {now.strftime('%d/%m/%Y à %H:%M')}", align="R")

    fname = f"TriRetina_{report_id}.pdf"
    return bytes(pdf.output()), fname

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-header">
        <h2>TriRetina</h2>
        <p>Analyse oculaire assistée par IA</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><h4>Informations patient</h4>', unsafe_allow_html=True)
    patient_ref = st.text_input("Référence patient", placeholder="ID, nom, numéro de dossier…")
    examiner    = st.text_input("Examinateur", placeholder="Nom de l'opérateur…")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><h4>Affichage</h4>', unsafe_allow_html=True)
    show_clahe = st.checkbox("Afficher image après CLAHE", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><h4>Modèle</h4>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-stat"><span>Architecture</span><span>EfficientNetB3</span></div>
    <div class="model-stat"><span>Paramètres</span><span>12M</span></div>
    <div class="model-stat"><span>Pré-entraînement</span><span>ImageNet</span></div>
    <div class="model-stat"><span>Dataset</span><span>ODIR · APTOS · RFMiD · ORIGA · G1020</span></div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><h4>Performance (AUC)</h4>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-stat">
        <span style="font-weight:700;color:#374151">Mean AUC</span>
        <span style="font-weight:800;color:#F07B45">0.9209</span>
    </div>
    <div class="model-stat">
        <span style="display:flex;align-items:center;gap:5px">
            <span style="width:8px;height:8px;border-radius:50%;background:#E74C3C;display:inline-block;flex-shrink:0"></span>Rétinopathie
        </span>
        <span style="color:#E74C3C;font-weight:700">0.9186</span>
    </div>
    <div class="model-stat">
        <span style="display:flex;align-items:center;gap:5px">
            <span style="width:8px;height:8px;border-radius:50%;background:#3B82F6;display:inline-block;flex-shrink:0"></span>Glaucome
        </span>
        <span style="color:#3B82F6;font-weight:700">0.9266</span>
    </div>
    <div class="model-stat">
        <span style="display:flex;align-items:center;gap:5px">
            <span style="width:8px;height:8px;border-radius:50%;background:#F5A623;display:inline-block;flex-shrink:0"></span>DMLA
        </span>
        <span style="color:#E8870A;font-weight:700">0.9175</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="triretina-header">
    <div>
        <p class="triretina-header-title">TriRetina · Analyse de Fond d'Œil</p>
        <p class="triretina-header-sub">
            Outil d'aide au dépistage ophtalmologique par analyse automatique d'images rétiniennes.
            Détecte la Rétinopathie Diabétique, le Glaucome et la DMLA à partir d'un cliché de fond d'œil.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Glissez votre image ici ou cliquez pour parcourir",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ── SANS IMAGE ───────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class="disease-cards-row">
        <div class="disease-card">
            <h4 style="color:#E74C3C">Rétinopathie Diabétique</h4>
            <span class="auc-badge" style="background:#F3F4F6;color:#6B7280">AUC 0.9186</span>
            <p style="font-size:0.7rem;color:#9CA3AF;margin:0.2rem 0 0.5rem 0;font-style:italic">Score de discrimination du modèle (max. 1.0)</p>
            <p>Lésions des capillaires rétiniens dues à l'hyperglycémie chronique.
            Première cause de cécité évitable chez l'adulte d'âge actif.</p>
        </div>
        <div class="disease-card">
            <h4 style="color:#3B82F6">Glaucome</h4>
            <span class="auc-badge" style="background:#F3F4F6;color:#6B7280">AUC 0.9266</span>
            <p style="font-size:0.7rem;color:#9CA3AF;margin:0.2rem 0 0.5rem 0;font-style:italic">Score de discrimination du modèle (max. 1.0)</p>
            <p>Neuropathie du nerf optique liée à une pression intraoculaire élevée.
            Évolution silencieuse, le dépistage précoce est essentiel.</p>
        </div>
        <div class="disease-card">
            <h4 style="color:#F5A623">DMLA</h4>
            <span class="auc-badge" style="background:#F3F4F6;color:#6B7280">AUC 0.9175</span>
            <p style="font-size:0.7rem;color:#9CA3AF;margin:0.2rem 0 0.5rem 0;font-style:italic">Score de discrimination du modèle (max. 1.0)</p>
            <p>Dégénérescence progressive de la macula liée à l'âge.
            Principale cause de malvoyance après 50 ans dans les pays développés.</p>
        </div>
    </div>

    <div class="disclaimer">
        <span><strong>Outil d'aide au diagnostic uniquement.</strong>
        Ce système ne remplace pas l'examen clinique ni l'avis d'un ophtalmologue qualifié.
        Les résultats obtenus sont à interpréter dans un contexte médical approprié.</span>
    </div>
    """, unsafe_allow_html=True)

# ── AVEC IMAGE ───────────────────────────────────────────────────────────────
else:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr   = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_clahe = apply_clahe(img_rgb)
    h_orig, w_orig = img_rgb.shape[:2]

    # Prédiction
    with st.spinner("Analyse en cours..."):
        model = load_model()
        if model is None:
            st.error(f"Modèle introuvable : {MODEL_PATH}")
            st.stop()
        t0 = time.time()
        probas = model.predict(preprocess(img_rgb), verbose=0)[0]
        inference_time = time.time() - t0

    keys     = list(DISEASES.keys())
    detected = [k for k, p in zip(keys, probas) if p >= DISEASES[k]["thresh"]]

    # ── Ligne 1 : Synthèse pleine largeur ──
    icon_style = "background:{bg};color:white;border-radius:50%;width:2.4rem;height:2.4rem;display:flex;align-items:center;justify-content:center;font-size:1rem;font-weight:800;flex-shrink:0;font-family:'Sora',sans-serif"
    tag_style  = "font-size:0.62rem;font-weight:700;letter-spacing:0.6px;text-transform:uppercase;padding:0.15rem 0.55rem;border-radius:4px;background:{tbg};color:{tc}"
    dur_style  = "font-size:0.68rem;color:#6B7280;margin-left:0.5rem"

    def remap(proba, thresh):
        return min((float(proba) / thresh) * 50, 100)

    if not detected:
        max_pct = max(remap(probas[i], DISEASES[keys[i]]["thresh"]) for i in range(len(keys)))
        st.markdown(f"""
        <div class="synth-ok">
            <div class="synth-icon" style="{icon_style.format(bg='#166534')}">N</div>
            <div class="synth-body">
                <div style="display:flex;align-items:center;margin-bottom:0.35rem">
                    <span style="{tag_style.format(tbg='rgba(22,101,52,0.12)',tc='#166534')}">Analyse terminée</span>
                    <span style="{dur_style}">Durée : {inference_time:.2f} s</span>
                </div>
                <h3 style="color:#166534;margin:0 0 0.2rem 0">Fond d'œil apparemment normal</h3>
                <p style="color:#166534;margin:0">Aucune pathologie détectée &nbsp;·&nbsp; score max. : <strong>{max_pct:.0f}%</strong></p>
            </div>
        </div>""", unsafe_allow_html=True)
    elif len(detected) == 1:
        lbl     = DISEASES[detected[0]]["label"]
        pct_det = remap(probas[keys.index(detected[0])], DISEASES[detected[0]]["thresh"])
        st.markdown(f"""
        <div class="synth-warn">
            <div class="synth-icon" style="{icon_style.format(bg='#854D0E')}">!</div>
            <div class="synth-body">
                <div style="display:flex;align-items:center;margin-bottom:0.35rem">
                    <span style="{tag_style.format(tbg='rgba(133,77,14,0.12)',tc='#854D0E')}">Analyse terminée</span>
                    <span style="{dur_style}">Durée : {inference_time:.2f} s</span>
                </div>
                <h3 style="color:#854D0E;margin:0 0 0.2rem 0">{lbl} détectée</h3>
                <p style="color:#854D0E;margin:0">Score : <strong>{pct_det:.0f}%</strong> &nbsp;·&nbsp; Consultation ophtalmologique recommandée.</p>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        lbls    = " + ".join(DISEASES[d]["label"] for d in detected)
        top_key = max(detected, key=lambda k: float(probas[keys.index(k)]))
        top_pct = remap(probas[keys.index(top_key)], DISEASES[top_key]["thresh"])
        st.markdown(f"""
        <div class="synth-alert">
            <div class="synth-icon" style="{icon_style.format(bg='#B91C1C')}">!</div>
            <div class="synth-body">
                <div style="display:flex;align-items:center;margin-bottom:0.35rem">
                    <span style="{tag_style.format(tbg='rgba(185,28,28,0.12)',tc='#B91C1C')}">Analyse terminée</span>
                    <span style="{dur_style}">Durée : {inference_time:.2f} s</span>
                </div>
                <h3 style="color:#B91C1C;margin:0 0 0.2rem 0">Pathologies multiples détectées</h3>
                <p style="color:#B91C1C;margin:0">{lbls} &nbsp;·&nbsp; probabilité principale : <strong>{top_pct:.0f}%</strong> &nbsp;·&nbsp; Consultation urgente.</p>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Ligne 2 : Images côte à côte ──
    st.markdown('<div class="section-title">Imagerie</div>', unsafe_allow_html=True)
    if show_clahe:
        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.markdown('<div style="width:345px"><p class="img-label">Image originale</p></div>', unsafe_allow_html=True)
            st.image(img_rgb, width=345)
            st.markdown(f'<div style="width:345px"><p class="img-meta">{w_orig} × {h_orig} px &nbsp;·&nbsp; sans traitement</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div style="width:345px"><p class="img-label">Après CLAHE</p></div>', unsafe_allow_html=True)
            st.image(img_clahe, width=345)
            st.markdown(f'<div style="width:345px"><p class="img-meta">{w_orig} × {h_orig} px &nbsp;·&nbsp; CLAHE appliqué</p></div>', unsafe_allow_html=True)
    else:
        st.image(img_rgb, width=345)
        st.markdown(f'<p class="img-meta">{w_orig} × {h_orig} px &nbsp;·&nbsp; sans traitement</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Ligne 3 : 3 cartes résultats côte à côte ──
    st.markdown('<div class="section-title">Résultats par pathologie</div>', unsafe_allow_html=True)
    cards = st.columns(3, gap="medium")
    for col, (key, proba) in zip(cards, zip(keys, probas)):
        cfg       = DISEASES[key]
        pct       = remap(proba, cfg["thresh"])
        color     = cfg["color"]
        positive  = float(proba) >= cfg["thresh"]
        uncertain = (not positive) and pct >= 37.5

        if positive:
            status_html = '<div style="background:#FEE2E2;color:#B91C1C;border-radius:8px;padding:0.5rem;font-weight:700;font-size:0.82rem;text-align:center;margin-top:0.8rem;letter-spacing:0.5px">DÉTECTÉ</div>'
        elif uncertain:
            status_html = '<div style="background:#FEF9C3;color:#854D0E;border-radius:8px;padding:0.5rem;font-weight:700;font-size:0.82rem;text-align:center;margin-top:0.8rem;letter-spacing:0.5px">À SURVEILLER</div>'
        else:
            status_html = '<div style="background:#DCFCE7;color:#166534;border-radius:8px;padding:0.5rem;font-weight:700;font-size:0.82rem;text-align:center;margin-top:0.8rem;letter-spacing:0.5px">NORMAL</div>'

        gauge_bg = f"conic-gradient({color} {pct:.1f}%, #E5E7EB 0%)"
        with col:
            st.markdown(f"""
            <div style="background:white;border-radius:14px;padding:1.4rem 1.2rem;
                        box-shadow:0 2px 12px rgba(0,0,0,0.07);border:1px solid #E5E7EB;text-align:center;">
                <p style="font-size:0.78rem;font-weight:700;color:#374151;text-transform:uppercase;
                           letter-spacing:0.5px;margin:0 0 1rem 0">{cfg['label']}</p>
                <div style="width:88px;height:88px;border-radius:50%;background:{gauge_bg};
                            display:flex;align-items:center;justify-content:center;margin:0 auto 0.8rem auto">
                    <div style="width:66px;height:66px;border-radius:50%;background:white;
                                display:flex;align-items:center;justify-content:center;">
                        <span style="font-size:1.25rem;font-weight:800;color:#111827">{pct:.0f}%</span>
                    </div>
                </div>
                <div style="background:#F3F4F6;border-radius:4px;height:4px;overflow:hidden;margin-bottom:0.4rem">
                    <div style="height:100%;width:{min(pct,100):.1f}%;background:#9CA3AF;border-radius:4px"></div>
                </div>
                <p style="font-size:0.68rem;color:#9CA3AF;margin:0">AUC {cfg['auc']}</p>
                {status_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Ligne 4 : Tableau récapitulatif ──
    rows_html = ""
    for key, proba in zip(keys, probas):
        cfg  = DISEASES[key]
        pct  = remap(proba, cfg["thresh"])
        pos  = float(proba) >= cfg["thresh"]
        decision = (
            '<span class="badge badge-pos">DÉTECTÉ</span>' if pos
            else '<span class="badge badge-neg">NORMAL</span>'
        )
        bar = f'<div style="height:4px;border-radius:2px;background:#F3F4F6;margin-top:0.3rem;overflow:hidden"><div style="height:100%;width:{min(pct,100):.1f}%;background:#9CA3AF;border-radius:2px"></div></div>'
        rows_html += f"""
        <tr>
            <td><strong style="color:#374151">{cfg['label']}</strong></td>
            <td><strong>{pct:.1f}%</strong>{bar}</td>
            <td>{decision}</td>
        </tr>"""

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">Récapitulatif</div>
        <table class="recap-table">
            <thead><tr>
                <th>Pathologie</th><th>Score</th><th>Décision</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        <div class="disclaimer" style="margin-top:1.2rem">
            <span><strong>Avertissement clinique :</strong> Outil d'aide au diagnostic expérimental.
            Les résultats ne constituent pas un diagnostic médical.
            Consultez toujours un ophtalmologue qualifié.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Bouton téléchargement PDF ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    pdf_bytes, pdf_name = generate_pdf_report(
        filename      = uploaded.name,
        img_rgb       = img_rgb,
        img_clahe     = img_clahe,
        probas        = probas,
        keys          = keys,
        detected      = detected,
        inference_time= inference_time,
        h_orig        = h_orig,
        w_orig        = w_orig,
        show_clahe    = show_clahe,
        diseases_cfg  = DISEASES,
        patient_ref   = patient_ref,
        examiner      = examiner,
    )
    st.download_button(
        label     = "Télécharger le rapport PDF",
        data      = pdf_bytes,
        file_name = pdf_name,
        mime      = "application/pdf",
    )
