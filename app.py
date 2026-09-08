"""
TaniLens — Diagnosa Penyakit Daun Tomat
Deployment model: CNN Custom
Nada Thahira Sosa — 2601 — MBC Lab Week 2
"""

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="TaniLens — Diagnosa Daun Tomat",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

IMG_SIZE = (224, 224)
MODEL_PATH = "models/custom_cnn.h5"

CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "nama": "Bercak Bakteri (Bacterial Spot)", "tingkat": "sedang",
        "penyebab": "Bakteri Xanthomonas spp.",
        "gejala": "Bercak kecil kehitaman/kecoklatan pada daun, sering dikelilingi lingkaran kuning (halo).",
        "saran": "Buang daun yang terinfeksi, hindari penyiraman dari atas, semprot bakterisida berbasis tembaga.",
    },
    "Tomato___Early_blight": {
        "nama": "Bercak Daun Awal (Early Blight)", "tingkat": "sedang",
        "penyebab": "Jamur Alternaria solani.",
        "gejala": "Bercak coklat berbentuk cincin konsentris (seperti target), dimulai dari daun tua bagian bawah.",
        "saran": "Rotasi tanaman, buang daun terinfeksi, gunakan fungisida (mis. klorotalonil) sesuai dosis.",
    },
    "Tomato___Late_blight": {
        "nama": "Bercak Daun Akhir (Late Blight)", "tingkat": "berat",
        "penyebab": "Oomycete Phytophthora infestans.",
        "gejala": "Bercak basah kehijauan-kehitaman yang cepat meluas, dapat menghancurkan tanaman dalam hitungan hari.",
        "saran": "Segera isolasi/musnahkan tanaman terinfeksi berat, semprot fungisida sistemik, perbaiki sirkulasi udara.",
    },
    "Tomato___Leaf_Mold": {
        "nama": "Jamur Daun (Leaf Mold)", "tingkat": "ringan",
        "penyebab": "Jamur Passalora fulva (dulu Fulvia fulva).",
        "gejala": "Bercak kuning pucat di permukaan atas daun, lapisan beludru zaitun di permukaan bawah.",
        "saran": "Kurangi kelembapan rumah kaca/greenhouse, tingkatkan ventilasi, gunakan fungisida bila perlu.",
    },
    "Tomato___Septoria_leaf_spot": {
        "nama": "Bercak Septoria (Septoria Leaf Spot)", "tingkat": "sedang",
        "penyebab": "Jamur Septoria lycopersici.",
        "gejala": "Bercak bulat kecil dengan pusat abu-abu dan tepi gelap, menyebar dari daun bawah ke atas.",
        "saran": "Mulsa tanah, hindari daun basah berlama-lama, buang daun terinfeksi, rotasi tanaman.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "nama": "Tungau Laba-laba (Two-Spotted Spider Mite)", "tingkat": "sedang",
        "penyebab": "Hama tungau Tetranychus urticae.",
        "gejala": "Bintik kuning kecil (stippling), jaring halus di bawah daun, daun mengering saat parah.",
        "saran": "Semprot air bertekanan pada bawah daun, gunakan akarisida/minyak neem, jaga kelembapan udara.",
    },
    "Tomato___Target_Spot": {
        "nama": "Bercak Target (Target Spot)", "tingkat": "sedang",
        "penyebab": "Jamur Corynespora cassiicola.",
        "gejala": "Bercak coklat dengan cincin konsentris mirip early blight, dapat menyebar ke batang dan buah.",
        "saran": "Rotasi tanaman, perbaiki sirkulasi udara, aplikasikan fungisida preventif.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "nama": "Virus Keriting Daun Kuning (TYLCV)", "tingkat": "berat",
        "penyebab": "Virus yang ditularkan oleh kutu kebul (whitefly).",
        "gejala": "Daun menguning, menggulung ke atas, tanaman kerdil dan pertumbuhan terhambat.",
        "saran": "Kendalikan populasi kutu kebul, cabut & musnahkan tanaman terinfeksi, gunakan varietas tahan virus.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "nama": "Virus Mosaik Tomat (ToMV)", "tingkat": "berat",
        "penyebab": "Tobamovirus, menular lewat kontak/alat pertanian.",
        "gejala": "Pola mosaik hijau muda-tua pada daun, daun keriting dan pertumbuhan terhambat.",
        "saran": "Sterilkan alat pertanian, cuci tangan sebelum menangani tanaman, musnahkan tanaman terinfeksi.",
    },
    "Tomato___healthy": {
        "nama": "Sehat", "tingkat": "sehat",
        "penyebab": "—",
        "gejala": "Tidak ditemukan tanda-tanda penyakit pada daun.",
        "saran": "Lanjutkan perawatan rutin: penyiraman cukup, pemupukan seimbang, pantau berkala.",
    },
}

VERSION_LOG = [
    {"versi": "v1.0", "tanggal": "2026-08-25", "perubahan": "Rilis awal: upload gambar, pilih model, tampilkan kelas prediksi & confidence dalam bentuk teks."},
    {"versi": "v2.0", "tanggal": "2026-08-29", "perubahan": "Tambah mode bandingkan 2 model side-by-side, bar chart confidence per kelas, kartu info penyakit berwarna."},
    {"versi": "v3.0 (final)", "tanggal": "2026-09-08", "perubahan": "Sederhanakan jadi 1 model (CNN Custom), alur single-page 3 langkah, dan toggle mode terang/gelap dengan kontras warna yang eksplisit."},
]

# ----------------------------------------------------------------------------
# TEMA (light & dark didefinisikan eksplisit, tidak bergantung tema bawaan Streamlit)
# ----------------------------------------------------------------------------
THEMES = {
    "light": {
        "bg": "#F2F5EC", "card": "#FFFFFF", "text": "#22301C", "muted": "#55654C",
        "border": "#DDE5CE", "primary": "#2F4B32", "primary_text": "#F2F5EC",
        "input_bg": "#FFFFFF", "track": "#E5EADA",
        "sehat": "#4F7A3D", "ringan": "#8AA24C", "sedang": "#C97A1F", "berat": "#B33A3A",
    },
    "dark": {
        "bg": "#161F13", "card": "#212B1C", "text": "#EAF0E1", "muted": "#AFC0A2",
        "border": "#39492F", "primary": "#7BB563", "primary_text": "#12190E",
        "input_bg": "#1B2417", "track": "#33422A",
        "sehat": "#7BB563", "ringan": "#AFC26A", "sedang": "#E0A64B", "berat": "#E07A6E",
    },
}

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

t = THEMES["dark"] if st.session_state.dark_mode else THEMES["light"]

# ----------------------------------------------------------------------------
# STYLING — semua warna eksplisit, tidak mewarisi warna default Streamlit
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Work+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, p, span, div {{ font-family: 'Work Sans', sans-serif; }}
    h1, h2, h3, .hero-title {{ font-family: 'Fraunces', serif; }}

    .stApp {{ background: {t['bg']} !important; }}
    .block-container {{ padding-top: 2rem; }}

    /* Paksa semua teks umum ikut warna tema kita */
    .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown, .stCaption, [data-testid="stCaptionContainer"] {{
        color: {t['text']} !important;
    }}

    .hero {{ text-align: center; padding: 0.6rem 1rem 0.4rem 1rem; }}
    .hero-title {{ font-size: 1.9rem; font-weight: 700; margin: 0.3rem 0; color: {t['primary']} !important; }}
    .hero-sub {{ font-size: 0.95rem; color: {t['muted']} !important; max-width: 480px; margin: 0 auto; line-height: 1.5; }}

    .steps {{ display: flex; justify-content: center; gap: 0.5rem; margin: 1.2rem 0 1.4rem 0; flex-wrap: wrap; }}
    .step {{ display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; color: {t['muted']} !important;
             padding: 0.4rem 0.9rem; border-radius: 20px; background: {t['track']}; }}
    .step.active {{ background: {t['primary']}; color: {t['primary_text']} !important; font-weight: 600; }}
    .step-num {{ width: 20px; height: 20px; border-radius: 50%; background: rgba(127,127,127,0.25);
                 display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; }}

    .card {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 14px;
             padding: 1.3rem 1.5rem; margin-bottom: 1rem; color: {t['text']} !important; }}
    .card b, .card p, .card li {{ color: {t['text']} !important; }}
    .tips-list {{ font-size: 0.88rem; margin: 0.4rem 0 0 0; padding-left: 1.1rem; }}
    .tips-list li {{ margin-bottom: 0.3rem; }}

    .result-card {{ border-radius: 14px; padding: 1.5rem 1.6rem; margin: 0.4rem 0 1rem 0; text-align: center; }}
    .result-label {{ font-size: 0.82rem; letter-spacing: 0.5px; opacity: 0.9; text-transform: uppercase; }}
    .result-name {{ font-family: 'Fraunces', serif; font-size: 1.7rem; font-weight: 600; margin: 0.3rem 0; }}
    .result-conf {{ font-size: 1rem; opacity: 0.95; }}

    .barrow {{ display: flex; align-items: center; margin: 0.35rem 0; gap: 0.6rem; }}
    .barrow-label {{ width: 230px; font-size: 0.8rem; color: {t['text']} !important; flex-shrink: 0; }}
    .barrow-track {{ flex: 1; background: {t['track']}; border-radius: 6px; height: 9px; overflow: hidden; }}
    .barrow-fill {{ height: 100%; border-radius: 6px; }}
    .barrow-pct {{ width: 44px; text-align: right; font-size: 0.78rem; color: {t['text']} !important; }}

    .version-row {{ border-left: 3px solid {t['primary']}; padding: 0.2rem 0 0.2rem 1rem; margin-bottom: 1rem; }}
    .version-tag {{ display: inline-block; background: {t['primary']}; color: {t['primary_text']} !important;
                     font-size: 0.76rem; padding: 0.12rem 0.6rem; border-radius: 20px; margin-right: 0.5rem; }}
    .version-date {{ color: {t['muted']} !important; font-size: 0.8rem; }}

    /* Komponen native Streamlit: uploader, tombol, expander */
    [data-testid="stFileUploaderDropzone"] {{
        background: {t['input_bg']} !important; border: 2px dashed {t['border']} !important; border-radius: 12px !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{ color: {t['text']} !important; }}
    [data-testid="stFileUploaderDropzone"] button {{
        background: {t['primary']} !important; color: {t['primary_text']} !important; border: none !important;
    }}
    .stButton button, .stDownloadButton button {{
        background: {t['primary']} !important; color: {t['primary_text']} !important;
        border: none !important; border-radius: 10px !important;
    }}
    [data-testid="stExpander"] {{ background: {t['card']} !important; border: 1px solid {t['border']} !important; border-radius: 12px !important; }}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: {t['text']} !important; }}
    [data-testid="stExpander"] p, [data-testid="stExpander"] li, [data-testid="stExpander"] span {{ color: {t['text']} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict(image: Image.Image):
    model = load_model()
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    batch = np.expand_dims(arr, axis=0)
    return model.predict(batch, verbose=0)[0]


def render_steps(active: int):
    labels = ["Upload Foto", "Proses", "Lihat Hasil"]
    html = '<div class="steps">'
    for i, label in enumerate(labels, start=1):
        cls = "active" if i == active else ""
        html += f'<div class="step {cls}"><span class="step-num">{i}</span>{label}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# TOGGLE TEMA
# ----------------------------------------------------------------------------
toggle_col1, toggle_col2 = st.columns([5, 1])
with toggle_col2:
    st.toggle("🌙", key="dark_mode", help="Mode gelap / terang")

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div style="font-size:2.2rem;">🍅</div>
        <div class="hero-title">TaniLens</div>
        <div class="hero-sub">Foto daun tomatmu, sistem langsung mendeteksi apakah sehat atau
        terkena salah satu dari 9 penyakit umum — lengkap dengan saran penanganannya.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# ----------------------------------------------------------------------------
# ALUR UTAMA: 3 LANGKAH
# ----------------------------------------------------------------------------
if uploaded is None:
    render_steps(1)
    st.markdown(
        """
        <div class="card">
        <b>Cara pakai (30 detik):</b>
        <ul class="tips-list">
            <li>Klik kotak di atas, atau tarik & lepas foto daun tomat.</li>
            <li>Ambil foto <b>close-up 1 daun</b>, cahaya cukup, latar polos.</li>
            <li>Hasil diagnosa & saran penanganan langsung muncul otomatis.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    image = Image.open(uploaded)
    render_steps(2)
    st.image(image, use_column_width=True, caption="Foto yang kamu unggah")

    with st.spinner("Sedang menganalisis daun..."):
        probs = predict(image)

    render_steps(3)

    top_idx = int(np.argmax(probs))
    top_class = CLASSES[top_idx]
    info = DISEASE_INFO[top_class]
    color = t[info["tingkat"]]

    st.markdown(
        f"""
        <div class="result-card" style="background:{color}; color:{t['primary_text'] if st.session_state.dark_mode else '#FFFFFF'};">
            <div class="result-label">Hasil Diagnosa</div>
            <div class="result-name">{info['nama']}</div>
            <div class="result-conf">Tingkat keyakinan model: {probs[top_idx]*100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="card">
            <p style="margin:0 0 0.4rem 0;"><b>Penyebab:</b> {info['penyebab']}</p>
            <p style="margin:0 0 0.4rem 0;"><b>Gejala:</b> {info['gejala']}</p>
            <p style="margin:0;"><b>Saran penanganan:</b> {info['saran']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Lihat rincian keyakinan untuk semua kelas"):
        order = np.argsort(probs)[::-1]
        bars_html = ""
        for i in order:
            pct = probs[i] * 100
            bars_html += f"""
            <div class="barrow">
                <div class="barrow-label">{DISEASE_INFO[CLASSES[i]]['nama']}</div>
                <div class="barrow-track"><div class="barrow-fill" style="width:{pct:.1f}%; background:{t[DISEASE_INFO[CLASSES[i]]['tingkat']]};"></div></div>
                <div class="barrow-pct">{pct:.1f}%</div>
            </div>
            """
        st.markdown(bars_html, unsafe_allow_html=True)

    if st.button("🔄 Coba foto lain", use_container_width=True):
        st.rerun()

# ----------------------------------------------------------------------------
# INFO TAMBAHAN
# ----------------------------------------------------------------------------
with st.expander("📖 Daftar penyakit yang bisa dikenali"):
    for cls in CLASSES:
        info = DISEASE_INFO[cls]
        color = t[info["tingkat"]]
        st.markdown(
            f"""
            <div class="card" style="border-left:4px solid {color}; margin-bottom:0.6rem;">
                <b>{info['nama']}</b>
                <span style="font-size:0.72rem; color:{color} !important; font-weight:600;"> · {info['tingkat'].upper()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("🕓 Riwayat versi aplikasi"):
    st.caption("Tambahkan screenshot versi sebelumnya di folder `docs/screenshots/` lalu tampilkan dengan `st.image()` di sini.")
    for v in VERSION_LOG:
        st.markdown(
            f"""
            <div class="version-row">
                <span class="version-tag">{v['versi']}</span>
                <span class="version-date">{v['tanggal']}</span>
                <p style="margin:0.3rem 0 0 0;">{v['perubahan']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption("TaniLens · Model: CNN Custom · Nada Thahira Sosa — 2601 · MBC Lab Week 2")
