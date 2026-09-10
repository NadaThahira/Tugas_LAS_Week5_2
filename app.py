"""
TomaLeaf Dx — Smart Diagnosis for Tomato Leaf Diseases
Deployment model: CNN Custom
"""

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="TomaLeaf Dx — Smart Diagnosis for Tomato Leaf Diseases",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="expanded",
)

IMG_SIZE = (224, 224)
MODEL_PATH = "tomato_leaf_best_model.h5"
MIN_PLANT_COLOR_RATIO = 0.15

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
if "page" not in st.session_state:
    st.session_state.page = "diagnosis"

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
    .hero-title {{ font-size: 1.9rem; font-weight: 700; margin: 0.3rem 0 0.1rem 0; color: {t['primary']} !important; }}
    .hero-tagline {{ font-size: 0.8rem; font-weight: 600; letter-spacing: 0.4px; text-transform: uppercase; color: {t['muted']} !important; margin-bottom: 0.5rem; }}
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
    .tips-list {{ font-size: 0.88rem; margin: 0.4rem 0 0 0; padding-left: 1.3rem; }}
    .tips-list li {{ margin-bottom: 0.35rem; }}

    /* Spec sheet untuk konfigurasi model */
    .card-eyebrow {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: {t['primary']} !important; margin-bottom: 0.3rem; }}
    .model-name {{ font-family: 'Fraunces', serif; font-size: 1.3rem; font-weight: 600; color: {t['text']} !important; margin-bottom: 1rem; }}

    .spec-grid {{ display: grid; grid-template-columns: 1fr 1fr; row-gap: 1rem; column-gap: 1.5rem; }}
    .spec-cell-label {{ font-size: 0.72rem; font-weight: 700; color: {t['muted']} !important; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; }}
    .spec-cell-value {{ font-size: 0.95rem; font-weight: 600; color: {t['text']} !important; }}

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
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary div {{ font-weight: 700 !important; font-size: 1.05rem !important; }}
    [data-testid="stExpander"] p, [data-testid="stExpander"] li, [data-testid="stExpander"] span {{ color: {t['text']} !important; }}

    /* Perkuat kontras teks tombol (beberapa versi Streamlit bungkus label di elemen anak) */
    .stButton button p, .stButton button div, .stButton button span {{ color: {t['primary_text']} !important; font-weight: 600 !important; }}

    /* Area scroll mandiri supaya 1 konteks selalu utuh dalam 1 kotak, tidak terpotong scroll halaman */
    .scroll-box {{ max-height: 380px; overflow-y: auto; padding-right: 6px; }}

    /* Batasi lebar preview foto supaya tidak raksasa di layar besar */
    .preview-wrap img {{ border-radius: 12px; }}

    /* Sidebar navigasi */
    section[data-testid="stSidebar"] {{ background: {t['card']} !important; border-right: 1px solid {t['border']}; }}
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div.sidebar-brand {{ color: {t['text']} !important; }}
    .sidebar-brand {{ font-family: 'Fraunces', serif; font-size: 1.15rem; font-weight: 700; padding: 0.3rem 0 1rem 0; }}
    section[data-testid="stSidebar"] hr {{ border-color: {t['border']} !important; border-top: 1px solid {t['border']} !important; opacity: 1 !important; margin: 1rem 0 !important; }}

    /* Tombol collapse/expand sidebar (ikon panah) — beberapa selector fallback */
    [data-testid="stSidebarCollapseButton"] svg, [data-testid="collapsedControl"] svg,
    button[kind="header"] svg, [data-testid="baseButton-header"] svg {{
        fill: {t['text']} !important; color: {t['text']} !important;
    }}
    [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"],
    button[kind="header"], [data-testid="baseButton-header"] {{
        background: {t['card']} !important; border-radius: 8px !important;
    }}
    section[data-testid="stSidebar"] .stButton button {{
        background: transparent !important; color: {t['text']} !important; border: none !important;
        text-align: left !important; justify-content: flex-start !important; font-weight: 500 !important;
        padding: 0.5rem 0.7rem !important; border-radius: 8px !important; box-shadow: none !important;
    }}
    section[data-testid="stSidebar"] .stButton button p {{ color: {t['text']} !important; font-weight: 500 !important; text-align: left !important; }}
    section[data-testid="stSidebar"] .stButton button:hover {{ background: {t['track']} !important; }}
    section[data-testid="stSidebar"] .stButton button:hover p,
    section[data-testid="stSidebar"] .stButton button:hover div,
    section[data-testid="stSidebar"] .stButton button:hover span {{ color: {t['text']} !important; }}
    section[data-testid="stSidebar"] button[kind="primary"]:hover {{ background: {t['primary']} !important; }}
    section[data-testid="stSidebar"] button[kind="primary"]:hover p,
    section[data-testid="stSidebar"] button[kind="primary"]:hover div,
    section[data-testid="stSidebar"] button[kind="primary"]:hover span {{ color: {t['primary_text']} !important; }}
    section[data-testid="stSidebar"] button[kind="primary"] {{ background: {t['primary']} !important; }}
    section[data-testid="stSidebar"] button[kind="primary"] p,
    section[data-testid="stSidebar"] button[kind="primary"] div,
    section[data-testid="stSidebar"] button[kind="primary"] span {{ color: {t['primary_text']} !important; font-weight: 600 !important; }}

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


def is_plant_like(image: Image.Image) -> tuple[bool, float]:
    """Cek cepat berbasis warna: apakah gambar didominasi warna khas daun
    (hijau/kuning/coklat) sebelum dikirim ke model. Tidak menyentuh model sama sekali."""
    img = image.convert("RGB").resize((100, 100))
    arr = np.array(img).astype("float32")
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    green_mask = (g > r * 1.05) & (g > b * 1.05)
    brown_yellow_mask = (r > b * 1.15) & (g > b * 1.05) & (r > 60) & (r < 230)

    plant_mask = green_mask | brown_yellow_mask
    ratio = float(np.mean(plant_mask))
    return ratio >= MIN_PLANT_COLOR_RATIO, ratio


def render_steps(active: int):
    labels = ["Upload & Konfirmasi", "Lihat Hasil"]
    html = '<div class="steps">'
    for i, label in enumerate(labels, start=1):
        cls = "active" if i == active else ""
        html += f'<div class="step {cls}"><span class="step-num">{i}</span>{label}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# HALAMAN: DIAGNOSIS
# ----------------------------------------------------------------------------
def render_diagnosis():
    st.markdown(
        """
        <div class="hero">
            <div style="font-size:2.2rem;">🍅</div>
            <div class="hero-title">TomaLeaf Dx</div>
            <div class="hero-tagline">Smart Diagnosis for Tomato Leaf Diseases</div>
            <div class="hero-sub">Penasaran dengan kondisi daun tomatmu? Upload fotonya untuk mengetahui
            apakah sehat atau terkena salah satu dari 9 penyakit umum, lengkap dengan saran penanganannya.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "stage" not in st.session_state:
        st.session_state.stage = 1
    if "image_bytes" not in st.session_state:
        st.session_state.image_bytes = None
    if "probs" not in st.session_state:
        st.session_state.probs = None

    render_steps(st.session_state.stage)

    if st.session_state.stage == 1:
        uploaded = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded is None:
            st.markdown(
                """
                <div class="card">
                <b>Panduan Penggunaan</b>
                <ol class="tips-list">
                    <li>Klik kotak di atas, atau tarik &amp; lepas foto daun tomat.</li>
                    <li>Ambil foto <b>close-up 1 daun</b>, cahaya cukup, latar polos.</li>
                    <li>Konfirmasi gambar sebelum sistem melakukan klasifikasi dan menampilkan rekomendasi penanganan.</li>
                </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            image = Image.open(uploaded)

            col_l, col_mid, col_r = st.columns([1, 2, 1])
            with col_mid:
                st.markdown('<div class="preview-wrap">', unsafe_allow_html=True)
                st.image(image, use_column_width=True, caption="Preview foto")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                """
                <div class="card" style="text-align:center;">
                <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.5rem;">Konfirmasi Foto</div>
                <div>Periksa foto sebelum memulai diagnosis.<br>
                Pastikan daun terlihat jelas dan fokus. Jika ingin mengganti foto, klik tombol × di bagian atas.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Mulai Diagnosa →", type="primary", use_container_width=True):
                plant_like, ratio = is_plant_like(image)
                if not plant_like:
                    st.error(
                        f"Gambar tidak terdeteksi sebagai daun tomat (kecocokan warna khas daun hanya {ratio*100:.0f}%). "
                        "Coba unggah foto close-up satu daun dengan pencahayaan yang cukup."
                    )
                else:
                    with st.spinner("Sedang menganalisis daun..."):
                        probs = predict(image)
                    st.session_state.image_bytes = uploaded.getvalue()
                    st.session_state.probs = probs.tolist()
                    st.session_state.stage = 2
                    st.rerun()

    elif st.session_state.stage == 2:
        probs = np.array(st.session_state.probs)
        top_idx = int(np.argmax(probs))
        top_class = CLASSES[top_idx]
        info = DISEASE_INFO[top_class]
        color = t[info["tingkat"]]

        st.markdown(
            f"""
            <div class="result-card" style="background:{color}; color:#FFFFFF;">
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
            bars = []
            for i in order:
                pct = probs[i] * 100
                cls_i = CLASSES[i]
                bars.append(
                    f'<div class="barrow">'
                    f'<div class="barrow-label">{DISEASE_INFO[cls_i]["nama"]}</div>'
                    f'<div class="barrow-track"><div class="barrow-fill" style="width:{pct:.1f}%; background:{t[DISEASE_INFO[cls_i]["tingkat"]]};"></div></div>'
                    f'<div class="barrow-pct">{pct:.1f}%</div>'
                    f'</div>'
                )
            bars_html = '<div class="scroll-box">' + "".join(bars) + "</div>"
            st.markdown(bars_html, unsafe_allow_html=True)

        if st.button("↻ Coba Foto Lain", use_container_width=True):
            st.session_state.stage = 1
            st.session_state.image_bytes = None
            st.session_state.probs = None
            st.rerun()

    with st.expander("Penyakit yang dapat dikenali"):
        st.markdown(
            '<div style="font-size:0.88rem; color:{}; margin-bottom:0.8rem;">Daftar mencakup tingkat keparahan penyakit tersebut.</div>'.format(t["muted"]),
            unsafe_allow_html=True,
        )
        cards = []
        for cls in CLASSES:
            info = DISEASE_INFO[cls]
            color = t[info["tingkat"]]
            tag = "" if info["tingkat"] == "sehat" else f'<span style="font-size:0.72rem; color:{color} !important; font-weight:600;"> · {info["tingkat"].upper()}</span>'
            cards.append(
                f'<div class="card" style="border-left:4px solid {color}; margin-bottom:0.6rem;">'
                f'<b>{info["nama"]}</b>'
                f'{tag}'
                f'</div>'
            )
        list_html = '<div class="scroll-box">' + "".join(cards) + "</div>"
        st.markdown(list_html, unsafe_allow_html=True)

    st.caption("TomaLeaf Dx · Model: CNN Custom · Tomato Leaf Disease Diagnosis")


# ----------------------------------------------------------------------------
# HALAMAN: TENTANG APLIKASI
# ----------------------------------------------------------------------------
def render_about():
    st.markdown('<div class="hero-title" style="text-align:left; font-size:1.6rem; margin-bottom:1rem;">Tentang Aplikasi</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <b>Tentang TomaLeaf Dx</b>
            <p style="margin:0.5rem 0 0.4rem 0;">TomaLeaf Dx merupakan aplikasi untuk membantu mengidentifikasi kondisi
            dan penyakit pada daun tomat berdasarkan foto yang diunggah pengguna.</p>
            <p style="margin:0;">Aplikasi ini membantu proses identifikasi awal berdasarkan gambar, dan bukan
            pengganti pemeriksaan langsung oleh ahli tanaman.</p>
        </div>

        <div class="card">
            <div class="card-eyebrow">Konfigurasi Model</div>
            <div class="model-name">CNN Custom</div>

            <div class="spec-grid">
                <div>
                    <div class="spec-cell-label">Arsitektur</div>
                    <div class="spec-cell-value">CNN bertingkat (4 blok)</div>
                </div>
                <div>
                    <div class="spec-cell-label">Klasifikasi</div>
                    <div class="spec-cell-value">10 kelas</div>
                </div>
                <div>
                    <div class="spec-cell-label">Input</div>
                    <div class="spec-cell-value">224 × 224 RGB</div>
                </div>
                <div>
                    <div class="spec-cell-label">Optimizer</div>
                    <div class="spec-cell-value">Adam</div>
                </div>
                <div>
                    <div class="spec-cell-label">Output</div>
                    <div class="spec-cell-value">9 Penyakit + Sehat</div>
                </div>
                <div>
                    <div class="spec-cell-label">Pembelajaran</div>
                    <div class="spec-cell-value">Dilatih dari awal</div>
                </div>
            </div>
        </div>

        <div class="card">
            <b>Cakupan Diagnosis</b>
            <p style="margin:0.5rem 0 0.4rem 0; font-weight:600;">9 penyakit umum dan kondisi sehat</p>
            <p style="margin:0;">Model mengenali kelas-kelas yang telah ditentukan pada dataset. Daftar lengkapnya
            dapat dilihat pada halaman Diagnosis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <b>Informasi Aplikasi</b>
            <p style="margin:0.6rem 0 0 0;">Project: TomaLeaf Dx<br>Kategori: Tomato Leaf Disease Diagnosis<br>Model: CNN Custom</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# SIDEBAR — NAVIGASI UTAMA
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">TomaLeaf Dx</div>', unsafe_allow_html=True)

    if st.button("› Diagnosis", key="nav_diagnosis", use_container_width=True,
                 type="primary" if st.session_state.page == "diagnosis" else "secondary"):
        st.session_state.page = "diagnosis"
        st.rerun()

    if st.button("› Tentang Aplikasi", key="nav_about", use_container_width=True,
                 type="primary" if st.session_state.page == "about" else "secondary"):
        st.session_state.page = "about"
        st.rerun()

    st.markdown("---")
    theme_label = "Mode: Gelap" if st.session_state.dark_mode else "Mode: Terang"
    if st.button(f"◐ {theme_label}", key="theme_toggle_btn", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ----------------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------------
if st.session_state.page == "diagnosis":
    render_diagnosis()
else:
    render_about()
