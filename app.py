"""
TaniLens — Diagnosa Penyakit Daun Tomat
Deployment untuk 2 model: CNN Custom & MobileNetV2 Transfer Learning
Nada Thahira Sosa — 2601 — MBC Lab Week 2
"""

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="TaniLens — Diagnosa Daun Tomat",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMG_SIZE = (224, 224)

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
        "nama": "Bercak Bakteri (Bacterial Spot)",
        "tingkat": "sedang",
        "penyebab": "Bakteri Xanthomonas spp.",
        "gejala": "Bercak kecil kehitaman/kecoklatan pada daun, sering dikelilingi lingkaran kuning (halo).",
        "saran": "Buang daun yang terinfeksi, hindari penyiraman dari atas, semprot bakterisida berbasis tembaga.",
    },
    "Tomato___Early_blight": {
        "nama": "Bercak Daun Awal (Early Blight)",
        "tingkat": "sedang",
        "penyebab": "Jamur Alternaria solani.",
        "gejala": "Bercak coklat berbentuk cincin konsentris (seperti target), dimulai dari daun tua bagian bawah.",
        "saran": "Rotasi tanaman, buang daun terinfeksi, gunakan fungisida (mis. klorotalonil) sesuai dosis.",
    },
    "Tomato___Late_blight": {
        "nama": "Bercak Daun Akhir (Late Blight)",
        "tingkat": "berat",
        "penyebab": "Oomycete Phytophthora infestans.",
        "gejala": "Bercak basah kehijauan-kehitaman yang cepat meluas, dapat menghancurkan tanaman dalam hitungan hari.",
        "saran": "Segera isolasi/musnahkan tanaman terinfeksi berat, semprot fungisida sistemik, perbaiki sirkulasi udara.",
    },
    "Tomato___Leaf_Mold": {
        "nama": "Jamur Daun (Leaf Mold)",
        "tingkat": "ringan",
        "penyebab": "Jamur Passalora fulva (dulu Fulvia fulva).",
        "gejala": "Bercak kuning pucat di permukaan atas daun, lapisan beludru zaitun di permukaan bawah.",
        "saran": "Kurangi kelembapan rumah kaca/greenhouse, tingkatkan ventilasi, gunakan fungisida bila perlu.",
    },
    "Tomato___Septoria_leaf_spot": {
        "nama": "Bercak Septoria (Septoria Leaf Spot)",
        "tingkat": "sedang",
        "penyebab": "Jamur Septoria lycopersici.",
        "gejala": "Bercak bulat kecil dengan pusat abu-abu dan tepi gelap, menyebar dari daun bawah ke atas.",
        "saran": "Mulsa tanah, hindari daun basah berlama-lama, buang daun terinfeksi, rotasi tanaman.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "nama": "Tungau Laba-laba (Two-Spotted Spider Mite)",
        "tingkat": "sedang",
        "penyebab": "Hama tungau Tetranychus urticae.",
        "gejala": "Bintik kuning kecil (stippling), jaring halus di bawah daun, daun mengering saat parah.",
        "saran": "Semprot air bertekanan pada bawah daun, gunakan akarisida/minyak neem, jaga kelembapan udara.",
    },
    "Tomato___Target_Spot": {
        "nama": "Bercak Target (Target Spot)",
        "tingkat": "sedang",
        "penyebab": "Jamur Corynespora cassiicola.",
        "gejala": "Bercak coklat dengan cincin konsentris mirip early blight, dapat menyebar ke batang dan buah.",
        "saran": "Rotasi tanaman, perbaiki sirkulasi udara, aplikasikan fungisida preventif.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "nama": "Virus Keriting Daun Kuning (TYLCV)",
        "tingkat": "berat",
        "penyebab": "Virus yang ditularkan oleh kutu kebul (whitefly).",
        "gejala": "Daun menguning, menggulung ke atas, tanaman kerdil dan pertumbuhan terhambat.",
        "saran": "Kendalikan populasi kutu kebul, cabut & musnahkan tanaman terinfeksi, gunakan varietas tahan virus.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "nama": "Virus Mosaik Tomat (ToMV)",
        "tingkat": "berat",
        "penyebab": "Tobamovirus, menular lewat kontak/alat pertanian.",
        "gejala": "Pola mosaik hijau muda-tua pada daun, daun keriting dan pertumbuhan terhambat.",
        "saran": "Sterilkan alat pertanian, cuci tangan sebelum menangani tanaman, musnahkan tanaman terinfeksi.",
    },
    "Tomato___healthy": {
        "nama": "Sehat",
        "tingkat": "sehat",
        "penyebab": "—",
        "gejala": "Tidak ditemukan tanda-tanda penyakit pada daun.",
        "saran": "Lanjutkan perawatan rutin: penyiraman cukup, pemupukan seimbang, pantau berkala.",
    },
}

SEVERITY_COLOR = {
    "sehat": "#4F7A3D",
    "ringan": "#8AA24C",
    "sedang": "#D98E30",
    "berat": "#B33A3A",
}

MODEL_META = {
    "CNN Custom": {
        "file": "tomato_leaf_best_model.h5",
        "preprocess": "rescale_0_1",
        "accuracy": 0.941,
        "f1_macro": 0.9409,
        "deskripsi": "Arsitektur CNN 4-blok (32→64→128→256 filter) dilatih dari awal khusus untuk dataset ini.",
    }
}

VERSION_LOG = [
    {
        "versi": "v1.0",
        "tanggal": "2026-08-25",
        "perubahan": "Rilis awal: upload gambar, pilih 1 model, tampilkan kelas prediksi & skor confidence dalam bentuk teks.",
        "screenshot": None,
    },
    {
        "versi": "v2.0",
        "tanggal": "2026-08-29",
        "perubahan": "Tambah mode 'Bandingkan Kedua Model' side-by-side, bar chart confidence per kelas, kartu info penyakit & tingkat keparahan berwarna.",
        "screenshot": None,
    },
    {
        "versi": "v3.0 (final)",
        "tanggal": "2026-09-07",
        "perubahan": "Redesain UI penuh (tema greenhouse), tab navigasi, halaman metodologi & panduan penyakit, riwayat versi terintegrasi di aplikasi.",
        "screenshot": None,
    },
]

# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Work+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"]  { font-family: 'Work Sans', sans-serif; }
    h1, h2, h3, .hero-title { font-family: 'Fraunces', serif; }

    .stApp {
        background: #F2F5EC;
    }

    .hero {
        background: linear-gradient(135deg, #2F4B32 0%, #3D5F3F 100%);
        border-radius: 18px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.6rem;
        color: #F2F5EC;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        color: #F2F5EC;
    }
    .hero-sub {
        font-size: 0.98rem;
        color: #D8E3CE;
        max-width: 640px;
        line-height: 1.5;
    }

    .card {
        background: #FFFFFF;
        border: 1px solid #E1E7D8;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
    }

    .result-card {
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        color: #FFFFFF;
        margin-bottom: 1rem;
    }
    .result-label { font-size: 0.85rem; letter-spacing: 0.3px; opacity: 0.85; }
    .result-name { font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 600; margin: 0.15rem 0 0.3rem 0; }
    .result-conf { font-size: 1rem; opacity: 0.95; }

    .barrow { display: flex; align-items: center; margin: 0.35rem 0; gap: 0.6rem; }
    .barrow-label { width: 230px; font-size: 0.82rem; color: #2B3A28; flex-shrink: 0; }
    .barrow-track { flex: 1; background: #E5EADA; border-radius: 6px; height: 10px; overflow: hidden; }
    .barrow-fill { height: 100%; border-radius: 6px; }
    .barrow-pct { width: 48px; text-align: right; font-size: 0.8rem; color: #2B3A28; }

    .version-row {
        border-left: 3px solid #4F7A3D;
        padding: 0.2rem 0 0.2rem 1rem;
        margin-bottom: 1.1rem;
    }
    .version-tag {
        display: inline-block;
        background: #2F4B32;
        color: #F2F5EC;
        font-size: 0.78rem;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        margin-right: 0.5rem;
    }
    .version-date { color: #6B7A63; font-size: 0.82rem; }

    section[data-testid="stSidebar"] {
        background: #26361F;
    }
    section[data-testid="stSidebar"] * { color: #EDF1E4 !important; }
    section[data-testid="stSidebar"] hr { border-color: #445A38; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    return tf.keras.models.load_model(path)


def preprocess_image(image: Image.Image, method: str) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    if method == "rescale_0_1":
        arr = arr / 255.0
    elif method == "mobilenet":
        arr = mobilenet_preprocess(arr)
    return np.expand_dims(arr, axis=0)


def predict(model_name: str, image: Image.Image):
    meta = MODEL_META[model_name]
    model = load_model(meta["file"])
    batch = preprocess_image(image, meta["preprocess"])
    probs = model.predict(batch, verbose=0)[0]
    return probs


def render_result_card(model_name: str, probs: np.ndarray):
    top_idx = int(np.argmax(probs))
    top_class = CLASSES[top_idx]
    info = DISEASE_INFO[top_class]
    color = SEVERITY_COLOR[info["tingkat"]]

    st.markdown(
        f"""
        <div class="result-card" style="background:{color};">
            <div class="result-label">{model_name.upper()} · HASIL DIAGNOSA</div>
            <div class="result-name">{info['nama']}</div>
            <div class="result-conf">Keyakinan model: {probs[top_idx]*100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    order = np.argsort(probs)[::-1]
    bars_html = ""
    for i in order:
        pct = probs[i] * 100
        bars_html += f"""
        <div class="barrow">
            <div class="barrow-label">{DISEASE_INFO[CLASSES[i]]['nama']}</div>
            <div class="barrow-track"><div class="barrow-fill" style="width:{pct:.1f}%; background:{SEVERITY_COLOR[DISEASE_INFO[CLASSES[i]]['tingkat']]};"></div></div>
            <div class="barrow-pct">{pct:.1f}%</div>
        </div>
        """
    st.markdown(f'<div class="card">{bars_html}</div>', unsafe_allow_html=True)

    with st.expander(f"Detail: {info['nama']}"):
        st.markdown(f"**Penyebab:** {info['penyebab']}")
        st.markdown(f"**Gejala:** {info['gejala']}")
        st.markdown(f"**Saran penanganan:** {info['saran']}")


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🍅 TaniLens")
    st.caption("Diagnosa penyakit daun tomat berbasis CNN")
    st.markdown("---")
    mode = st.radio(
        "Mode diagnosa",
        ["1 model", "Bandingkan 2 model"],
        index=1,
    )
    if mode == "1 model":
        selected_model = st.selectbox("Pilih model", list(MODEL_META.keys()))
    st.markdown("---")
    st.markdown("**Tentang model**")
    for name, meta in MODEL_META.items():
        st.markdown(
            f"- **{name}** — akurasi {meta['accuracy']*100:.1f}%, F1-macro {meta['f1_macro']*100:.1f}%"
        )
    st.markdown("---")
    st.caption("Nada Thahira Sosa · 2601 · MBC Lab Week 2")

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Diagnosa Daun Tomat dalam Sekali Foto</div>
        <div class="hero-sub">
            Unggah foto daun tomat, dan bandingkan hasil prediksi dari dua model:
            CNN yang dilatih dari awal, dan MobileNetV2 hasil transfer learning.
            Cocok untuk membantu identifikasi awal 9 penyakit umum dan kondisi sehat.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_diagnosa, tab_metodologi, tab_panduan, tab_versi = st.tabs(
    ["🔍 Diagnosa", "🧪 Metodologi Model", "📖 Panduan Penyakit", "🕓 Riwayat Versi"]
)

# ----------------------------------------------------------------------------
# TAB 1: DIAGNOSA
# ----------------------------------------------------------------------------
with tab_diagnosa:
    col_upload, col_result = st.columns([1, 1.4], gap="large")

    with col_upload:
        st.markdown("#### Unggah foto daun")
        uploaded = st.file_uploader(
            "Format JPG/PNG, idealnya close-up satu daun dengan latar polos.",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, caption="Preview", use_container_width=True)
            run = st.button("Diagnosa Sekarang", type="primary", use_container_width=True)
        else:
            st.info("Belum ada gambar diunggah.")
            run = False

    with col_result:
        st.markdown("#### Hasil")
        if uploaded is not None and run:
            with st.spinner("Menganalisis daun..."):
                if mode == "1 model":
                    probs = predict(selected_model, image)
                    render_result_card(selected_model, probs)
                else:
                    sub_a, sub_b = st.columns(2)
                    names = list(MODEL_META.keys())
                    with sub_a:
                        probs_a = predict(names[0], image)
                        render_result_card(names[0], probs_a)
                    with sub_b:
                        probs_b = predict(names[1], image)
                        render_result_card(names[1], probs_b)
        elif uploaded is not None:
            st.write("Klik **Diagnosa Sekarang** untuk melihat hasil.")
        else:
            st.write("Hasil prediksi akan muncul di sini setelah gambar diunggah.")

# ----------------------------------------------------------------------------
# TAB 2: METODOLOGI
# ----------------------------------------------------------------------------
with tab_metodologi:
    st.markdown("#### Perbandingan Model")
    c1, c2 = st.columns(2)
    for col, name in zip([c1, c2], MODEL_META.keys()):
        meta = MODEL_META[name]
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <h4 style="margin-top:0;">{name}</h4>
                    <p style="color:#4B5A44; font-size:0.9rem;">{meta['deskripsi']}</p>
                    <p><b>Akurasi:</b> {meta['accuracy']*100:.1f}%<br>
                    <b>F1-Score (macro):</b> {meta['f1_macro']*100:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown(
        """
        <div class="card">
        <b>Catatan:</b> CNN Custom dilatih dari nol dan lebih menyesuaikan diri dengan karakteristik
        spesifik dataset daun tomat, sehingga akurasinya lebih tinggi pada eksperimen ini.
        MobileNetV2 Transfer menggunakan bobot ImageNet yang dibekukan (frozen), sehingga lebih ringan
        dan cepat dilatih namun kurang spesifik terhadap domain daun tomat tanpa fine-tuning lebih lanjut.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# TAB 3: PANDUAN PENYAKIT
# ----------------------------------------------------------------------------
with tab_panduan:
    st.markdown("#### 10 Kelas yang Dikenali Model")
    for cls in CLASSES:
        info = DISEASE_INFO[cls]
        color = SEVERITY_COLOR[info["tingkat"]]
        st.markdown(
            f"""
            <div class="card" style="border-left:4px solid {color};">
                <h5 style="margin:0 0 0.3rem 0;">{info['nama']} <span style="font-size:0.75rem; color:{color}; font-weight:600;">· {info['tingkat'].upper()}</span></h5>
                <p style="margin:0 0 0.2rem 0; font-size:0.88rem;"><b>Penyebab:</b> {info['penyebab']}</p>
                <p style="margin:0 0 0.2rem 0; font-size:0.88rem;"><b>Gejala:</b> {info['gejala']}</p>
                <p style="margin:0; font-size:0.88rem;"><b>Saran:</b> {info['saran']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# TAB 4: RIWAYAT VERSI
# ----------------------------------------------------------------------------
with tab_versi:
    st.markdown("#### Riwayat Pengembangan Aplikasi")
    st.caption("Tambahkan screenshot versi sebelumnya pada folder `docs/screenshots/` lalu tampilkan dengan `st.image()` di bawah tiap entri.")
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
        if v["screenshot"]:
            st.image(v["screenshot"], caption=f"Screenshot {v['versi']}", width=400)
