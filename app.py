# 1. Import streamlit and set page config
import streamlit as st

# Import other necessary libraries
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2, os, glob
from ultralytics import YOLO
import azure.cognitiveservices.speech as speechsdk
from PIL import Image
import tempfile
from audiorecorder import audiorecorder
from io import BytesIO
import openai
from pydub import AudioSegment
from dotenv import load_dotenv
import os
import html

# Load .env file
load_dotenv()

# ffmpeg path
AudioSegment.converter = r"ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

api_key = os.getenv("AZURE_OPEN_AI_API_KEY")
endpoint = os.getenv("AZURE_OPEN_AI_ENDPOINT")
api_version = os.getenv("AZURE_OPEN_AI_API_VERSION")

# Open AI
from openai import AzureOpenAI
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key
)

# Load YOLO model
model = YOLO("best.pt")

# Azure Speech Config
speech_api_key = os.getenv("AZURE_SPEECH_KEY")
speech_region = os.getenv("AZURE_SPEECH_REGION")
voice_name = os.getenv("AZURE_SPEECH_VOICE")

api_key = '7puy3olT86G8gwRYRCBwZnujbb3xm534rjJVSksdqdPeYksqmi7CJQQJ99BEACqBBLyXJ3w3AAAYACOGyRuq'
region = 'southeastasia'
speech_config = speechsdk.SpeechConfig(subscription=speech_api_key, region=speech_region)
speech_config.speech_synthesis_voice_name = 'id-ID-ArdiNeural'
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

# Mapping class index to letters A-Y (without J, Z)
def get_class_mapping():
    import string
    letters = [c for c in string.ascii_uppercase if c not in ['J', 'Z']]
    return {i: l for i, l in enumerate(letters)}

# Load example images from dataset
@st.cache_data
def load_label_images(dataset_folder="train"):
    label_map = {}
    label_folder = os.path.join(dataset_folder, "labels")
    image_folder = os.path.join(dataset_folder, "images")

    for label_file in glob.glob(os.path.join(label_folder, "*.txt")):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            if not lines: continue
            class_id = lines[0].split()[0]
            if class_id not in label_map:
                img_name = os.path.basename(label_file).replace(".txt", ".jpg")
                img_path = os.path.join(image_folder, img_name)
                if os.path.exists(img_path):
                    label_map[class_id] = img_path
    return label_map

# Webcam Real-Time Detection
class SignLanguageDetector(VideoTransformerBase):
    def __init__(self):
        self.detected_text = ""
        self.last_label = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror
        results = model.predict(img, imgsz=1080, conf=0.5, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = get_class_mapping().get(cls, "?")

            if label != self.last_label:
                self.detected_text += label
                self.last_label = label

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if len(results.boxes) == 0:
            self.last_label = ""

        return img

# 1. Import streamlit and set page config IMMEDIATELY
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2, os, glob
from ultralytics import YOLO
import azure.cognitiveservices.speech as speechsdk
from PIL import Image
import tempfile
from audiorecorder import audiorecorder
from io import BytesIO
import openai
from pydub import AudioSegment
from dotenv import load_dotenv
import os
import base64

# Load environment variables
load_dotenv()

# Set page config with modern theme
st.set_page_config(
    page_title="InSignia - Sign Language Translator",
    page_icon="👐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Modern CSS Styling ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")  # We'll create this file

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Beranda"
if 'sidebar_expanded' not in st.session_state:
    st.session_state.sidebar_expanded = False

# Sidebar Navigation
with st.sidebar:
    # Sidebar content
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0 3rem 0;">
        <h2 style="color: var(--primary); margin-bottom: 0.5rem;">InSignia</h2>
        <p style="color: var(--dark); opacity: 0.8; font-size: 0.9rem;">Bridging Communication Gaps</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected = st.radio(
        "Menu Navigasi",
        ["🏠 Beranda", "🌟 Fitur Unggulan", "📷 Deteksi", "📚 Kamus", "🎤 Speech to Visual", "💬 Chatbot"],
        key='nav_radio',
        index=["🏠 Beranda", "🌟 Fitur Unggulan", "📷 Deteksi", "📚 Kamus", "🎤 Speech to Visual", "💬 Chatbot"].index(st.session_state.current_page)
    )
    
    if selected != st.session_state.current_page:
        st.session_state.current_page = selected
        st.rerun()

# Page Content
def landing_page():
    # Hero Section
    with st.container():
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown("""
            <div class="hero">
                <h1 style="font-size: 10rem; margin-bottom: 1rem;"><span class="gradient-text">InSignia</span></h1>
                <h2 style="font-size: 2rem; font-weight: 600; margin-top: 0; color: var(--primary-dark);">
                    Inovasi Deteksi Bahasa Isyarat SIBI Berbasis AI
                </h2>
                <p style="font-size: 1.2rem; line-height: 1.8; margin-bottom: 2rem; opacity: 0.9;">
                    Platform AI inovatif untuk deteksi Bahasa Isyarat SIBI real-time, menjembatani komunikasi inklusif bagi penyandang disabilitas pendengaran dan masyarakat umum.
                </p>
                <div style="margin-top: 3rem;">
                    <span class="badge" style="background-color: var(--primary);">Inklusif</span>
                    <span class="badge" style="background-color: var(--success);">Real-time</span>
                    <span class="badge" style="background-color: var(--secondary);">Mudah Digunakan</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Mulai Sekarang", key="start_button_landing", use_container_width=True, type="primary"):
                st.session_state.current_page = "🌟 Fitur Unggulan"
                st.rerun()
        
        with col2:
            st.image("WhatsApp Image 2025-06-01 at 03.24.53_e4edf93b.jpg", 
                    use_container_width=True, caption="Komunikasi Tanpa Batas")

    st.divider()

    # About Section
    with st.container():
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0;">
            <h1 style="color: var(--primary-dark);">Tentang <span class="gradient-text">InSignia</span></h1>
            <p style="color: var(--dark); opacity: 0.8; font-size: 1.1rem; max-width: 800px; margin: 0 auto;">
                InSignia hadir sebagai solusi inovatif untuk mengatasi hambatan komunikasi yang dialami oleh lebih dari 2,5 juta penyandang disabilitas pendengaran di Indonesia. Dengan pendekatan berbasis teknologi, kami memberdayakan komunikasi inklusif antara pengguna bahasa isyarat dan masyarakat luas.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            <div style="background: white; border-radius: 16px; padding: 2rem; height: 100%;">
                <h3 style="color: var(--primary); margin-top: 0;">Visi Kami</h3>
                <p style="line-height: 1.8;">
                    <span style="font-size: 1.5rem; color: var(--secondary);">"</span>
                    Menciptakan dunia yang lebih inklusif di mana bahasa isyarat dapat dipahami oleh semua orang,
                    menghilangkan hambatan komunikasi antara penyandang disabilitas pendengaran dengan masyarakat umum.
                    <span style="font-size: 1.5rem; color: var(--secondary);">"</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: white; border-radius: 16px; padding: 2rem; height: 100%;">
                <h3 style="color: var(--primary); margin-top: 0;">Teknologi Kami</h3>
                <p style="line-height: 1.8;">
                    Menggunakan kombinasi <strong>YOLO Object Detection</strong> dan <strong>Azure Speech Recognition</strong>,
                    InSignia mampu menerjemahkan bahasa isyarat secara real-time dengan akurasi tinggi.
                </p>
                <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                    <span class="badge" style="background-color: var(--primary-light);">Computer Vision</span>
                    <span class="badge" style="background-color: var(--accent);">AI</span>
                    <span class="badge" style="background-color: var(--secondary);">NLP</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Value Proposition
    st.markdown("""
    <div style="text-align: center; margin: 4rem 0;">
        <h1 style="color: var(--primary-dark);">✨ Keunggulan <span class="gradient-text">InSignia</span></h1>
        <p style="color: var(--dark); opacity: 0.8; font-size: 1.1rem;">Solusi lengkap untuk komunikasi inklusif</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    features = [
        {"icon": "⚡", "title": "Real-time Detection", "desc": "Deteksi bahasa isyarat secara langsung dengan akurasi tinggi menggunakan model YOLO terbaru"},
        {"icon": "🤖", "title": "Kecerdasan Buatan", "desc": "Ditenagai oleh teknologi AI dari Azure untuk hasil terbaik"},
        {"icon": "👐", "title": "Multi-Modal", "desc": "Dukung input suara, teks, dan visual dalam satu platform"}
    ]
    
    for i, feature in enumerate(features):
        with cols[i]:
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <div style="font-size: 2.5rem; margin-bottom: 1.5rem; color: var(--primary);">{feature['icon']}</div>
                    <h3>{feature['title']}</h3>
                    <p style="line-height: 1.7; color: var(--dark); opacity: 0.9;">{feature['desc']}</p>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # How It Works
    st.markdown("""
    <div style="text-align: center; margin: 4rem 0;">
        <h1 style="color: var(--primary-dark);">🛠️ Cara Kerja <span class="gradient-text">InSignia</span></h1>
        <p style="color: var(--dark); opacity: 0.8; font-size: 1.1rem;">Proses sederhana untuk komunikasi yang kompleks</p>
    </div>
    """, unsafe_allow_html=True)

    steps = st.columns(3)
    with steps[0]:
        st.markdown("""
        <div class="card">
            <div style="background: rgba(67,97,238,0.1); width: 60px; height: 60px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">1</span>
            </div>
            <h3>Input</h3>
            <p style="line-height: 1.7; color: var(--dark); opacity: 0.9;">
                Masukkan bahasa isyarat melalui kamera atau suara melalui mikrofon
            </p>
        </div>
        """, unsafe_allow_html=True)
    with steps[1]:
        st.markdown("""
        <div class="card">
            <div style="background: rgba(67,97,238,0.1); width: 60px; height: 60px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">2</span>
            </div>
            <h3>Proses</h3>
            <p style="line-height: 1.7; color: var(--dark); opacity: 0.9;">
                Sistem AI kami akan mengenali dan menerjemahkan input Anda
            </p>
        </div>
        """, unsafe_allow_html=True)
    with steps[2]:
        st.markdown("""
        <div class="card">
            <div style="background: rgba(67,97,238,0.1); width: 60px; height: 60px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">3</span>
            </div>
            <h3>Output</h3>
            <p style="line-height: 1.7; color: var(--dark); opacity: 0.9;">
                Hasil terjemahan ditampilkan dalam format yang mudah dipahami
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Testimonials
    st.markdown("""
    <div style="text-align: center; margin: 4rem 0;">
        <h1 style="color: var(--primary-dark);">📢 Testimoni <span class="gradient-text">Pengguna</span></h1>
        <p style="color: var(--dark); opacity: 0.8; font-size: 1.1rem;">Apa kata mereka tentang InSignia</p>
    </div>
    """, unsafe_allow_html=True)

    testimonials = st.columns(3)
    testimonial_data = [
        {"name": "Guru SLB", "role": "Pengajar Bahasa Isyarat", "quote": "Membantu siswa saya belajar bahasa isyarat dengan lebih interaktif dan menyenangkan."},
        {"name": "Profesional", "role": "HRD Perusahaan", "quote": "Antarmuka yang intuitif sangat membantu komunikasi yang lancar dengan rekan tuli di lingkungan kerja."},
        {"name": "Tenaga Medis", "role": "Dokter Umum", "quote": "InSignia adalah alat revolusioner untuk memberikan layanan kesehatan yang lebih inklusif dan ramah disabilitas."}
    ]
    
    for i, testimonial in enumerate(testimonial_data):
        with testimonials[i]:
            st.markdown(f"""
            <div class="testimonial">
                <p style="font-style: italic; margin-bottom: 1.5rem; line-height: 1.7; color: var(--dark);">
                    "{testimonial['quote']}"
                </p>
                <p style="font-weight: bold; color: var(--primary); margin-bottom: 0.25rem;">{testimonial['name']}</p>
                <p style="font-size: 0.9rem; color: var(--dark); opacity: 0.7;">{testimonial['role']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Call to Action
    st.markdown("""
    <div style="text-align: center; margin: 4rem 0;">
        <h1 style="color: var(--primary-dark); margin-bottom: 1.5rem;">Siap Memulai Perjalanan <span class="gradient-text">Inklusivitas Anda?</span></h1>
        <p style="font-size: 1.2rem; color: var(--dark); opacity: 0.9; margin-bottom: 2rem; max-width: 700px; margin-left: auto; margin-right: auto;">
            Bergabunglah dengan ribuan pengguna yang telah merasakan manfaat InSignia dalam memecahkan hambatan komunikasi!
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Mulai Sekarang", key="start_button", use_container_width=True, type="primary"):
            st.session_state.current_page = "🌟 Fitur Unggulan"
            st.rerun()

def features_page():
    if st.button("← Kembali ke Beranda", key="back_features_page_top"):
        st.session_state.current_page = "🏠 Beranda"
        st.rerun()

    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h1 style="color: var(--primary-color);">🌟 Jelajahi InSignia</h1>
        <p style="color: var(--text-color); font-size: 1.1rem;">Pilih fitur yang ingin Anda gunakan:</p>
    </div>
    """, unsafe_allow_html=True)

    feature_data = [
        {"icon": "📷", "title": "Deteksi Real-time", "description": "Deteksi bahasa isyarat melalui kamera perangkat Anda secara langsung dan akurat.", "key": "open_detect", "page": "📷 Deteksi", "color": "#4361ee"},
        {"icon": "📚", "title": "Kamus SIBI", "description": "Pelajari bahasa isyarat dengan panduan lengkap, ilustrasi, dan contoh penggunaan.", "key": "open_dict", "page": "📚 Kamus", "color": "#3a0ca3"},
        {"icon": "🎤", "title": "Speech to Visual", "description": "Konversi ucapan Anda menjadi visual bahasa isyarat, mempermudah komunikasi dua arah.", "key": "open_speech", "page": "🎤 Speech to Visual", "color": "#7209b7"},
        {"icon": "💬", "title": "Chatbot InSignia", "description": "Dapatkan bantuan interaktif dan informasi seputar bahasa isyarat dari chatbot cerdas kami.", "key": "open_chat", "page": "💬 Chatbot", "color": "#f72585"}
    ]

    # Display features in a 2x2 grid
    rows = [feature_data[i:i + 2] for i in range(0, len(feature_data), 2)]
    for row in rows:
        cols = st.columns(len(row))
        for i, feature in enumerate(row):
            with cols[i]:
                with st.container():
                    st.markdown(f"""
                    <div class="card" style="border-top: 4px solid {feature['color']};">
                        <div style="font-size: 2.5rem; margin-bottom: 1.5rem; color: {feature['color']};">{feature['icon']}</div>
                        <h3>{feature['title']}</h3>
                        <p style="margin-bottom: 1.5rem;">{feature['description']}</p>
                        <div style="margin-top: auto;">
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Buka {feature['title'].split(' ')[0]}", key=feature['key'], use_container_width=True):
                        st.session_state.current_page = feature['page']
                        st.rerun()

    st.divider()
    if st.button("← Kembali ke Beranda", key="back_features_page_bottom", use_container_width=True):
        st.session_state.current_page = "🏠 Beranda"
        st.rerun()

def detection_page():
    if st.button("← Kembali ke Fitur Unggulan", key="back_from_detection"):
        st.session_state.current_page = "🌟 Fitur Unggulan"
        st.rerun()
    
    with st.container():
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: var(--primary-color);">📷 Deteksi Bahasa Isyarat</h1>
            <p style="color: var(--text-color);">Aktifkan kamera dan pastikan tangan terlihat jelas di area kamera.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("⚙️ Pengaturan Kamera", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                show_fps = st.checkbox("Tampilkan FPS", value=True)
            with col2:
                detection_threshold = st.slider("Threshold Deteksi", 0.1, 1.0, 0.5)
        
        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: var(--border-radius); padding: 1.5rem; margin-bottom: 2rem;">
            <h3 style="color: var(--primary-color); margin-top: 0;">Petunjuk Penggunaan:</h3>
            <ol style="padding-left: 1.5rem;">
                <li>Pastikan pencahayaan cukup</li>
                <li>Posisikan tangan di tengah frame</li>
                <li>Gunakan latar belakang yang kontras</li>
                <li>Buat gerakan jelas dan terpisah</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        ctx = webrtc_streamer(
            key="sign-lang",
            video_transformer_factory=SignLanguageDetector,
            media_stream_constraints={
                "video": {
                    "width": {"min": 640, "ideal": 1280, "max": 1920},
                    "height": {"min": 480, "ideal": 720, "max": 1080},
                    "frameRate": {"ideal": 30, "max": 60},
                    "facingMode": "user"
                },
                "audio": False
            }
        )
        
        if ctx.state.playing:
            st.info("🔍 Sedang mendeteksi bahasa isyarat...", icon="ℹ️")
        else:
            st.warning("⚠️ Kamera belum diaktifkan. Klik 'Start' untuk memulai deteksi.", icon="⚠️")

def dictionary_page():
    if st.button("← Kembali ke Fitur Unggulan", key="back_from_dictionary"):
        st.session_state.current_page = "🌟 Fitur Unggulan"
        st.rerun()
    
    with st.container():
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: var(--primary-color);">📚 Kamus SIBI</h1>
            <p style="color: var(--text-color);">Pelajari bahasa isyarat Indonesia dengan panduan visual lengkap.</p>
        </div>
        """, unsafe_allow_html=True)
        
        search_term = st.text_input("🔍 Cari huruf atau kata", placeholder="Masukkan huruf atau kata kunci")
        
        class_map = get_class_mapping()
        image_map = load_label_images()

        # Filter dictionary berdasarkan pencarian
        if search_term:
            filtered_items = [(k, v) for k, v in class_map.items() if search_term.lower() in v.lower()]
        else:
            filtered_items = list(class_map.items())

        # Display dictionary in a grid
        st.markdown("### Alfabet Bahasa Isyarat")
        st.markdown("Berikut adalah daftar lengkap huruf dalam Sistem Isyarat Bahasa Indonesia (SIBI):")
        
        cols_per_row = 5
        items = filtered_items
        num_items = len(items)
        num_rows = (num_items + cols_per_row - 1) // cols_per_row

        for r in range(num_rows):
            cols = st.columns(cols_per_row)
            for i in range(cols_per_row):
                idx = r * cols_per_row + i
                if idx < num_items:
                    class_id, letter = items[idx]
                    img_path = image_map.get(str(class_id))
                    with cols[i]:
                        with st.container():
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1rem; border-radius: var(--border-radius); background-color: white; box-shadow: var(--box-shadow);">
                                <h3 style="margin-top: 0; color: var(--primary-color);">{letter}</h3>
                            """, unsafe_allow_html=True)
                            if img_path:
                                st.image(img_path, use_container_width=True)
                            else:
                                st.markdown("*(Gambar tidak tersedia)*")
                            st.markdown("</div>", unsafe_allow_html=True)

def speech_page():
    if st.button("← Kembali ke Fitur Unggulan", key="back_from_speech"):
        st.session_state.current_page = "🌟 Fitur Unggulan"
        st.rerun()
    
    with st.container():
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: var(--primary-color);">🎤 Speech to Visual</h1>
            <p style="color: var(--text-color);">Konversi ucapan Anda menjadi visual bahasa isyarat.</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🎙️ Rekam Suara", "📂 Upload Audio"])
        
        with tab1:
            st.markdown("### 🔴 Rekam Suara Anda")
            audio = audiorecorder("🎙️ Mulai Rekam", "⏹️ Berhenti Rekam", key="recorder")
            
            detected_text = None
            audio_path = None
            
            if audio is not None and len(audio) > 0:
                if st.button("🔊 Proses Rekaman", key="process_recording", use_container_width=True):
                    with st.spinner("🔄 Memproses rekaman..."):                        
                        if isinstance(audio, AudioSegment):
                            audio_segment = audio
                        else:
                            audio_segment = AudioSegment.from_file(BytesIO(audio), format="webm")

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                            audio_segment.export(f.name, format="wav")
                            audio_path = f.name
                        detected_text = None  # reset detected_text untuk input baru
                        
                        if audio_path:
                            audio_input = speechsdk.AudioConfig(filename=audio_path)
                            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
                            result = recognizer.recognize_once()
                        
                            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                                detected_text = result.text.upper()
                                st.session_state.detected_text = detected_text
                                st.success(f"🗣️ Teks Terdeteksi: {detected_text}")
                            else:
                                st.error(f"Gagal mengenali suara. Reason: {result.reason}")
            
        with tab2:
            st.markdown("### 📂 Upload File Audio")
            uploaded_file = st.file_uploader("Pilih file audio (.wav/.mp3)", type=["wav", "mp3"], label_visibility="collapsed")
            
            if uploaded_file is not None:
                st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
                
                if st.button("🔊 Proses File Audio", key="process_upload", use_container_width=True):
                    with st.spinner("🔄 Memproses file audio..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                            f.write(uploaded_file.read())
                            audio_path = f.name
                        
                        audio_input = speechsdk.AudioConfig(filename=audio_path)
                        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
                        result = recognizer.recognize_once()
                        
                        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                            detected_text = result.text.upper()
                            st.session_state.detected_text = detected_text
                            st.success(f"🗣️ Teks Terdeteksi: {detected_text}")
                        else:
                            st.error(f"Gagal mengenali suara. Reason: {result.reason}")

        # Display Sign Language Visualization
        if 'detected_text' in st.session_state and st.session_state.detected_text:
            detected_text = st.session_state.detected_text
            class_map = get_class_mapping()
            inv_class_map = {v: str(k) for k, v in class_map.items()}
            image_map = load_label_images()

            st.markdown("### 👐 Visualisasi Bahasa Isyarat")
            st.markdown(f"**Teks:** {detected_text}")
            
            # Display in rows of 8 characters each
            chars = list(detected_text)
            num_rows = (len(chars) + 7) // 8  # 8 characters per row
            
            for row in range(num_rows):
                start_idx = row * 8
                end_idx = start_idx + 8
                current_chars = chars[start_idx:end_idx]
                
                cols = st.columns(len(current_chars))
                for i, char in enumerate(current_chars):
                    with cols[i]:
                        if char in inv_class_map:
                            class_id = inv_class_map[char]
                            img_path = image_map.get(class_id)
                            if img_path:
                                st.image(img_path, caption=char, width=100)
                            else:
                                st.markdown(f"**{char}**\n*(gambar tidak ditemukan)*")
                        else:
                            if char == " ":
                                st.markdown("**␣**\n*(spasi)*")
                            elif char == ".":
                                st.markdown("**.**\n*(titik)*")
                            elif char == ",":
                                st.markdown("**.**\n*(koma)*")
                            else:
                                st.markdown(f"**{char}**\n*(tidak valid)*")


def chatbot_page():
    if st.button("← Kembali ke Fitur Unggulan", key="back_from_chatbot"):
        st.session_state.current_page = "🌟 Fitur Unggulan"
        st.rerun()
    
    with st.container():
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: var(--primary-color);">💬 Chatbot InSignia</h1>
            <p style="color: var(--text-color);">Tanya apa saja tentang bahasa isyarat dan komunikasi inklusif.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "system", "content": "Kamu adalah asisten yang membantu menjelaskan bahasa isyarat untuk pengguna tunarungu dan teman bicara mereka. Gunakan bahasa yang ramah dan mudah dipahami."}
            ]

        if "trigger_chat" not in st.session_state:
            st.session_state.trigger_chat = False

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history[1:]:
                if msg["role"] == "user":
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-end; margin-bottom: 1.5rem;">
                            <div style="
                                background: linear-gradient(90deg, #0066cc, #3399ff);
                                color: white;
                                padding: 1rem 1.25rem;
                                border-radius: 18px 18px 0 18px;
                                max-width: 80%;
                                box-shadow: var(--box-shadow);
                            ">
                                <strong>Anda 🙋‍♀</strong><br>{html.escape(msg['content'])}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 1.5rem;">
                            <div style="
                                background-color: white;
                                color: var(--text-color);
                                padding: 1rem 1.25rem;
                                border-radius: 18px 18px 18px 0;
                                max-width: 80%;
                                box-shadow: var(--box-shadow);
                            ">
                                <strong>🤖 InSignia Bot</strong><br>{msg['content']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # Show processing indicator
        if st.session_state.trigger_chat:
            with chat_container:
                st.markdown(
                    """
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 1.5rem;">
                        <div style="
                            background-color: white;
                            color: var(--text-color);
                            padding: 1rem 1.25rem;
                            border-radius: 18px 18px 18px 0;
                            max-width: 80%;
                            box-shadow: var(--box-shadow);
                        ">
                            <strong>InSignia Bot</strong><br>
                            <div style="display: flex; align-items: center;">
                                <div style="margin-right: 0.5rem;">⚡</div>
                                <div>Sedang memproses jawaban...</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=st.session_state.chat_history,
                    temperature=0.7,
                    max_tokens=500
                )
                reply = response.choices[0].message.content
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Gagal memanggil chatbot: {e}")
                st.session_state.chat_history.append({"role": "assistant", "content": "Maaf, terjadi kesalahan saat memproses permintaan Anda."})
            finally:
                st.session_state.trigger_chat = False
                st.rerun()

        # Chat input form
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area("Tulis pertanyaanmu:", key="chat_input", label_visibility="collapsed", 
                                    placeholder="Tanyakan tentang bahasa isyarat...", height=100)
            cols = st.columns([1, 0.1])
            with cols[0]:
                send = st.form_submit_button("Kirim", use_container_width=True)
            with cols[1]:
                if st.form_submit_button("🔄", help="Clear chat", use_container_width=True):
                    st.session_state.chat_history = st.session_state.chat_history[:1]
                    st.rerun()

        if send and user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.trigger_chat = True
            st.rerun()

# Main page routing
if st.session_state.current_page == "🏠 Beranda":
    landing_page()
elif st.session_state.current_page == "🌟 Fitur Unggulan":
    features_page()
elif st.session_state.current_page == "📷 Deteksi":
    detection_page()
elif st.session_state.current_page == "📚 Kamus":
    dictionary_page()
elif st.session_state.current_page == "🎤 Speech to Visual":
    speech_page()
elif st.session_state.current_page == "💬 Chatbot":
    chatbot_page()