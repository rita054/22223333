import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
import nltk

from sklearn.metrics.pairwise import cosine_similarity

# ===== NLTK =====
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

from nltk.tokenize import word_tokenize

# ===== 頁面設定 =====
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# ===== CSS（保留你原本的）=====
st.markdown("""
<style>
    .main-title {font-size: 2.5em;font-weight: 700;color: #1E3A8A;text-align: center;margin-bottom: 0.5em;padding-bottom: 10px;border-bottom: 2px solid #E5E7EB;}
    .sub-title {font-size: 1.1em;color: #6B7280;text-align: center;margin-bottom: 2em;font-weight: 300;}
    .scenario-container {background-color: #F9FAFB;padding: 20px;border-radius: 10px;margin-bottom: 2em;}
    .song-item {padding: 12px 16px;margin: 6px 0;background: white;border-radius: 8px;border: 1px solid #E5E7EB;transition: all 0.2s ease;font-size: 1em;}
    .song-item:hover {border-color: #3B82F6;box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);transform: translateY(-1px);}
    .song-title {font-weight: 600;color: #111827;}
    .song-artist {color: #6B7280;font-size: 0.95em;}
    .stButton > button {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);color: white;border: none;padding: 12px 28px;border-radius: 8px;font-weight: 500;font-size: 1em;transition: all 0.2s ease;width: 100%;}
    .stButton > button:hover {background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);transform: translateY(-1px);box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);}
    .stats-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);color: white;padding: 20px;border-radius: 10px;text-align: center;margin: 10px 0;}
    .stats-number {font-size: 2em;font-weight: 700;margin: 10px 0;}
    .stats-label {font-size: 0.9em;opacity: 0.9;}
</style>
""", unsafe_allow_html=True)

# ===== 標題 =====
st.markdown('<div class="main-title">Music Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Select a scenario or describe your mood to get personalized song recommendations</div>', unsafe_allow_html=True)

# ===== 固定設定 =====
EMOTIONS = ["Q1", "Q2", "Q3", "Q4"]

SCENARIO_TO_EMOTION = {
    "深夜放鬆": [0.05, 0.05, 0.40, 0.50],
    "運動健身": [0.70, 0.30, 0.00, 0.00],
    "旅行": [0.55, 0.10, 0.05, 0.30],
    "失戀難過": [0.02, 0.25, 0.70, 0.03],
    "專注讀書": [0.05, 0.10, 0.05, 0.80],
    "派對狂歡": [0.90, 0.07, 0.01, 0.01],
    "通勤": [0.30, 0.00, 0.00, 0.70],
}

SCENARIO_TRANSLATIONS = {
    "深夜放鬆": "Late Night Relax",
    "運動健身": "Workout & Exercise",
    "旅行": "Road Trip",
    "失戀難過": "Heartbreak",
    "專注讀書": "Study Focus",
    "派對狂歡": "Party & Celebration",
    "通勤": "Daily Commute"
}

# ===== 工具函數 =====
def parse_prob(v) -> float:
    """把csv裡的機率欄統一轉成0~1浮點數（支援 '20.78%' 或 0.2078 或 20.78）"""
    if pd.isna(v):
        return 0.0
    s = str(v).strip()
    if s.endswith("%"):
        try:
            return float(s.replace("%", "").strip()) / 100.0
        except:
            return 0.0
    try:
        x = float(s)
        # 如果有人存成 0~100 的數字（不是百分比字串），做保護
        if x > 1.0:
            return x / 100.0
        return x
    except:
        return 0.0

def get_scenario_vector(scenario: str) -> np.ndarray:
    q = np.array(SCENARIO_TO_EMOTION[scenario], dtype=float).reshape(1, -1)
    s = q.sum()
    return q / s if s > 0 else q

def clean_lyrics(text: str, stop_words: set) -> str:
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)

def get_text_emotion_vector_fallback(user_text: str) -> np.ndarray:
    """沒有模型時的備援（保底），避免整個功能壞掉"""
    text = str(user_text).lower()
    happy_words = ['happy', 'joy', 'love', 'smile', 'fun', 'party', 'dance', 'celebrate']
    angry_words = ['angry', 'hate', 'fight', 'rage', 'mad', 'furious']
    sad_words   = ['sad', 'cry', 'tear', 'hurt', 'pain', 'alone', 'miss', 'goodbye']
    calm_words  = ['calm', 'peace', 'quiet', 'rest', 'sleep', 'dream', 'relax']
    scores = [
        sum(text.count(w) for w in happy_words),
        sum(text.count(w) for w in angry_words),
        sum(text.count(w) for w in sad_words),
        sum(text.count(w) for w in calm_words),
    ]
    total = sum(scores) + 1e-6
    return (np.array([scores], dtype=float) / total)

@st.cache_resource
def load_model_assets():
    """B方案：直接載入你已經export的joblib，不在UI訓練"""
    vec_path = "model_assets/vectorizer.joblib"
    clf_path = "model_assets/svm_emotion_model.joblib"
    sw_path  = "model_assets/stop_words.joblib"

    if not (os.path.exists(vec_path) and os.path.exists(clf_path) and os.path.exists(sw_path)):
        return None, None, None, "找不到 model_assets 裡的joblib，文字輸入模式將使用備援方法"

    try:
        vectorizer = joblib.load(vec_path)
        clf = joblib.load(clf_path)
        stop_words = set(joblib.load(sw_path))
        return vectorizer, clf, stop_words, None
    except Exception as e:
        return None, None, None, f"載入模型失敗：{e}"

@st.cache_data
def load_song_probabilities():
    data_file = "all_songs_emotion_probabilities.csv"
    if not os.path.exists(data_file):
        return None, f"找不到資料檔：{data_file}"
    try:
        df = pd.read_csv(data_file)

        needed = ["artist", "song", "Q1_prob", "Q2_prob", "Q3_prob", "Q4_prob"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            return None, f"資料檔缺少欄位：{missing}"

        # ✅超強健：把各種格式都轉成0~1 float
        def to_float01(series: pd.Series) -> pd.Series:
            s = series.astype(str).str.strip()
            # 把百分比字串轉掉
            is_pct = s.str.endswith("%")
            s2 = s.str.replace("%", "", regex=False)
            # 轉數字（轉不了變NaN）
            x = pd.to_numeric(s2, errors="coerce")
            # 如果原本是百分比字串 -> 除以100
            x = np.where(is_pct, x / 100.0, x)
            x = pd.Series(x)

            # 可能有人存成 0~100 但沒%（例如 51.1）
            x = x.where(x <= 1.0, x / 100.0)

            # NaN補0
            return x.fillna(0.0)

        for c in ["Q1_prob", "Q2_prob", "Q3_prob", "Q4_prob"]:
            df[c] = to_float01(df[c])

        if "popularity" not in df.columns:
            df["popularity"] = 0

        # ✅保護機制：如果轉完幾乎全是0，直接回報（避免similarity全0）
        mat = df[["Q1_prob", "Q2_prob", "Q3_prob", "Q4_prob"]].to_numpy(dtype=float)
        row_sum = mat.sum(axis=1)
        zero_ratio = float((row_sum == 0).mean())
        if zero_ratio > 0.1:  # 超過10%歌曲是全0向量，代表資料格式很可能不對
            return None, f"情緒機率欄位有大量無法轉成數字（全0比例={zero_ratio:.1%}）。請檢查csv的Q1_prob~Q4_prob格式。"

        return df, None
    except Exception as e:
        return None, f"讀取資料失敗：{e}"


def user_text_to_q_vector(user_text: str, vectorizer, clf, stop_words: set) -> np.ndarray:
    if vectorizer is None or clf is None or stop_words is None:
        return get_text_emotion_vector_fallback(user_text)

    clean = clean_lyrics(user_text, stop_words)
    if not clean.strip():
        return get_text_emotion_vector_fallback(user_text)

    X_user = vectorizer.transform([clean])
    proba = clf.predict_proba(X_user)[0]
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}

    q = np.zeros((1, 4), dtype=float)
    for j, emo in enumerate(EMOTIONS):
        q[0, j] = proba[class_to_idx[emo]] if emo in class_to_idx else 0.0
    s = q.sum()
    return q / s if s > 0 else q

def recommend_songs(df: pd.DataFrame, q: np.ndarray, top_n=15, max_per_artist=2) -> pd.DataFrame:
    emotion_mat = df[["Q1_prob","Q2_prob","Q3_prob","Q4_prob"]].to_numpy(dtype=float)
    sims = cosine_similarity(q, emotion_mat).flatten()

    out = df.copy()
    out["similarity"] = sims
    out = out.sort_values(["similarity", "popularity"], ascending=[False, False])

    picked = []
    cnt = {}
    for _, row in out.iterrows():
        a = str(row.get("artist", "Unknown"))
        if cnt.get(a, 0) >= max_per_artist:
            continue
        picked.append(row)
        cnt[a] = cnt.get(a, 0) + 1
        if len(picked) >= top_n:
            break

    res = pd.DataFrame(picked).reset_index(drop=True)
    keep_cols = [c for c in ["artist", "song", "track_id", "popularity", "similarity"] if c in res.columns]
    return res[keep_cols]

# ===== 載入資源 =====
vectorizer, clf, stop_words, model_err = load_model_assets()
df, data_err = load_song_probabilities()

# ===== UI 主區 =====
st.markdown('<div class="scenario-container">', unsafe_allow_html=True)

mode = st.radio(
    "Choose Input Method",
    ["Select Scenario", "Describe Your Mood"],
    horizontal=True,
    index=0
)

user_text = ""
selected_key = None

if mode == "Select Scenario":
    col1, col2 = st.columns([3, 1])
    with col1:
        scenarios = list(SCENARIO_TRANSLATIONS.keys())
        selected_scenario_cn = st.selectbox(
            "Select Listening Scenario",
            scenarios,
            index=0,
            format_func=lambda x: SCENARIO_TRANSLATIONS[x]
        )
        selected_key = selected_scenario_cn
    with col2:
        st.write("")
else:
    col1, col2 = st.columns([3, 1])
    with col1:
        user_text = st.text_area(
            "Describe your mood or situation",
            placeholder="e.g., I want to go to exercise!",
            height=100
        )
        selected_key = "custom"


    with col2:
        st.write("")

st.markdown('</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    get_recommendations = st.button(
        "Get Song Recommendations",
        type="primary",
        use_container_width=True
    )

st.markdown("---")

# ===== 推薦顯示 =====
if get_recommendations:
    if data_err or df is None:
        st.error(data_err or "資料尚未載入")
        st.stop()

    if mode == "Select Scenario":
        q = get_scenario_vector(selected_key)
        display_scenario = SCENARIO_TRANSLATIONS.get(selected_key, selected_key)
    else:
        q = user_text_to_q_vector(user_text, vectorizer, clf, stop_words)
        pred_emotion = EMOTIONS[int(np.argmax(q.flatten()))]
        emotion_names = {
            "Q1": "Happy/Excited",
            "Q2": "Angry/Tense",
            "Q3": "Sad/Painful",
            "Q4": "Relaxed/Calm"
        }
        display_scenario = f"Your Mood"


    rec = recommend_songs(df, q, top_n=15, max_per_artist=2)

    st.markdown(f"### Recommended Songs for **{display_scenario}**")
    st.caption(f"Showing {len(rec)} personalized recommendations")

    # 統計卡片
    if mode == "Select Scenario":
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(rec)}</div>
                <div class="stats-label">Total Songs</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            unique_artists = rec["artist"].nunique() if "artist" in rec.columns else 0
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{unique_artists}</div>
                <div class="stats-label">Unique Artists</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            word = display_scenario.split()[0] if display_scenario else "-"
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{word}</div>
                <div class="stats-label">Scenario</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(rec)}</div>
                <div class="stats-label">Total Songs</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            unique_artists = rec["artist"].nunique() if "artist" in rec.columns else 0
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{unique_artists}</div>
                <div class="stats-label">Unique Artists</div>
            </div>
            """, unsafe_allow_html=True)

    # 歌曲列表
    st.markdown("---")
    st.markdown("### Song List")

    for _, row in rec.iterrows():
        song = str(row.get("song", "Unknown Song"))
        artist = str(row.get("artist", "Unknown Artist"))
        sim = float(row.get("similarity", 0.0))

        st.markdown(f"""
        <div class="song-item">
            <span class="song-title">{song}</span>
            <span style="color: #9CA3AF; margin: 0 8px">|</span>
            <span class="song-artist">{artist}</span>
        </div>
        """, unsafe_allow_html=True)

# ===== Sidebar =====
st.sidebar.markdown("## About")
st.sidebar.markdown("""
This system recommends songs based on emotional analysis of lyrics.

**Features:**
- Emotion-based matching
- Personalized recommendations
- Two input methods: Scenario selection or text description

**How to use:**
- Choose input method: Select scenario or describe your mood
- Choose from predefined scenarios or type your own description
- Click "Get Song Recommendations"
- View your personalized playlist

**Available scenarios:**
- Late Night Relax
- Workout & Exercise
- Road Trip
- Study Focus
- Heartbreak
- Party & Celebration
- Daily Commute
""")
st.sidebar.markdown("---")