# -*- coding: utf-8 -*-
"""
TF-IDF + Cosine 推薦（支援：情境選單 / 情境文字輸入）
【Train/Rec 分離版 + 匯出模型資產版】
- 訓練資料：new_songs_for_human_labeling.xlsx（含 human_label）
- 推薦候選歌庫：lyrics_with_spotify_meta_merged.xlsx（可無 human_label）
- 使用 SVM Emotion Model（linear SVM + probability=True）

新增功能：
A) 匯出可供 UI 使用的模型資產（joblib）：
   - vectorizer.joblib
   - svm_emotion_model.joblib
   - stop_words.joblib
B)（可選）匯出推薦歌庫每首歌的情緒機率：
   - all_songs_emotion_probabilities.csv

Usage:
    1) 兩個 xlsx 放同一層（或修改 main() 裡路徑）
    2) python new2_TFIDF_v4_export_assets.py
"""

import os
import re
import numpy as np
import pandas as pd

# NLP / TF-IDF
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Model: SVM (probability=True 才能 predict_proba)
from sklearn.svm import SVC

# Similarity
from sklearn.metrics.pairwise import cosine_similarity

# Export
import joblib


# ===== 0) 全域設定：情緒維度（固定順序）=====
EMOTIONS = ["Q1", "Q2", "Q3", "Q4"]  # Q1開心/興奮, Q2憤怒/緊張, Q3悲傷/痛苦, Q4放鬆/平靜


# ===== 1) Label 清理 =====
def extract_q_label(val):
    """把 human_label 萃取成 'Q1'~'Q4'，找不到就回 NaN。"""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    m = re.search(r"(Q[1-4])", s, re.IGNORECASE)
    return m.group(1).upper() if m else np.nan


# ===== 2) 讀取資料 =====
def load_train_songs(excel_path: str) -> pd.DataFrame:
    """
    讀取訓練資料（含 human_label）。
    預期欄位：artist, song, text, track_id, popularity, human_label
    """
    df = pd.read_excel(excel_path)
    df["text"] = df.get("text", "").fillna("").astype(str)

    if "human_label" not in df.columns:
        df["human_label"] = np.nan

    df["label_clean"] = df["human_label"].apply(extract_q_label)
    return df


def load_rec_songs(excel_path: str) -> pd.DataFrame:
    """
    讀取推薦候選歌庫（可無 human_label）。
    預期欄位至少要有：artist, song, text（track_id/popularity 可選）
    """
    df = pd.read_excel(excel_path)
    df["text"] = df.get("text", "").fillna("").astype(str)

    if "human_label" not in df.columns:
        df["human_label"] = np.nan

    df["label_clean"] = df["human_label"].apply(extract_q_label)
    return df


# ===== 3) 歌詞清理 =====
def get_stop_words() -> set:
    """
    NLTK 英文停用詞 + 歌詞常見口水詞/段落標記
    """
    sw = set(stopwords.words("english")).union({
        "oh", "yeah", "hey", "la", "da", "ooh", "ah", "na", "ha",
        "chorus", "verse", "intro", "outro", "bridge", "refrain",
        "im", "youre", "hes", "shes", "theyre", "aint", "gonna",
        "wanna", "gotta", "feat", "ft", "yo", "uh", "mmm"
    })
    return sw


def clean_lyrics(text: str, stop_words: set) -> str:
    """
    清理流程：
    1) 小寫
    2) 去掉 [Chorus] 這種括號段落
    3) 去標點、去數字
    4) tokenize
    5) 移除停用詞、長度<=1 的 token
    """
    text = text.lower()
    text = re.sub(r"\[.*?\]", " ", text)     # 去掉 [chorus] [verse] ...
    text = re.sub(r"[^\w\s]", " ", text)     # 去標點
    text = re.sub(r"\d+", " ", text)         # 去數字

    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)


# ===== 4) 建 TF-IDF（Train/Rec 分離，但共用同一個 vectorizer）=====
def build_vectorizer_and_matrices(
    df_train: pd.DataFrame,
    df_rec: pd.DataFrame,
    max_features: int = 5000,
    min_df: int = 5,
    max_df: float = 0.8,
    fit_on: str = "both"  # "both" or "train"
):
    stop_words = get_stop_words()

    df_train_clean = df_train.copy()
    df_rec_clean = df_rec.copy()

    df_train_clean["clean_text"] = df_train_clean["text"].apply(lambda t: clean_lyrics(t, stop_words))
    df_rec_clean["clean_text"] = df_rec_clean["text"].apply(lambda t: clean_lyrics(t, stop_words))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=list(stop_words)
    )

    if fit_on == "train":
        vectorizer.fit(df_train_clean["clean_text"])
    else:
        combined = pd.concat([df_train_clean["clean_text"], df_rec_clean["clean_text"]], ignore_index=True)
        vectorizer.fit(combined)

    X_train_all = vectorizer.transform(df_train_clean["clean_text"])
    X_rec = vectorizer.transform(df_rec_clean["clean_text"])
    return vectorizer, X_train_all, X_rec, df_train_clean, df_rec_clean, stop_words


# ===== 5) 訓練情緒模型：SVM =====
def train_emotion_model_svm(X, y: pd.Series) -> SVC:
    clf = SVC(kernel="linear", probability=True, random_state=42)
    clf.fit(X, y)
    print("clf.classes_ =", clf.classes_)
    return clf


def predict_emotion_scores(clf: SVC, X) -> np.ndarray:
    """
    回傳每首歌的情緒機率向量 shape = (n_songs, 4)
    順序對齊 EMOTIONS（Q1,Q2,Q3,Q4）
    """
    proba = clf.predict_proba(X)  # shape: (n, n_classes_in_training)
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}

    scores = np.zeros((proba.shape[0], len(EMOTIONS)), dtype=float)
    for j, emo in enumerate(EMOTIONS):
        scores[:, j] = proba[:, class_to_idx[emo]] if emo in class_to_idx else 0.0
    return scores


# ===== 6) 情境 -> 情緒規則 =====
SCENARIO_TO_EMOTION = {
    "深夜放鬆": [0.05, 0.05, 0.40, 0.50],
    "運動健身": [0.70, 0.30, 0.00, 0.00],
    "旅行": [0.55, 0.10, 0.05, 0.30],
    "失戀難過": [0.02, 0.25, 0.70, 0.03],
    "專注讀書": [0.05, 0.10, 0.05, 0.80],
    "派對狂歡": [0.90, 0.07, 0.01, 0.01],
    "通勤": [0.30, 0.00, 0.00, 0.70],
}


def get_scenario_vector(scenario: str) -> np.ndarray:
    if scenario not in SCENARIO_TO_EMOTION:
        raise ValueError(f"未知情境：{scenario}。請先在 SCENARIO_TO_EMOTION 裡新增規則。")

    q = np.array(SCENARIO_TO_EMOTION[scenario], dtype=float).reshape(1, -1)
    s = q.sum()
    if s <= 0:
        raise ValueError("情境向量加總不可為 0。")
    return q / s


# ===== 7) 情境文字輸入 -> 機率分佈 query 向量 =====
def get_user_text_proba_vector(user_text: str, vectorizer, clf: SVC, stop_words: set) -> np.ndarray:
    """回傳模型對使用者輸入的 Q1~Q4 機率分佈（對齊 EMOTIONS 順序）。"""
    clean = clean_lyrics(user_text, stop_words)
    X_user = vectorizer.transform([clean])

    proba = clf.predict_proba(X_user)[0]  # 順序 = clf.classes_
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}

    vec = np.zeros((1, len(EMOTIONS)), dtype=float)
    for j, emo in enumerate(EMOTIONS):
        vec[0, j] = proba[class_to_idx[emo]] if emo in class_to_idx else 0.0

    s = vec.sum()
    return vec / s if s > 0 else vec


# ===== 8) 推薦：給定 query 情緒向量 q（1,4）-> cosine -> Top-N =====
def recommend_top_n_by_query_vector(
    df: pd.DataFrame,
    emotion_scores: np.ndarray,
    q: np.ndarray,
    top_n: int = 20,
    max_per_artist: int = 3
) -> pd.DataFrame:
    sims = cosine_similarity(q, emotion_scores).flatten()

    out = df.copy()
    out["sim"] = sims

    if "popularity" not in out.columns:
        out["popularity"] = 0

    out = out.sort_values(["sim", "popularity"], ascending=[False, False])

    picked = []
    artist_count = {}

    for _, row in out.iterrows():
        a = row.get("artist", "")
        if artist_count.get(a, 0) >= max_per_artist:
            continue
        picked.append(row)
        artist_count[a] = artist_count.get(a, 0) + 1
        if len(picked) >= top_n:
            break

    cols = ["artist", "song", "track_id", "popularity", "sim"]
    cols = [c for c in cols if c in out.columns]
    return pd.DataFrame(picked)[cols].reset_index(drop=True)


def export_assets(vectorizer, clf, stop_words, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.joblib"))
    joblib.dump(clf, os.path.join(out_dir, "svm_emotion_model.joblib"))
    joblib.dump(list(stop_words), os.path.join(out_dir, "stop_words.joblib"))
    print(f"[OK] Exported assets to: {os.path.abspath(out_dir)}")


def export_emotion_probabilities_csv(df_rec_clean: pd.DataFrame, emotion_scores_rec: np.ndarray, out_csv: str):
    """
    將推薦歌庫每首歌的情緒機率（Q1~Q4）寫成 csv，供 UI 直接讀取。
    """
    out = df_rec_clean.copy()
    for j, emo in enumerate(EMOTIONS):
        out[f"{emo}_prob"] = emotion_scores_rec[:, j]
    # 建議保留一些UI會用到的欄位
    keep = [c for c in ["artist", "song", "track_id", "popularity", "text"] if c in out.columns]
    keep += [f"{emo}_prob" for emo in EMOTIONS]
    out[keep].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Exported emotion probabilities CSV: {os.path.abspath(out_csv)}")


# ===== 9) 主程式 =====
def main():
    train_excel = "new_songs_for_human_labeling.xlsx"
    rec_excel = "lyrics_with_spotify_meta_merged.xlsx"

    if not os.path.exists(train_excel):
        print(f"找不到訓練檔：{train_excel}")
        return
    if not os.path.exists(rec_excel):
        print(f"找不到推薦歌庫檔：{rec_excel}")
        return

    df_train = load_train_songs(train_excel)
    df_rec = load_rec_songs(rec_excel)

    vectorizer, X_train_all, X_rec, df_train_clean, df_rec_clean, stop_words = build_vectorizer_and_matrices(
        df_train, df_rec, fit_on="both"
    )

    # 訓練只用 train 裡有 label 的那部分
    if not df_train_clean["label_clean"].notna().any():
        print("訓練檔目前沒有有效 human_label（Q1~Q4），無法訓練 SVM。")
        print("請先在 new_songs_for_human_labeling.xlsx 裡標註一些資料再跑。")
        return

    labeled_mask = df_train_clean["label_clean"].notna().to_numpy()
    X_labeled = X_train_all[labeled_mask]
    y = df_train_clean.loc[labeled_mask, "label_clean"].astype(str)

    clf = train_emotion_model_svm(X_labeled, y)

    # 推薦歌庫：用 rec 的 X_rec 產生 emotion_scores_rec
    emotion_scores_rec = predict_emotion_scores(clf, X_rec)

    # ===== 新增：匯出模型資產（給 UI 用）=====
    export_assets(vectorizer, clf, stop_words, out_dir="model_assets")

    # ===== 新增：匯出歌庫情緒機率（如果你已經有這份csv可跳過）=====
    export_emotion_probabilities_csv(df_rec_clean, emotion_scores_rec, "all_songs_emotion_probabilities.csv")

    # ======（可留可刪）原本CLI互動推薦 ======
    print("\n你想用哪種方式推薦？")
    print("1. 選情境（深夜放鬆/運動健身...）")
    print("2. 輸入情境文字（例如：I want to go to exercise!）")

    while True:
        mode = input("輸入 1 或 2：").strip()
        if mode in ["1", "2"]:
            break
        print("輸入無效，請輸入 1 或 2。")

    if mode == "1":
        scenarios = list(SCENARIO_TO_EMOTION.keys())
        print("\n請選擇情境：")
        for i, s in enumerate(scenarios, start=1):
            print(f"{i}. {s}")

        while True:
            choice = input("輸入選項編號：").strip()
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(scenarios):
                    scenario = scenarios[idx - 1]
                    break
            print("輸入無效，請輸入清單中的編號。")

        q = get_scenario_vector(scenario)
        rec = recommend_top_n_by_query_vector(df_rec_clean, emotion_scores_rec, q, top_n=15, max_per_artist=2)
        print(f"\n=== 情境：{scenario} 的 Top-N 推薦（來源：{rec_excel}）===")
        print(rec)

    else:
        user_text_raw = input("\n請輸入情境文字（例如：I want to go to exercise!）：\n> ").strip()
        user_text = re.sub(r"^(scenario:|context:)\s*", "", user_text_raw, flags=re.IGNORECASE)

        q = get_user_text_proba_vector(user_text, vectorizer, clf, stop_words)
        pred = EMOTIONS[int(np.argmax(q))]

        print(f"\n=== 文字輸入的機率分佈向量（Q1,Q2,Q3,Q4），模型主預測 {pred} ===")
        print(q.flatten())

        rec = recommend_top_n_by_query_vector(df_rec_clean, emotion_scores_rec, q, top_n=15, max_per_artist=2)
        print(f"\n=== 文字輸入的 Top-N 推薦（來源：{rec_excel}）===")
        print(rec)


if __name__ == "__main__":
    main()
