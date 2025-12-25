# -*- coding: utf-8 -*-
"""
Part 1: SVM Emotion Model Trainer
- 讀取資料
- 資料前處理 (TF-IDF)
- 模型訓練與評估 (Train/Val/Test Split)
- 輸出所有歌曲的情緒機率矩陣 (song_emotion_probabilities.csv)
"""

import re
import os
import numpy as np
import pandas as pd

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# 設定檔案路徑
BASE_DIR = "文字探勘期末專案"
INPUT_FILENAME = "new_songs_for_human_labeling.xlsx - Sheet1.csv"
OUTPUT_PROB_FILE = "song_emotion_probabilities.csv"

EMOTIONS = ["Q1", "Q2", "Q3", "Q4"]  # Q1開心/興奮, Q2憤怒/緊張, Q3悲傷/痛苦, Q4放鬆/平靜

def get_file_path(filename):
    if os.path.exists(filename):
        return filename
    elif os.path.exists(os.path.join(BASE_DIR, filename)):
        return os.path.join(BASE_DIR, filename)
    else:
        # Fallback Logic
        if filename.endswith(".csv"):
            fallback = filename.replace(".csv", "").strip() # Try removing extension if double
            if not fallback.endswith(".xlsx"):
                fallback = "new_songs_for_human_labeling.xlsx"
            
            if os.path.exists(fallback): return fallback
            if os.path.exists(os.path.join(BASE_DIR, fallback)): return os.path.join(BASE_DIR, fallback)
    return None

def load_songs(file_path: str) -> pd.DataFrame:
    print(f"正在讀取檔案: {file_path}")
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    if "human_label" not in df.columns:
        print("警告：檔案中缺少 'human_label' 欄位，將無法進行訓練。")
        df["human_label"] = np.nan
        
    df["text"] = df["text"].fillna("").astype(str)
    
    def clean_label(val):
        if pd.isna(val): return np.nan
        val = str(val).strip()
        match = re.search(r"(Q[1-4])", val, re.IGNORECASE)
        if match: return match.group(1).upper()
        return np.nan

    df["label_clean"] = df["human_label"].apply(clean_label)
    return df

def get_stop_words() -> set:
    sw = set(stopwords.words("english")).union({
        "oh", "yeah", "hey", "la", "da", "ooh", "ah", "na", "ha",
        "chorus", "verse", "intro", "outro", "bridge", "refrain",
        "im", "youre", "hes", "shes", "theyre", "aint", "gonna",
        "wanna", "gotta", "feat", "ft", "yo", "uh", "mmm"
    })
    return sw

def clean_lyrics(text: str, stop_words: set) -> str:
    text = text.lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)

def build_tfidf_matrix(df: pd.DataFrame, max_features: int = 5000):
    stop_words = get_stop_words()
    print("正在清理歌詞與建立 TF-IDF 矩陣...")
    df = df.copy()
    df["clean_text"] = df["text"].apply(lambda t: clean_lyrics(t, stop_words))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=5,
        max_df=0.8,
        stop_words=list(stop_words)
    )
    X = vectorizer.fit_transform(df["clean_text"])
    return vectorizer, X, df

def train_emotion_model(X, y: pd.Series) -> SVC:
    print(f"正在訓練 SVM 模型 (樣本數: {len(y)})...")
    clf = SVC(kernel='linear', probability=True, random_state=42)
    clf.fit(X, y)
    print("模型訓練完成！")
    return clf

def predict_emotion_scores(clf: SVC, X) -> np.ndarray:
    print("正在預測所有歌曲的情緒機率...")
    proba = clf.predict_proba(X)
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
    scores = np.zeros((proba.shape[0], len(EMOTIONS)), dtype=float)
    for j, emo in enumerate(EMOTIONS):
        if emo in class_to_idx:
            scores[:, j] = proba[:, class_to_idx[emo]]
        else:
            scores[:, j] = 0.0
    return scores

def main():
    file_path = get_file_path(INPUT_FILENAME)
    if not file_path:
        print(f"錯誤：找不到輸入檔案 {INPUT_FILENAME}")
        return

    df = load_songs(file_path)
    vectorizer, X, df_clean = build_tfidf_matrix(df)
    
    labeled_mask = df_clean["label_clean"].notna()
    if not labeled_mask.any():
        print("錯誤：資料中沒有有效的人工標註 (Q1~Q4)，無法訓練模型。")
        return

    X_labeled = X[labeled_mask.to_numpy()]
    y_labeled = df_clean.loc[labeled_mask, "label_clean"]
    print(f"\n有效標註資料筆數: {len(y_labeled)}")
    
    # === 模型評估 (Train/Test Split 策略) ===
    # 策略: Test 30%, Train 70% (其中 Train 裡面的 20% 當 Validation)
    print("\n[模型評估] 資料切割: Test 30%, Train 70% (Validation is 20% of Train)...")
    
    # 1. 切分 Test (30%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.3, random_state=42, stratify=y_labeled
    )
    
    # 2. 從 Train 切分 Validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"Dataset Counts -> Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # 訓練與驗證評估
    clf_eval = train_emotion_model(X_train, y_train)
    val_acc = clf_eval.score(X_val, y_val)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    print("\n--- Test Set Evaluation Result ---")
    y_pred = clf_eval.predict(X_test)
    y_prob = clf_eval.predict_proba(X_test)
    
    print(classification_report(y_test, y_pred, target_names=clf_eval.classes_))
    
    # 繪製圖表
    cm = confusion_matrix(y_test, y_pred, labels=clf_eval.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_eval.classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix (SVM)")
    plt.savefig("svm_confusion_matrix.png")
    plt.close()
    print("已儲存 Confusion Matrix 圖表")
    
    # ROC Curve
    y_test_bin = label_binarize(y_test, classes=clf_eval.classes_)
    n_classes = y_test_bin.shape[1]
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC {clf_eval.classes_[i]} (AUC = {roc_auc:.2f})')
        except Exception: pass
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc="lower right")
    plt.title('ROC Curve (One-vs-Rest)')
    plt.savefig("svm_roc_curve.png")
    plt.close()
    print("已儲存 ROC Curve 圖表")

    # === 最終訓練與輸出 ===
    print("\n[最終模型] 使用所有標註資料重新訓練，以輸出所有歌曲的情緒機率...")
    clf_final = train_emotion_model(X_labeled, y_labeled)
    emotion_scores = predict_emotion_scores(clf_final, X)
    
    # 準備輸出 DataFrame，務必保留 popularity 以供後續推薦使用
    cols_to_keep = ["artist", "song"]
    if "track_id" in df_clean.columns: cols_to_keep.append("track_id")
    if "popularity" in df_clean.columns: cols_to_keep.append("popularity") # 關鍵!
    
    prob_df = df_clean[cols_to_keep].copy()
    for i, emo in enumerate(EMOTIONS):
        prob_df[f"{emo}_prob"] = emotion_scores[:, i]
        
    prob_df.to_csv(OUTPUT_PROB_FILE, index=False, encoding="utf-8-sig")
    print(f"\n[成功] 情緒機率矩陣已儲存至: {OUTPUT_PROB_FILE}")
    print("請接著執行 `python emotion_recommender.py` 進行推薦。")

if __name__ == "__main__":
    main()
