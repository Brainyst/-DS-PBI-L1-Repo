import os

import pandas as pd
import numpy as np
from datetime import datetime, timezone

import tgt
# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

channel_master = pd.read_csv(r"C:\Users\Asus\PycharmProjects\DS-YouTube-Project\data\channel_master.csv")
video_summary = pd.read_csv(r"C:\Users\Asus\PycharmProjects\DS-YouTube-Project\data\video_summary.csv")

merged_df = channel_master.merge(
    video_summary,
    how="left",
    left_on="Channel_Id",   # matches channel_master
    right_on="ChannelId"    # matches video_summary
)

selected_cols = [
    "Video_Id", "Title_x", "Uploader_x", "Channel_Id",
    "Duration_Sec", "View_Count", "Language",
    "SubscriberCount", "ShortDescription", "TotalViews"
]

clean_df = merged_df[selected_cols].rename(columns={
    "Title_x": "video_title",
    "Uploader_x": "video_uploader",
    "Duration_Sec": "duration_sec",
    "View_Count": "view_count",
    "SubscriberCount": "subscribercount",
    "ShortDescription": "shortdescription",
    "TotalViews": "totalviews",
    "Language": "language"
})

clean_df.head(2)

for col in ["duration_sec","view_count","subscribercount","totalviews"]:
    clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").fillna(0)

clean_df["video_title"] = clean_df["video_title"].astype(str)
clean_df["shortdescription"] = clean_df["shortdescription"].astype(str)
clean_df["language"] = clean_df["language"].astype(str)

def combine_text(df):
    return (df["video_title"].fillna("") + " " + df["shortdescription"].fillna("")).astype(str)

clean_df["text_all"] = combine_text(clean_df)

clean_df.head(3)

kids_kw = [
    "nursery","rhyme","cocomelon","peppa","kid","kids","toddler","baby","rhymes",
    "cartoon","animation","story time","learn colors","abc","phonics","lullaby",
    "play doh","toy","toys","kindergarten","little angel","diana","nika",
    "shorts #kids","#kid","#kids","family","clean","disney","pixar","song",
    "music video","vlog","comedy","funny","minecraft","roblox","cartoons",
    "drawing","craft","origami"
]

adult_kw = [
    "18+","nsfw","xxx","sex","sexy","nude","erotic","violence","gore","blood",
    "horror","killing","gun","weapon","crime","murder","drugs","alcohol",
    "gambling","politics","war","terror","explicit","uncensored","hot",
    "bikini","lingerie"
]

def label_audience_binary(row):
    text = (row["text_all"] or "").lower()
    if any(k in text for k in adult_kw):
        return "18+"
    return "kids"

clean_df["audience_label_binary"] = clean_df.apply(label_audience_binary, axis=1)

clean_df.head(3)

category_keywords = {
    "Education": ["tutorial","how to","lesson","learn","course","math","science","experiment","study","exam","class","education","tips"],
    "Tech": ["technology","tech","unboxing","review","iphone","android","software","coding","programming","python","ai","gadget","laptop"],
    "Entertainment": ["entertainment","vlog","prank","funny","comedy","challenge","dance","movie","film","web series","serial","drama"],
    "Music": ["music","song","lyrics","official video","cover","remix","album","concert","guitar","piano","singer"],
    "Gaming": ["gaming","gameplay","pubg","free fire","minecraft","roblox","valorant","bgmi","fortnite","live stream"],
    "News": ["news","update","breaking","headlines","report","debate","interview"],
    "Kids": ["kids","kid","nursery","rhyme","cartoon","animation","toy","toys","phonics","abc","learn colors","lullaby"],
    "Sports": ["cricket","football","soccer","ipl","highlights","match","t20","world cup","kabaddi","badminton"],
    "Devotional": ["bhajan","kirtan","aarti","devotional","mandir","temple","allah","qawwali","namaz","prayer","gospel"]
}

def label_category_silver(text):
    t = (text or "").lower()
    for cat, kws in category_keywords.items():
        if any(k in t for k in kws):
            return cat
    return "Entertainment"

clean_df["category_label_silver"] = clean_df["text_all"].apply(label_category_silver)

numeric_features = ["duration_sec","view_count","subscribercount","totalviews"]
text_feature = "text_all"

X_text = clean_df[[text_feature] + numeric_features]
y_aud = clean_df["audience_label_binary"]
y_cat = clean_df["category_label_silver"]

shared_vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,2))

def select_numeric_features(X):
    return X[numeric_features]

preprocess_shared = ColumnTransformer(
    transformers=[
        ("text", shared_vectorizer, text_feature),
        ("num", Pipeline([
    ("select", FunctionTransformer(select_numeric_features, validate=False)),
    ("scaler", StandardScaler(with_mean=False))
]), numeric_features)
    ],
    remainder="drop"
)

def train_eval(X, y, models):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    reports, trained_models = {}, {}
    for name, model in models.items():
        pipe = Pipeline([
            ("prep", clone(preprocess_shared)),
            ("clf", model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        reports[name] = rep
        trained_models[name] = pipe
    return reports, trained_models

aud_models = {
    "LogReg": LogisticRegression(max_iter=200, multi_class="auto"),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
}
aud_reports, aud_trained = train_eval(X_text, y_aud, aud_models)

cat_models = {
    "LogReg": LogisticRegression(max_iter=300, multi_class="auto"),
    "MultinomialNB": MultinomialNB()
}
cat_reports, cat_trained = train_eval(X_text, y_cat, cat_models)

def get_weighted_f1(report):
    return report["weighted avg"]["f1-score"]

best_aud_name = max(aud_reports, key=lambda k: get_weighted_f1(aud_reports[k]))
best_cat_name = max(cat_reports, key=lambda k: get_weighted_f1(cat_reports[k]))

best_aud_model = aud_trained[best_aud_name]
best_cat_model = cat_trained[best_cat_name]

aud_preds = best_aud_model.predict(X_text)
cat_preds = best_cat_model.predict(X_text)

# -------------------------
# Folders
# -------------------------
project_dir = r"C:\Users\Asus\PycharmProjects\DS-YouTube-Project"
tgt_dir = os.path.join(project_dir, "tgt")
model_dir = os.path.join(project_dir, "model")

# Ensure the folders exist
os.makedirs(tgt_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# -------------------------
# Add predictions to the full dataset
# -------------------------
output_df = clean_df.copy()
output_df["audience_safety_pred"] = aud_preds
output_df["content_category_pred"] = cat_preds
output_df["aud_model_used"] = best_aud_name
output_df["cat_model_used"] = best_cat_name
output_df["prediction_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# Save full predictions
output_df.to_csv(os.path.join(tgt_dir, "video_predictions.csv"), index=False)
print("✅ Full predictions saved to tgt/video_predictions.csv")

# -------------------------
# Predictions-only dataset
# -------------------------
output_preds = clean_df[["Video_Id", "Channel_Id"]].copy()
output_preds["audience_safety_pred"] = aud_preds
output_preds["content_category_pred"] = cat_preds
output_preds["aud_model_used"] = best_aud_name
output_preds["cat_model_used"] = best_cat_name
output_preds["prediction_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# Save predictions-only CSV
output_preds.to_csv(os.path.join(tgt_dir, "video_predictions_only.csv"), index=False)
print("✅ Predictions-only saved to tgt/video_predictions_only.csv")

# -------------------------
# Save models
# -------------------------
import joblib

joblib.dump(best_aud_model, os.path.join(model_dir, "best_audience_model.pkl"))
joblib.dump(best_cat_model, os.path.join(model_dir, "best_category_model.pkl"))
print("✅ Models saved to model folder")

# -------------------------
# Print best models and F1-scores
# -------------------------
print("Best Audience Model:", best_aud_name, "F1-score:", get_weighted_f1(aud_reports[best_aud_name]))
print("Best Category Model:", best_cat_name, "F1-score:", get_weighted_f1(cat_reports[best_cat_name]))
