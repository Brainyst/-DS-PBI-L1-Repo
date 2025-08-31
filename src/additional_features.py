import pandas as pd
import os

# -----------------------------
# Load datasets
# -----------------------------
channel_master = pd.read_csv(r"C:\Users\Asus\PycharmProjects\DS-YouTube-Project\data\Channel_Master.csv")
video_summary = pd.read_csv(r"C:\Users\Asus\PycharmProjects\DS-YouTube-Project\data\Language_Master.csv")
revenue_master = pd.read_csv(r"C:\Users\Asus\PycharmProjects\DS-YouTube-Project\data\Revenue_Master.csv")

# -----------------------------
# Create tgt folder if not exists
# -----------------------------
tgt_dir = r"C:\Users\Asus\PycharmProjects\DS-YouTube-Project\tgt"
os.makedirs(tgt_dir, exist_ok=True)

# -----------------------------
# Feature 1: Average views per channel (language_master)
# -----------------------------
video_summary['avg_views_per_channel'] = video_summary['TotalViews'] / video_summary['ChannelCount']

# -----------------------------
# Feature 2: Average speakers per channel (language_master)
# -----------------------------
video_summary['avg_speakers_per_channel'] = (
    video_summary['speakers_count_ChatGpt'] +
    video_summary['speakers_count_Gemini'] +
    video_summary['Speakers_Count_Google']
) / video_summary['ChannelCount']

# -----------------------------
# Feature 3: Subscriber-to-views ratio (language_master)
# -----------------------------
video_summary['subs_to_views_ratio'] = video_summary['SubscriberCount'] / video_summary['TotalViews']

# -----------------------------
# Merge with channel_master and revenue_master
# -----------------------------
merged_df = channel_master.merge(
    revenue_master,
    how='left',
    left_on='Channel_Id',
    right_on='Channelid'
)

merged_df = merged_df.merge(
    video_summary[['Language', 'avg_views_per_channel', 'avg_speakers_per_channel', 'subs_to_views_ratio']],
    how='left',
    left_on='Language',
    right_on='Language'
)

# -----------------------------
# Save the output to tgt folder
# -----------------------------
merged_df.to_csv(os.path.join(tgt_dir, "merged_with_additional_features.csv"), index=False)
print(f"✅ Additional features added and saved in {tgt_dir}")
