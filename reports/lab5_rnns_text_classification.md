# Báo cáo lab 5: RNNs for text classification

File mã nguồn: https://github.com/quangdamvinh/nlp_lab/blob/main/notebooks/lab5_rnns_for_text_classification.ipynb

## So sánh định lượng

| Pipeline                       | F1-score (macro)| Validation loss |
|--------------------------------|-----------------|-----------------|
| TF-IDF + Logistic Regression   | 0.84            | N/A             |
| Word2Vec (avg) + Dense         | 0.78            | 0.8148          |
| Embedding (Pre-trained) + LSTM | 0.81            | 0.6849          |
| Embedding (Scratch) + LSTM     | 0.78            | 0.8923          |

## Phân tích định tính

| Test samples | True labels | TF-IDF + LogReg | W2V (avg) + Dense | W2V + LSTM | Embedding + LSTM |
|----------|----------|----------|----------|----------|----------|
| "raise volume to level seven on music player" | "audio_volume_up" | 'audio_volume_up' | 'audio_volume_down' | 'audio_volume_up' | 'audio_volume_up' |
| "which time zone are we in please change to current" | "datetime_convert" | 'datetime_convert' | 'datetime_query' | 'datetime_convert' | 'datetime_query' |
| "show number and contact email of rehan" | "email_querycontact" | 'email_querycontact' | 'email_querycontact' | 'email_querycontact' | 'email_querycontact' |
| "will you check and confirm the instruction please." | "general_confirm" | 'general_confirm' | 'general_confirm' | 'general_confirm' | 'general_confirm' |
| "is mile marker sixty five where the hanging tree is located" | "general_quirky" | 'qa_factoid' | 'qa_factoid' | 'qa_factoid' | 'recommendation_locations' |
| "brighten living room lights" | "iot_hue_lightup" | 'iot_hue_lightup' | 'iot_hue_lightup' | 'iot_hue_lightdim' | 'iot_hue_lightup' |
| "can you locate some gospel music for me" | "music_query" | 'play_music' | 'play_music' | 'play_music' | 'music_likeness' |
| "how much is ten dollars in euros" | "qa_currency" | 'qa_currency' | 'qa_currency' | 'qa_currency' | 'qa_currency' |
| "what is the capital of new hampshire" | "qa_factoid" | 'qa_factoid' | 'qa_factoid' | 'music_query' | 'qa_factoid' |
| "find updates from vicki's facebook from mardi gras day" | "social_query" | 'social_query' | 'social_query' | 'social_query' | 'social_query' |

