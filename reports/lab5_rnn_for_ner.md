# Báo cáo Lab 5: RNN for NER

File mã nguồn:  https://github.com/quangdamvinh/nlp_lab/blob/main/notebooks/lab5_rnn_for_ner.ipynb

- Độ chính xác trên tập validation: 0.9483
- Ví dụ dự đoán câu mới:
    - "VNU University is located in Hanoi"
    - Dự đoán: ('VNU', 'B-ORG'), ('University', 'I-ORG'), ('is', 'O'), ('located', 'O'), ('in', 'O'), ('Hanoi', 'B-LOC')