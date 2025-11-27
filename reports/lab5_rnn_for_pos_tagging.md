# Báo cáo lab 5: RNN for POS Tagging

File mã nguồn: https://github.com/quangdamvinh/nlp_lab/blob/main/notebooks/lab5_rnn_for_pos_tagging.ipynb

- Độ chính xác trên tập dev: 0.8669
- Ví dụ dự đoán câu mới:
    - Câu: "This is a test sentence."
    - Dự đoán: ('This', 'DET'), ('is', 'AUX'), ('a', 'DET'), ('test', 'NOUN'), ('sentence', 'NOUN'), ('.', 'PUNCT')
    - Câu: "I love NLP"
    - Dự đoán: ('I', 'PRON'), ('love', 'VERB'), ('NLP', 'VERB')
    - Câu: "I love Natural Language Processing."
    - Dự đoán: ('I', 'PRON'), ('love', 'VERB'), ('Natural', 'ADJ'), ('Language', 'NOUN'), ('Processing', 'NOUN'), ('.', 'PUNCT')
    - Câu: "I record the blue record."
    - Dự đoán: ('I', 'PRON'), ('record', 'VERB'), ('the', 'DET'), ('blue', 'ADJ'), ('record', 'NOUN'), ('.', 'PUNCT')