# Báo cáo Lab 5: Text Classification

## Các bước triển khai:
- Pipeline: raw text -> Tokenization (sử dụng RegexTokenizer có sẵn) -> Vectorization (sử dụng TfidfVectorizer có sẵn) -> ML Model (Logistic Regression trong thư viện scikit-learn) -> Prediction (Accuracy, Precision, Recall, F1-Score).
- Triển khai lớp TextClassifier trong file text_classifier.py với các phương thức fit() để gọi mô hình Logistic Regression và thực hiện huấn luyện, predict() thực hiện dự đoán trên tập test và evaluate() đưa ra các chỉ số đánh giá cho tập test.
- Viết file lab5_test.py thực hiện và đánh giá trên dữ liệu tạo sẵn.

## Kết quả chạy code:
- Dữ liệu giả lập:
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
    ]

labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative
- Kết quả:
Evaluation metrics:  
accuracy: 0.5000  
precision: 0.5000  
recall: 1.0000  
f1-score: 0.6667  

## Cải tiến mô hình:
...