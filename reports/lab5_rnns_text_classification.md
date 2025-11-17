# Report lab 5: RNNs for text classification

## So sánh định lượng:

| Pipeline                       | F1-score (macro)| Validation loss |
|--------------------------------|-----------------|-----------------|
| TF-IDF Logistic Regression     | 0.84            | N/A             |
| Word2Vec (avg) + Dense         | 0.73            | 0.9062          |
| Embedding (Pre-trained) + LSTM | 0.77            | 0.7529          |
| Embedding (Scratch) + LSTM     | 0.81            | 0.8146          |