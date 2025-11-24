# Báo cáo lab 4:

## Phần 1: Giảm chiều và trực quan hóa vector
**Các bước thực hiện cũng như đánh giá kết quả của phần 1 đã có trong file dimension_reduced_n_visualized_vector.ipynb (hoặc file PDF nộp trên classroom).**

## Phần 2: Word Embedding với Word2Vec

### Các bước thực hiện:
- Task 1: Chuẩn bị thw viện (gensm).
- Task 2: Viết file word_embedder.py triển khai lớp WordEmbedder thực hiện nạp mô hình 'glove-wiki-gigaword-50' và triển khai các phương thứ lấy vector và tìm độ tương đồng.
- Task 3: Triển khai thêm phương thức embed_documet() để biến các câu thành các vector.
- Đánh giá kết quả trong lab4_test.py.

### Kết quả chạy code:
Lab 4 test:
Vector for the word "king": [ 0.50451   0.68607  -0.59517  -0.022801  0.60046  -0.13498  -0.08813
  0.47377  -0.61798  -0.31012  -0.076666  1.493    -0.034189 -0.98173
  0.68229   0.81722  -0.51874  -0.31503  -0.55809   0.66421   0.1961
 -0.13495  -0.11476  -0.30344   0.41177  -2.223    -1.0756   -1.0783
 -0.34354   0.33505   1.9927   -0.04234  -0.64319   0.71125   0.49159
  0.16754   0.34344  -0.25663  -0.8523    0.1661    0.40102   1.1685
 -1.0137   -0.21585  -0.15155   0.78321  -0.91241  -1.6106   -0.64426
 -0.51042 ]
Similarity between "king" and "queen": 0.78390425
Similarity between "king" and "man": 0.53093773
10 most similar words to "computer": [('computers', 0.9165045022964478), ('software', 0.8814992904663086), ('technology', 0.852556049823761), ('electronic', 0.812586784362793), ('internet', 0.8060455322265625), ('computing', 0.802603542804718), ('devices', 0.8016185760498047), ('digital', 0.7991793751716614), ('applications', 0.7912740707397461), ('pc', 0.7883159518241882)]
Document: The queen rules the country.
Vector: [ 0.04564168  0.36531    -0.55974333  0.04014383  0.0965555   0.15623933
 -0.33622833 -0.12495166 -0.01031508 -0.50067167  0.18690467  0.17482167
 -0.268985   -0.03096624  0.36686516  0.29983267  0.01397333 -0.06872117
 -0.32606832 -0.210115    0.168354   -0.03151733 -0.06204717  0.04301083
 -0.06958767 -1.77921669 -0.54365399 -0.06104483 -0.17617999  0.009181
  3.39163339  0.08742473 -0.46754166 -0.213435    0.02391886 -0.04470453
  0.20636833 -0.12902867 -0.28527133 -0.24318051 -0.31144227 -0.03833717
  0.11977984 -0.01418401 -0.37086334  0.22069355 -0.28848937 -0.361888
 -0.00549529 -0.4699725 ]

 ### Giải thích kết quả:
- Vector nhận được là các vector dày đặc, mỗi từ trong từ điển tương ứng với một vector.
- Độ tương đồng giữa các từ được tinh bằng cosine giữa các vector tương ứng.
- Vector biểu diễn một câu được tính bằng trung bình của các vector tương ứng với các từ có trong câu đó.