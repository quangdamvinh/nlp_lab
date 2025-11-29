# Báo cáo Lab 6: Introduction to Transformers

File mã nguồn: https://github.com/quangdamvinh/nlp_lab/blob/main/notebooks/lab6_intro_transformers.ipynb

## Bài 1: Khôi phục Masked Token
Kết quả:

Câu gốc: Hanoi is the <mask> of Vietnam.  
Dự đoán: ' capital' với độ tin cậy: 0.9341  
 -> Câu hoàn chỉnh: Hanoi is the capital of Vietnam.  
Dự đoán: ' Republic' với độ tin cậy: 0.0300  
 -> Câu hoàn chỉnh: Hanoi is the Republic of Vietnam.  
Dự đoán: ' Capital' với độ tin cậy: 0.0105  
 -> Câu hoàn chỉnh: Hanoi is the Capital of Vietnam.  
Dự đoán: ' birthplace' với độ tin cậy: 0.0054  
 -> Câu hoàn chỉnh: Hanoi is the birthplace of Vietnam.  
Dự đoán: ' heart' với độ tin cậy: 0.0014  
 -> Câu hoàn chỉnh: Hanoi is the heart of Vietnam.

 1. Mô hình dự đoán đúng từ 'capital'.
 2. Các mô hình Encoder-only như BERT phù hợp cho tác vụ masked token vì cách nó được huấn luyện là dự đoán các từ bị thiếu trong câu và phương thức huấn luyện theo cả hai chiều nên có thể nắm bắt được quan hệ giữa các từ trong câu từ đó đưa ra dự đoán chính xác cho từ bị thiếu.

 ## Bài 2: Dự đoán từ tiếp theo
 Kết quả:

Câu mồi: 'The best thing about learning NLP is'  
Văn bản được sinh ra:  
The best thing about learning NLP is that it's simple, straightforward to understand, and is highly enjoyable. It also teaches you how to get to know and listen to your own music. I recommend learning it as a beginner or intermediate to help you get over the initial learning curve.

So what should I do?

After reading a lot of advice and getting over the initial learning curve, I'm sure you'll be impressed with NLP. It's easy to listen to, and it can be mastered by anyone. Learning my own songs is much easier, and it's much more rewarding. Also, it's a great way to get started with NLP.

I highly recommend this book. It's called NLP, and it's the best book for beginners.

NLP is a very well-structured book, and it's filled with information on all of the different stages of NLP. It's also a great way to get to know your own music.

I'm sure you'll find this book as helpful as I have found it to be, but I'd recommend reading it.

If you liked this post, please share it with your friends. If you liked this post, please read it.

Related Posts:

1. Kết quả sinh ra không được hợp lí cho lắm có thể do dữ liệu huấn luyện chưa đủ nhiều nên mô hình không liên hệ được NLP với Natural Language Processing.
2. Các mô hình Decoder-only như GPT phù hợp cho tác vụ này vì cách chúng được huấn luyện là dự đoán từ tiếp theo từ các từ trước đó.

## Bài 3: Tính toán vector biểu diễn của câu
1. Kích thước của vector biểu diễn là 768 ứng với số chiều của embedding vector.
2. Ta cần sử dụng attention_mask khi thực hiện mean pooling vì để loại bỏ ảnh hưởng của các padding tokens vốn chiếm rất lớn nếu câu ngắn.