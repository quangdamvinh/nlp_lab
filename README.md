# Báo cáo Lab 1 và Lab 2:

## Mô tả công việc
- Lab 1:
    - Viết file interfecrs.py với lớp trừu tượng Tokenizer.
    - Viết file simple_tokenizer.py chứa lớp SimpleTokenizer làm nhiệm vụ tách văn bản dựa trên khoảng trắng và dấu câu.
    - Viết file regex_tokenizer.py chứa lớp RegexTokenizer giúp tách câu thông qua biểu thức chính quy `\w+|[^\w\s]`.
    - Viết file main.py đánh giá hai phương pháp trên một vài câu thử và trên bộ dữ liệu UD_English-EWT.
- Lab 2:
    - Triển khai lớp trừu tượng Vectorizer trong interfaces.py.
    - Triển khai lớp CountVectorizer trong count_vectorizer.py có nhiệm vụ biến đổi văn bản thành vector thưa.
    - Đánh giá trên file lab2_test.py.

## Kết quả chạy code:
- Lab 1:
Text 1: Hello, world! This is a test.  
Simple tokens: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']  
Regex tokens: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']  

Text 2: NLP is fascinating... isn't it?  
Simple tokens: ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', 't', 'it', '?']  
Regex tokens: ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']  

Text 3: Let's see how it handles 123 numbers and punctuation!  
Simple tokens: ['let', 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']  
Regex tokens: ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']  


Task 3:
--- Tokenizing Sample Text from UD_English-EWT ---  
Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of ...  

SimpleTokenizer Output (first 20 tokens): ['al', 'zaman', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim']  

RegexTokenizer Output (first 20 tokens): ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']  

- Lab 2:
Learned vocabulary: {'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}  
Document-term matrix:  
[[1, 0, 0, 1, 0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 1, 0, 0, 1, 0], [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]]

## Giải thích kết quả:
- So sánh giữa SimpleTokenizer và RegexTokenizer: SimpleTokenizer chỉ tách câu dựa trên khoảng trắng và tách riêng các dấu câu còn RegexTokenizer tách câu dựa trên nhiều điều kiện hơn ví dụ như chữ số hay các kí tự đặc biệt.
- CountVectorizer: các câu được chuyển thành các vector có độ dài bằng với độ dài từ điển và đa số các vị trí là 0.