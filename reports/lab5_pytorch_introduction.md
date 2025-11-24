# Lab 5 Pytorch Introduction review:

File mã nguồn: https://github.com/quangdamvinh/nlp_lab/blob/main/lab5_pytorch_introduction.ipynb

Câu hỏi: Chuyện gì xảy ra nếu bạn gọi z.backward() một lần nữa? Tại sao?

Trả lời: Nếu tiếp tục gọi `z.backward()` một lần nữa (với điều kiện phải khai báo lại `z = y * y + 3`) thì kết quả sẽ đổi từ 18 thành 36 (tức là 18 + 18) vì tensor.grad không tự reset sau mỗi lần gọi backward() nên kết quả sẽ được cộng dồn vào x.grad. Còn nếu chỉ gọi `z.backward()` thì sẽ báo lỗi vì PyTorch mặc định sẽ giải phóng đồ thị tính toán (computational graph) sau khi gọi backward() nên PyTorch không còn đồ thị để tính và sẽ báo lỗi.