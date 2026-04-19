# Hệ thống Nhận diện Mẫu xe bằng OCR (Smart Parking System)

Dự án này ứng dụng công nghệ Nhận dạng Ký tự Quang học (OCR) để tự động phát hiện, trích xuất và nhận diện văn bản từ ảnh chụp phần đuôi xe. Hệ thống được thiết kế để đối chuẩn hiệu năng giữa nhiều mô hình OCR khác nhau và triển khai mô hình tối ưu nhất vào ứng dụng thực tế.

## Cấu trúc dự án

Dự án bao gồm các tệp và thư mục chính sau:

* **`app_main.py`**: Ứng dụng chính thức (Official App). Triển khai 파ipeline end-to-end kết hợp mô hình **CRAFT** (từ thư viện EasyOCR để phát hiện vùng chữ) và **TrOCR** (để nhận dạng chữ). Đây là phiên bản mang lại kết quả tối ưu nhất để tích hợp vào hệ thống bãi đỗ xe.
* **`app_test.py`**: Ứng dụng thử nghiệm và đối chuẩn (Benchmarking App). Được xây dựng để chạy đánh giá độc lập và so sánh độ chính xác của 4 kiến trúc mô hình OCR khác nhau bao gồm: *EasyOCR, PaddleOCR, TesseractOCR* và *TrOCR*.
* **`evaluation.py`**: Script tính toán các chỉ số đánh giá hiệu năng mô hình. Hệ thống tập trung đo lường thông qua:
    * **CER (Character Error Rate):** Tỷ lệ lỗi ký tự.
    * **CAR (Character Accuracy Rate):** Tỷ lệ chính xác ký tự (được tính trực tiếp bằng `1 - CER` để mang lại góc nhìn trực quan và dễ theo dõi hơn về hiệu suất nhận diện).
* **`image_test/`**: Thư mục chứa tập dữ liệu ảnh chụp đuôi xe thực tế, được sử dụng làm các test case đầu vào để đánh giá mô hình và báo cáo số liệu thực nghiệm.
