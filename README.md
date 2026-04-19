app_main.py: app chính thức, sử dụng CRAFT (EasyOCR) + TrOCR
app_test.py: app test, dùng để thử nghiệm độ chính xác của bốn mô hình OCR là EasyOCR, PaddleOCR, TesseractOCR và TrOCR
evaluation.py: dùng để tính nhanh chỉ số đánh giá CER và CAR
image_test: file chứa các ảnh đã được dùng làm test case cho mô hình OCR và được đánh giá trong báo cáo