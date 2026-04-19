import editdistance

GroundTruth = "VIOS"
H = "VOS"
N = len(GroundTruth)

total_errors = editdistance.distance(GroundTruth, H)

cer_value = total_errors / N

print(f"Tổng số ký tự: {N}")
print(f"Tổng số lỗi: {total_errors}")
print(f"CER: {cer_value:.4f} ({cer_value*100:.2f}%)")

car_value = 1 - cer_value
if car_value < 0:
    car_value = 0.0
print(f"CAR: {car_value:.4f} ({car_value*100:.2f}%)")