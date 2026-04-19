import os
import cv2
import numpy as np
import threading
from difflib import get_close_matches
from tkinter import Tk, filedialog, Button, Label, Frame, Canvas, Scrollbar, messagebox
from PIL import Image, ImageTk

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import easyocr


class OCR:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Model OCR (EasyOCR + TrOCR)")
        self.root.geometry("1200x800")

        self.ocr_reader = None
        self.trocr_processor = None
        self.trocr_model = None

        self.image_path = None
        self.original_cv_image = None
        self.crops = []

        self.model_dict = [m.lower() for m in [
            "320d", "320i", "330i", "335d", "335i", "530e", "530i", "540i",
            "M2", "M3", "M4", "M5", "M8", "X3", "X4", "X5", "Z3", "Z4",
            "Figo", "Focus", "Fusion", "Mondeo", "Ranger",
            "Civic", "Accord", "City", "CR-V", "HR-V",
            "Accent", "Elantra", "Tucson", "SantaFe", "Sonata",
            "K3", "K5", "Morning", "Seltos", "Sorento",
            "C200", "C300", "E200", "E300", "GLC200", "GLC300",
            "GLE53", "GLE450", "S450", "S500",
            "Attrage", "Mirage", "PAJERO", "OUTLANDER", "MIRAGE", "PAJERO SPORT",
            "Camry", "Corolla", "Highlander", "Prius",
            "VF3", "VF5", "VF6", "VF8", "VF9"
        ]]

        self._setup_ui()

    def _setup_ui(self):
        control_frame = Frame(self.root, pady=10)
        control_frame.pack(side="top", fill="x")

        self.btn_select = Button(
            control_frame, text="1. Chọn ảnh",
            command=self.select_image, bg="#4CAF50", fg="white",
            width=20, height=2, font=("Arial", 11, "bold")
        )
        self.btn_select.pack(side="left", padx=10)

        self.btn_detect = Button(
            control_frame, text="2. Phát hiện vùng chữ",
            command=self._thread_detect_text, bg="#2196F3", fg="white",
            width=30, height=2, font=("Arial", 11, "bold"), state="disabled"
        )
        self.btn_detect.pack(side="left", padx=10)

        self.btn_recognize = Button(
            control_frame, text="3. Nhận diện chữ",
            command=self._thread_recognize_text, bg="#f44336", fg="white",
            width=30, height=2, font=("Arial", 11, "bold"), state="disabled"
        )
        self.btn_recognize.pack(side="left", padx=10)

        self.status_label = Label(self.root, text="Sẵn sàng.", bd=1, relief="sunken", anchor="w")
        self.status_label.pack(side="bottom", fill="x")

        canvas_frame = Frame(self.root)
        canvas_frame.pack(side="left", fill="both", expand=True)

        self.canvas = Canvas(canvas_frame)
        self.scrollbar = Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.display_frame = Frame(self.canvas)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.display_frame, anchor="nw")
        self.display_frame.bind("<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def _update_status(self, text):
        self.status_label.config(text=text)
        self.root.update_idletasks()

    def _set_button_state(self, detect=False, recognize=False):
        self.btn_detect.config(state="normal" if detect else "disabled")
        self.btn_recognize.config(state="normal" if recognize else "disabled")

    def _get_easyocr_reader(self):
        if self.ocr_reader is None:
            self._update_status("Đang khởi tạo EasyOCR reader...")
            self.ocr_reader = easyocr.Reader(['en'])
            self._update_status("EasyOCR đã sẵn sàng.")
        return self.ocr_reader

    def _get_trocr_model(self):
        if self.trocr_model is None:
            self._update_status("Đang khởi tạo TrOCR...")
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
            self._update_status("TrOCR đã sẵn sàng.")
        return self.trocr_processor, self.trocr_model

    def select_image(self):
        file = filedialog.askopenfilename(
            title="Chọn ảnh", filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if not file:
            return

        self.image_path = file
        self.original_cv_image = cv2.imread(file)
        self.crops.clear()

        for widget in self.display_frame.winfo_children():
            widget.destroy()

        Label(
            self.display_frame,
            text=f"Ảnh được chọn: {os.path.basename(file)}",
            font=("Arial", 12, "bold")
        ).pack(pady=5)

        self._display_image(self.original_cv_image)
        self._update_status(f"Đã tải ảnh: {os.path.basename(file)}")
        self._set_button_state(detect=True)

    def _display_image(self, cv_image, parent=None, max_size=(900, 600)):
        parent = parent or self.display_frame
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail(max_size)
        img_tk = ImageTk.PhotoImage(img_pil)
        lbl = Label(parent, image=img_tk)
        lbl.image = img_tk
        lbl.pack(pady=10)

    def _thread_detect_text(self):
        threading.Thread(target=self._detect_text_task, daemon=True).start()

    def _detect_text_task(self, padding=5, crop_zoom=2.0):
        if not self.image_path:
            return

        self._set_button_state(False)
        self._update_status("Đang phát hiện vùng chữ...")

        for widget in self.display_frame.winfo_children():
            widget.destroy()

        reader = self._get_easyocr_reader()
        results = reader.readtext(self.image_path, detail=1)

        img_with_boxes = self.original_cv_image.copy()
        h, w = img_with_boxes.shape[:2]
        self.crops.clear()
        valid_boxes_count = 0

        for bbox, text, conf in results:
            box_np = np.array(bbox, dtype=np.int32)
            x_min, y_min = np.min(box_np, axis=0)
            x_max, y_max = np.max(box_np, axis=0)

            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            if x_min >= x_max or y_min >= y_max:
                continue

            crop = self.original_cv_image[y_min:y_max, x_min:x_max]
            if crop_zoom != 1.0 and crop.size > 0:
                crop = cv2.resize(
                    crop,
                    (int((x_max - x_min) * crop_zoom),
                     int((y_max - y_min) * crop_zoom)),
                    interpolation=cv2.INTER_CUBIC
                )

            self.crops.append(crop)
            cv2.polylines(img_with_boxes, [box_np], True, (0, 255, 0), 2)
            valid_boxes_count += 1

        Label(
            self.display_frame,
            text=f"Phát hiện {valid_boxes_count} vùng chữ hợp lệ",
            font=("Arial", 12, "bold")
        ).pack()

        self._display_image(img_with_boxes)
        self._update_status("Hoàn tất phát hiện vùng chữ.")

        if valid_boxes_count > 0:
            self._set_button_state(detect=True, recognize=True)
        else:
            self._set_button_state(detect=True, recognize=False)
            self._update_status("Không phát hiện vùng chữ hợp lệ.")

    def _thread_recognize_text(self):
        threading.Thread(target=self._recognize_text_task, daemon=True).start()

    def _recognize_text_task(self):
        if not self.crops:
            return

        self._set_button_state(False)
        self._update_status("Đang nhận diện...")

        processor, model = self._get_trocr_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        for widget in self.display_frame.winfo_children():
            widget.destroy()

        Label(
            self.display_frame,
            text="Kết quả nhận diện",
            font=("Arial", 14, "bold")
        ).pack(pady=10)

        for idx, crop in enumerate(self.crops):
            crop_frame = Frame(self.display_frame, relief="groove", borderwidth=2)
            crop_frame.pack(fill="x", padx=20, pady=10)

            Label(crop_frame, text=f"Vùng chữ #{idx + 1}", font=("Arial", 12, "bold")).pack()
            self._display_image(crop, crop_frame, max_size=(400, 150))

            try:
                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)

                with torch.no_grad():
                    generated_ids = model.generate(pixel_values)

                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                clean_text = ''.join(filter(str.isalnum, text)).lower()
                match = get_close_matches(clean_text, self.model_dict, n=1, cutoff=0.7)
                match_str = f"→ Khớp: {match[0]}" if match else "→ Không khớp"

                Label(
                    crop_frame, text=f"Kết quả: {text} | {match_str}",
                    font=("Consolas", 11), fg="#333"
                ).pack(anchor="w", padx=10, pady=5)

            except Exception as e:
                Label(
                    crop_frame, text=f"Lỗi: {e}",
                    font=("Consolas", 10), fg="red"
                ).pack(anchor="w", padx=10, pady=5)

        self._update_status("Hoàn tất nhận diện.")
        self._set_button_state(detect=True, recognize=True)


if __name__ == "__main__":
    root = Tk()
    app = OCR(root)
    root.mainloop()
