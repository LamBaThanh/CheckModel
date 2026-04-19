import os
import cv2
import numpy as np
import threading
from difflib import get_close_matches
from tkinter import Tk, filedialog, Button, Label, Frame, Canvas, Scrollbar, messagebox
from PIL import Image, ImageTk
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

paddle_ocr = None
easy_ocr = None
processor = None
trocr_model = None

MODEL_DICT = [
    "320d","320i","330i","335d","335i","530e","530i","540i",
    "M2","M3","M4","M5","M8","X3","X4","X5","Z3","Z4",
    "Figo","Focus","Fusion","Mondeo","Ranger",
    "Civic","Accord","City","CR-V","HR-V",
    "Accent","Elantra","Tucson","SantaFe","Sonata",
    "K3","K5","Morning","Seltos","Sorento",
    "C200","C300","E200","E300","GLC200","GLC300",
    "GLE53","GLE450","S450","S500",
    "Attrage","Mirage","PAJERO","OUTLANDER","MIRAGE","PAJERO SPORT",
    "Camry","Corolla","Highlander","Prius",
    "VF3","VF5","VF6","VF8","VF9"
]
MODEL_DICT_LOWER = [m.lower() for m in MODEL_DICT]

def init_models():
    global paddle_ocr, easy_ocr, processor, trocr_model
    
    try:
        import easyocr
        easy_ocr = easyocr.Reader(['en'])
        print("EasyOCR loaded")
    except Exception as e:
        print(f"Lỗi EasyOCR: {e}")

    try:
        import paddle
        from paddleocr import PaddleOCR
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        try:
            paddle.device.set_device('cpu')
        except:
            pass
        paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        print("PaddleOCR loaded")
    except Exception as e:
        print(f"Lỗi PaddleOCR: {e}")

    try:
        import pytesseract
        tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # Tải tesseract
        if os.path.exists(tess_path):
            pytesseract.pytesseract.tesseract_cmd = tess_path
            print("Tesseract loaded")
        else:
            print(f"Check lại đường dẫn: {tess_path}")
    except Exception as e:
        print(f"Lỗi Tesseract: {e}")

    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
        trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1") 
        # Xài trocr-large-printed
        print("TrOCR loaded")
    except Exception as e:
        print(f"Lỗi TrOCR: {e}")

init_models()

def trocr_predict(crop_img):
    if processor is None or trocr_model is None: return "Not Loaded"
    img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

root = Tk()
root.title("Car Model OCR Tool (Final Fixed Version)")
root.geometry("1200x800")

image_path = None
crops = []

canvas = Canvas(root)
canvas.pack(side="left", fill="both", expand=True)
scrollbar = Scrollbar(root, command=canvas.yview)
scrollbar.pack(side="right", fill="y")
canvas.configure(yscrollcommand=scrollbar.set)
frame = Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

status_label = Label(root, text="Sẵn sàng.", anchor="w", bd=1, relief="sunken")
status_label.pack(fill="x", side="bottom")

def update_status(text):
    status_label.config(text=text)
    root.update_idletasks()

def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))
frame.bind("<Configure>", on_configure)

def select_image():
    global image_path, crops
    file = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image", "*.png *.jpg *.jpeg")])
    if not file: return
    image_path = file
    crops = []
    for w in frame.winfo_children(): w.destroy()
    Label(frame, text=f"Ảnh: {os.path.basename(image_path)}", font=("Arial", 11, "bold")).pack(pady=5)
    img = Image.open(image_path)
    img.thumbnail((800, 600))
    photo = ImageTk.PhotoImage(img)
    lbl = Label(frame, image=photo)
    lbl.image = photo
    lbl.pack()
    update_status("Đã tải ảnh.")

def detect_text(padding=5, crop_zoom=2.0):
    global crops
    if not image_path: return
    if easy_ocr is None:
        messagebox.showerror("Lỗi", "EasyOCR chưa được tải.")
        return

    for w in frame.winfo_children(): w.destroy()
    Label(frame, text="Đang Detect Text...", font=("Arial", 11, "bold")).pack()
    root.update()

    img_cv = cv2.imread(image_path)
    h_img, w_img = img_cv.shape[:2]
    crops = []
    
    results = easy_ocr.readtext(image_path, detail=1, paragraph=False)
    img_display = img_cv.copy()
    count = 0
    
    for (bbox, text, prob) in results:
        poly = np.array(bbox).astype(int)
        x_min = max(0, np.min(poly[:,0]) - padding)
        y_min = max(0, np.min(poly[:,1]) - padding)
        x_max = min(w_img, np.max(poly[:,0]) + padding)
        y_max = min(h_img, np.max(poly[:,1]) + padding)

        if x_max <= x_min or y_max <= y_min: continue

        cv2.rectangle(img_display, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
        crop = img_cv[y_min:y_max, x_min:x_max].copy()
        
        if crop_zoom > 1.0 and crop.size > 0:
            crop = cv2.resize(crop, None, fx=crop_zoom, fy=crop_zoom, interpolation=cv2.INTER_CUBIC)
            
        if crop.size > 0:
            crops.append(crop)
            count += 1

    img_pil = Image.fromarray(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    img_pil.thumbnail((800, 500))
    photo = ImageTk.PhotoImage(img_pil)
    Label(frame, image=photo).pack(pady=5)
    Label(frame, image=photo).image = photo
    Label(frame, text=f"Tìm thấy {count} vùng chữ.", font=("Arial", 10)).pack()
    update_status(f"Detect xong: {count} vùng.")

def recognize_text():
    if not crops: return
    for w in frame.winfo_children(): w.destroy()
    Label(frame, text="Kết quả Nhận diện", font=("Arial", 14, "bold"), fg="#333").pack(pady=10)
    root.update()
    
    import pytesseract

    for idx, crop_img in enumerate(crops):
        row = Frame(frame, relief="groove", bd=2)
        row.pack(fill="x", padx=10, pady=5)
        
        c_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        c_pil.thumbnail((250, 80))
        c_tk = ImageTk.PhotoImage(c_pil)
        l_img = Label(row, image=c_tk)
        l_img.image = c_tk
        l_img.pack(side="left", padx=10)
        
        txt_frame = Frame(row)
        txt_frame.pack(side="left", fill="both", expand=True)
        
        res_dict = {}

        if paddle_ocr:
            try:
                img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                p_res = paddle_ocr.predict(img_rgb)
                text = ""
                if p_res:
                    if isinstance(p_res[0], dict) and 'rec_texts' in p_res[0]:
                        rec_list = p_res[0]['rec_texts']
                        if rec_list and len(rec_list) > 0:
                            text = rec_list[0]
                    
                    elif isinstance(p_res[0], list) and len(p_res[0]) > 0:
                        item = p_res[0]
                        if isinstance(item, list): text = item[0][0]
                        elif isinstance(item, tuple): text = item[0]
                        
                res_dict["Paddle"] = str(text)
            except Exception as e:
                res_dict["Paddle"] = "Err"
                print(f"Paddle Error: {e}")
        else:
            res_dict["Paddle"] = "Not Installed"

        if easy_ocr:
            try:
                e_res = easy_ocr.readtext(crop_img, detail=0)
                res_dict["EasyOCR"] = " ".join(e_res)
            except: res_dict["EasyOCR"] = "Err"

        try:
            tess_text = pytesseract.image_to_string(crop_img, config='--psm 7')
            res_dict["Tesseract"] = tess_text.strip()
        except: res_dict["Tesseract"] = "Err (Check Path)"

        try:
            res_dict["TrOCR"] = trocr_predict(crop_img)
        except: res_dict["TrOCR"] = "Err"

        for model, content in res_dict.items():
            Label(txt_frame, text=f"{model}: {content}", font=("Consolas", 10)).pack(anchor="w")

        candidates = []
        for v in res_dict.values():
            if not v or "Err" in v or "Not" in v: continue
            clean = "".join(x for x in v if x.isalnum()).lower()
            candidates.append(clean)
        
        found_model = None
        for cand in candidates:
            matches = get_close_matches(cand, MODEL_DICT_LOWER, n=1, cutoff=0.6)
            if matches:
                idx_match = MODEL_DICT_LOWER.index(matches[0])
                found_model = MODEL_DICT[idx_match]
                break

        if found_model:
            Label(txt_frame, text=f"==> Kết quả: {found_model}", font=("Arial", 11, "bold"), fg="red").pack(anchor="w", pady=2)
        else:
            Label(txt_frame, text="==> Không khớp model nào", font=("Arial", 10, "italic"), fg="gray").pack(anchor="w")

    update_status("Hoàn tất.")

btn_bar = Frame(root, pady=10)
btn_bar.pack(side="top", fill="x")
Button(btn_bar, text="Chọn Ảnh", command=select_image, bg="#4CAF50", fg="white", font=("Arial",10,"bold")).pack(side="left", padx=10)
Button(btn_bar, text="Detect", command=detect_text, bg="#2196F3", fg="white", font=("Arial",10,"bold")).pack(side="left", padx=10)
Button(btn_bar, text="Recognize", command=recognize_text, bg="#FF5722", fg="white", font=("Arial",10,"bold")).pack(side="left", padx=10)

root.mainloop()