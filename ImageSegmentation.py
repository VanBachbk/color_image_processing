import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# =============================
#  GUI APP XỬ LÝ ẢNH
# =============================

class EdgeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Edge Detection GUI - Sobel, Laplacian, Canny, Adaptive Threshold")
        self.root.geometry("1100x600")

        self.image = None          # ảnh gốc OpenCV (BGR)
        self.processed = None      # ảnh xử lý (BGR)
        self.current_method = None # phương pháp hiện tại
        self.display_image = None  # ảnh hiển thị Tkinter

        # Biến cho trackbars
        self.sobel_ksize = IntVar(value=3)
        self.sobel_thresh = IntVar(value=50)
        self.laplace_ksize = IntVar(value=3)
        self.laplace_thresh = IntVar(value=50)
        self.canny_low_thresh = IntVar(value=100)
        self.canny_high_thresh = IntVar(value=200)
        self.adapt_block_size = IntVar(value=11)
        self.adapt_c = IntVar(value=2)

        # --- Frame điều khiển ---
        control = Frame(root, width=200, height=600, bg="#ececec")
        control.pack(side=LEFT, fill=Y)
        Label(
            control,
            text="TRẦN VĂN BÁCH - 20226253",
            font=("Arial", 20, "bold"),   # tên font, cỡ chữ, kiểu chữ (optional)
            fg="blue"                     # màu chữ
        ).pack(pady=5)
        Button(control, text="Load Image", command=self.load_image, width=20).pack(pady=10)
        Button(control, text="Sobel Edge", command=self.apply_sobel, width=20).pack(pady=10)
        Button(control, text="Laplacian", command=self.apply_laplacian, width=20).pack(pady=10)
        Button(control, text="Canny Edge", command=self.apply_canny, width=20).pack(pady=10)
        Button(control, text="Adaptive Threshold", command=self.apply_adaptive_threshold, width=20).pack(pady=10)

        # Trackbars with command to update on change
        Scale(control, from_=1, to=31, resolution=2, orient=HORIZONTAL, label="Sobel Ksize", variable=self.sobel_ksize, command=self.on_sobel_change).pack(pady=5)
        Scale(control, from_=0, to=255, orient=HORIZONTAL, label="Sobel Thresh", variable=self.sobel_thresh, command=self.on_sobel_change).pack(pady=5)
        Scale(control, from_=1, to=31, resolution=2, orient=HORIZONTAL, label="Laplace Ksize", variable=self.laplace_ksize, command=self.on_laplacian_change).pack(pady=5)
        Scale(control, from_=0, to=255, orient=HORIZONTAL, label="Laplace Thresh", variable=self.laplace_thresh, command=self.on_laplacian_change).pack(pady=5)
        Scale(control, from_=0, to=255, orient=HORIZONTAL, label="Canny Low Thresh", variable=self.canny_low_thresh, command=self.on_canny_change).pack(pady=5)
        Scale(control, from_=0, to=255, orient=HORIZONTAL, label="Canny High Thresh", variable=self.canny_high_thresh, command=self.on_canny_change).pack(pady=5)


        # --- Frame hiển thị ảnh ---
        self.canvas = Label(root)
        self.canvas.pack(side=RIGHT, expand=True)

    # =============================
    # FUNCTION: Load Image
    # =============================
    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        self.image = cv2.imread(path)
        self.processed = self.image
        self.current_method = None
        self.update_display()

    # =============================
    # FUNCTION: Update Display
    # =============================
    def update_display(self):
        if self.image is None:
            return

        original = self.image
        processed = self.processed if self.processed is not None else original

        # Resize cho hiển thị song song
        orig_resz = cv2.resize(original, (450, 600))

        if self.current_method == 'sobel':
            # Chuẩn bị ảnh phụ cho X và Y (đã ở BGR)
            x_resz = cv2.resize(self.sobel_x_bgr, (225, 300))
            y_resz = cv2.resize(self.sobel_y_bgr, (225, 300))
            sub = np.hstack((x_resz, y_resz))
            proc_resz = cv2.resize(processed, (450, 300))
            right = np.vstack((proc_resz, sub))
        else:
            proc_resz = cv2.resize(processed, (450, 600))
            right = proc_resz

        # Ghép original + right
        combined = np.hstack((orig_resz, right))

        # Chuyển BGR → RGB → PIL
        img_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        # Không cần resize nữa vì đã resize combined thành (900, 600)

        self.display_image = ImageTk.PhotoImage(img_pil)
        self.canvas.config(image=self.display_image)

    # =============================
    # On Change Handlers
    # =============================
    def on_sobel_change(self, val):
        if self.current_method == 'sobel':
            self.apply_sobel()

    def on_laplacian_change(self, val):
        if self.current_method == 'laplacian':
            self.apply_laplacian()

    def on_canny_change(self, val):
        if self.current_method == 'canny':
            self.apply_canny()

    def on_adaptive_change(self, val):
        if self.current_method == 'adaptive':
            self.apply_adaptive_threshold()

    # =============================
    # SOBEL
    # =============================
    def apply_sobel(self):
        if self.image is None:
            return

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ksize = self.sobel_ksize.get()
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

        sobel = cv2.magnitude(sobel_x, sobel_y)

        # Normalize to 0-255
        sobel_norm = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
        sobel_norm = np.uint8(sobel_norm)

        # Apply threshold
        thresh_val = self.sobel_thresh.get()
        _, sobel_thresh = cv2.threshold(sobel_norm, thresh_val, 255, cv2.THRESH_BINARY)

        # Ảnh phụ cho X và Y (normalize similarly)
        sobel_x_abs = cv2.normalize(np.absolute(sobel_x), None, 0, 255, cv2.NORM_MINMAX)
        sobel_x_abs = np.uint8(sobel_x_abs)
        sobel_y_abs = cv2.normalize(np.absolute(sobel_y), None, 0, 255, cv2.NORM_MINMAX)
        sobel_y_abs = np.uint8(sobel_y_abs)
        self.sobel_x_bgr = cv2.cvtColor(sobel_x_abs, cv2.COLOR_GRAY2BGR)
        self.sobel_y_bgr = cv2.cvtColor(sobel_y_abs, cv2.COLOR_GRAY2BGR)

        self.processed = cv2.cvtColor(sobel_thresh, cv2.COLOR_GRAY2BGR)
        self.current_method = 'sobel'
        self.update_display()

    # =============================
    # LAPLACIAN
    # =============================
    def apply_laplacian(self):
        if self.image is None:
            return

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Giảm nhiễu trước Laplacian
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        ksize = self.laplace_ksize.get()
        laplace = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        laplace_abs = cv2.convertScaleAbs(laplace)

        # Apply threshold
        thresh_val = self.laplace_thresh.get()
        _, laplace_thresh = cv2.threshold(laplace_abs, thresh_val, 255, cv2.THRESH_BINARY)

        self.processed = cv2.cvtColor(laplace_thresh, cv2.COLOR_GRAY2BGR)
        self.current_method = 'laplacian'
        self.update_display()

    # =============================
    # CANNY
    # =============================
    def apply_canny(self):
        if self.image is None:
            return

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        low = self.canny_low_thresh.get()
        high = self.canny_high_thresh.get()
        canny = cv2.Canny(gray, low, high)

        self.processed = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        self.current_method = 'canny'
        self.update_display()

    # =============================
    # ADAPTIVE THRESHOLD
    # =============================
    
    def apply_adaptive_threshold(self):
        if self.image is None:
            return

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Initial T: average intensity
        T = np.mean(gray)

        while True:
            # Group 1: pixels > T
            g1 = gray[gray > T]
            # Group 2: pixels <= T
            g2 = gray[gray <= T]

            if len(g1) == 0 or len(g2) == 0:
                break

            m1 = np.mean(g1)
            m2 = np.mean(g2)

            new_T = (m1 + m2) / 2

            if abs(new_T - T) < 1:  # Convergence threshold
                break

            T = new_T

        # Apply threshold
        _, thresh = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)

        self.processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.current_method = 'adaptive'
        self.update_display()

# =============================
# CHẠY CHƯƠNG TRÌNH
# =============================
root = Tk()
app = EdgeGUI(root)
root.mainloop()