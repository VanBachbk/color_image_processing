import cv2
import numpy as np
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# ----------------------- FREQUENCY DOMAIN FUNCTIONS -----------------------
def to_float_img(img):
    """Chuyển ảnh uint8 -> float32 trong khoảng [0,1]."""
    return img.astype(np.float32) / 255.0

def to_uint8_img(img):
    """Chuyển ảnh float -> uint8 (clamp 0..1)."""
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def fft2c(img):
    """2D FFT với shift (centered)."""
    return np.fft.fftshift(np.fft.fft2(img))

def ifft2c(fft_img):
    """Inverse 2D FFT từ centered FFT."""
    return np.fft.ifft2(np.fft.ifftshift(fft_img))

def make_meshgrid(h, w):
    """Tạo lưới toạ độ (u,v) có tâm ở giữa."""
    cy, cx = h // 2, w // 2
    y = np.arange(0, h) - cy
    x = np.arange(0, w) - cx
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    return D, X, Y

def ideal_lowpass_mask(shape, cutoff):
    h, w = shape
    D, _, _ = make_meshgrid(h, w)
    H = np.zeros_like(D)
    H[D <= cutoff] = 1.0
    return H

def butterworth_lowpass_mask(shape, cutoff, order=2):
    h, w = shape
    D, _, _ = make_meshgrid(h, w)
    H = 1.0 / (1.0 + (D / (cutoff + 1e-9)) ** (2 * order))
    return H

def gaussian_lowpass_mask(shape, sigma):
    h, w = shape
    D, _, _ = make_meshgrid(h, w)
    H = np.exp(-(D**2) / (2 * (sigma**2 + 1e-9)))
    return H

def laplacian_freq_mask(shape):
    h, w = shape
    D, X, Y = make_meshgrid(h, w)
    nx = X / float(w)
    ny = Y / float(h)
    freq2 = nx**2 + ny**2
    H = -4 * (np.pi**2) * freq2
    max_abs = np.max(np.abs(H)) + 1e-9
    H = H / max_abs
    return H

def apply_frequency_filter_channel(channel, H):
    """SỬA: Xử lý tốt hơn cho filter có giá trị âm"""
    f = fft2c(channel)
    f_filtered = f * H
    img_back = np.real(ifft2c(f_filtered))
    
    # Nếu mask có giá trị âm (Laplacian/Highpass) -> normalize khác
    if np.min(H) < -0.1:  # Laplacian
        img_back = cv2.normalize(img_back, None, -128, 127, cv2.NORM_MINMAX)
        img_back = np.clip(img_back + 128, 0, 255).astype(np.uint8)
    else:
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        img_back = np.uint8(img_back)
    
    return img_back

def process_color_by_channel(img_float, mask_func, *mask_args, **mask_kwargs):
    """Áp dụng mask (mask_func(shape, ...)) cho từng kênh và trả về ảnh kết quả."""
    h, w = img_float.shape[:2]
    if img_float.ndim == 2:  # grayscale
        H = mask_func((h, w), *mask_args, **mask_kwargs) if mask_func is not laplacian_freq_mask else mask_func((h, w))
        out = apply_frequency_filter_channel(img_float, H)
        return out
    out_channels = []
    for c in range(3):
        ch = img_float[..., c]
        H = mask_func((h, w), *mask_args, **mask_kwargs) if mask_func is not laplacian_freq_mask else mask_func((h, w))
        out_ch = apply_frequency_filter_channel(ch, H)
        out_channels.append(out_ch)
    out = np.stack(out_channels, axis=-1)
    return out

def highpass_from_lowpass_mask(lowpass_mask):
    return 1.0 - lowpass_mask

def sharpen_with_laplacian_freq(img_float, alpha=1.0):
    """
    Sharpen by adding -alpha * Laplacian(img) in spatial domain is equivalent to
    adding alpha * (-H_lap) * F(u,v) in frequency domain.
    """
    h, w = img_float.shape[:2]
    H_lap = laplacian_freq_mask((h, w))
    combined_mask = 1.0 - alpha * H_lap
    if img_float.ndim == 2:
        res = apply_frequency_filter_channel(img_float, combined_mask)
        return res
    out = []
    for c in range(3):
        res_ch = apply_frequency_filter_channel(img_float[..., c], combined_mask)
        out.append(res_ch)
    return np.stack(out, axis=-1)

def hybrid_image(img_low_float, img_high_float, low_mask_func, low_args=(), high_mask_func=None, high_args=()):
    h, w = img_low_float.shape[:2]
    H_low = low_mask_func((h, w), *low_args) if low_mask_func is not laplacian_freq_mask else low_mask_func((h, w))
    if high_mask_func is None:
        H_high = 1.0 - H_low
    else:
        H_high = high_mask_func((h, w), *high_args)
    def apply_mask(img, H):
        if img.ndim == 2:
            return apply_frequency_filter_channel(img, H)
        out_ch = []
        for c in range(3):
            out_ch.append(apply_frequency_filter_channel(img[..., c], H))
        return np.stack(out_ch, axis=-1)
    low = apply_mask(img_low_float, H_low)
    high = apply_mask(img_high_float, H_high)
    hybrid = low + high
    return hybrid, low, high

# ----------------------- HYBRID IMAGE -----------------------
# def make_hybrid(img1, img2, cutoff_low=30, cutoff_high=15,
#                 method='gaussian', bw_order=2, progress_callback=None):
#     """
#     Tạo ảnh lai giữa img1 (low frequency) và img2 (high frequency)
#     """
#     # Chuyển ảnh về grayscale nếu cần
#     if img1.ndim == 3:
#         img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     if img2.ndim == 3:
#         img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     # Resize ảnh 2 cho trùng kích thước
#     if img1.shape != img2.shape:
#         img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

#     # Tạo mặt nạ lọc
#     low_mask_func = gaussian_lowpass_mask
#     high_mask_func = lambda shape, D0: 1 - gaussian_lowpass_mask(shape, D0)

#     # FFT
#     F1 = np.fft.fftshift(np.fft.fft2(img1))
#     F2 = np.fft.fftshift(np.fft.fft2(img2))

#     # Áp dụng lọc
#     H_low = low_mask_func(img1.shape, cutoff_low)
#     H_high = high_mask_func(img2.shape, cutoff_high)

#     F1_low = F1 * H_low
#     F2_high = F2 * H_high

#     # Biến đổi ngược
#     low = np.fft.ifft2(np.fft.ifftshift(F1_low))
#     high = np.fft.ifft2(np.fft.ifftshift(F2_high))

#     # Ghép lại hybrid image
#     hybrid = np.real(low) + np.real(high)
#     hybrid = np.clip(hybrid, 0, 255).astype(np.uint8)

#     # Chuẩn hóa các ảnh kết quả để hiển thị
#     low_disp = cv2.normalize(np.real(low), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     high_disp = cv2.normalize(np.real(high), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#     if progress_callback:
#         progress_callback(100)

#     return low_disp, high_disp, hybrid
def apply_frequency_filter_channel(channel, H):
    """
    Áp dụng bộ lọc tần số H cho một kênh ảnh (float32, 0..1).
    Args:
        channel: Kênh ảnh đầu vào (2D, float32, trong khoảng [0,1]).
        H: Mặt nạ tần số (2D, cùng kích thước với channel).
    Returns:
        Kênh ảnh sau lọc (uint8, 0..255), đã chuẩn hóa để hiển thị đúng.
    """
    # Thực hiện FFT với shift
    f = fft2c(channel)
    
    # Áp dụng mặt nạ tần số
    f_filtered = f * H
    
    # Biến đổi ngược IFFT
    img_back = ifft2c(f_filtered)
    
    # Lấy phần thực và chuẩn hóa
    img_back = np.real(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    
    return img_back
def make_hybrid(img1, img2, cutoff_low=30, cutoff_high=15,
                method='gaussian', bw_order=2, progress_callback=None):

    # Resize ảnh 2 để khớp kích thước
    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    # # Chuyển sang ảnh xám (đen trắng)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Nếu ảnh có 3 kênh, chuyển sang ảnh xám (tùy mục đích)
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray, img2_gray = img1, img2

    # Chuyển sang float [0,1]
    img1_float = to_float_img(img1_gray)
    img2_float = to_float_img(img2_gray)

    # Chọn loại bộ lọc
    if method == 'ideal':
        low_mask_func = ideal_lowpass_mask
        high_mask_func = lambda shape, D0: 1 - ideal_lowpass_mask(shape, D0)
    elif method == 'butterworth':
        low_mask_func = lambda shape, D0: butterworth_lowpass_mask(shape, D0, bw_order)
        high_mask_func = lambda shape, D0: 1 - butterworth_lowpass_mask(shape, D0, bw_order)
    else:  # Gaussian
        low_mask_func = gaussian_lowpass_mask
        high_mask_func = lambda shape, D0: 1 - gaussian_lowpass_mask(shape, D0)

    alpha = 0.7  # Trọng số pha trộn giữa low-pass và high-pass

    # Kiểm tra nếu là ảnh màu (3 kênh)
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        low_channels, high_channels, hybrid_channels = [], [], []
        for c in range(3):
            ch1 = to_float_img(img1[:, :, c])
            ch2 = to_float_img(img2[:, :, c])

            H_low = low_mask_func((h, w), cutoff_low)
            H_high = high_mask_func((h, w), cutoff_high)

            f1 = fft2c(ch1)
            f2 = fft2c(ch2)

            low_ch = np.real(ifft2c(f1 * H_low))
            high_ch = np.real(ifft2c(f2 * H_high))

            # Chuẩn hóa về 0..1
            low_ch = (low_ch - low_ch.min()) / (low_ch.max() - low_ch.min() + 1e-8)
            high_ch = (high_ch - high_ch.min()) / (high_ch.max() - high_ch.min() + 1e-8)

            hybrid_ch = np.clip(low_ch + alpha * high_ch, 0, 1)

            low_channels.append(low_ch)
            high_channels.append(high_ch)
            hybrid_channels.append(hybrid_ch)

            if progress_callback:
                progress_callback(33.33 * (c + 1))

        low = np.stack(low_channels, axis=-1)
        high = np.stack(high_channels, axis=-1)
        hybrid = np.stack(hybrid_channels, axis=-1)

    else:
        # Ảnh xám (1 kênh)
        H_low = low_mask_func((h, w), cutoff_low)
        H_high = high_mask_func((h, w), cutoff_high)

        f1 = fft2c(img1_float)
        f2 = fft2c(img2_float)

        low = np.real(ifft2c(f1 * H_low))
        high = np.real(ifft2c(f2 * H_high))

        low = (low - low.min()) / (low.max() - low.min() + 1e-8)
        high = (high - high.min()) / (high.max() - high.min() + 1e-8)

        hybrid = np.clip(low + alpha * high, 0, 1)

        if progress_callback:
            progress_callback(100)

    # Đưa về uint8 để hiển thị
    low = (low * 255).astype(np.uint8)
    high = (high * 255).astype(np.uint8)
    hybrid = (hybrid * 255).astype(np.uint8)

    return low, high, hybrid

# ----------------------- GUI & HIỂN THỊ -----------------------
class FilterGUI:
    def __init__(self, root):
        self.root = root
        root.title("Image Filters - Frequency & Spatial Domain (Hybrid)")
        root.geometry("1300x900")
        self.img1 = None
        self.img2 = None
        self.proc_img = None
        top = Frame(root)
        top.pack(pady=8)
        Label(top, text="Trần Văn Bách - Image Filters", font=("Arial", 18, "bold"), fg="blue").grid(row=0, column=0, columnspan=6, sticky=W, padx=8)
        Button(top, text="📁 Chọn ảnh 1", command=self.load_image1).grid(row=1, column=0, padx=5, pady=6)
        Button(top, text="📁 Chọn ảnh 2 (dùng cho Hybrid)", command=self.load_image2).grid(row=1, column=1, padx=5, pady=6)
        Button(top, text="🔄 Reset", command=self.reset_images).grid(row=1, column=2, padx=5)
        Button(top, text="Áp dụng bộ lọc", command=self.apply_filter).grid(row=1, column=3, padx=5)
        Button(top, text="Lưu ảnh kết quả", command=self.save_result).grid(row=1, column=4, padx=5)
        control = LabelFrame(root, text="Control")
        control.pack(pady=6, fill=X, padx=10)
        Label(control, text="Chọn bộ lọc:").grid(row=0, column=0, sticky=W, padx=6, pady=4)
        self.filter_combo = ttk.Combobox(control, values=[
            "Ideal Lowpass (frequency)",
            "Butterworth Lowpass (frequency)",
            "Gaussian Lowpass (frequency)",
            "Laplacian Filter (Freq)",
            "Sharpening Filter (Freq)",
            "Hybrid Image"
        ], width=35)
        self.filter_combo.set("Gaussian Lowpass (frequency)")
        self.filter_combo.grid(row=0, column=1, padx=6, pady=4)
        Label(control, text="Cutoff D0 (px):").grid(row=1, column=0, sticky=W, padx=6)
        self.cutoff_var = IntVar(value=30)
        Entry(control, textvariable=self.cutoff_var, width=6).grid(row=1, column=1, sticky=W)
        Label(control, text="Cutoff High (hybrid):").grid(row=1, column=2, sticky=W)
        self.cutoff_high_var = IntVar(value=15)
        Entry(control, textvariable=self.cutoff_high_var, width=6).grid(row=1, column=3, sticky=W)
        Label(control, text="Butterworth order:").grid(row=1, column=4, sticky=W)
        self.bw_order_var = IntVar(value=2)
        Entry(control, textvariable=self.bw_order_var, width=4).grid(row=1, column=5, sticky=W)
        Label(control, text="Hiển thị spectrum:").grid(row=2, column=2, sticky=W)
        self.show_spec_var = BooleanVar(value=False)
        Checkbutton(control, variable=self.show_spec_var).grid(row=2, column=3, sticky=W)
        self.progress = ttk.Progressbar(control, orient=HORIZONTAL, length=200, mode='determinate')
        self.progress.grid(row=2, column=4, columnspan=2, padx=6, pady=4, sticky=W)
        display = Frame(root)
        display.pack(padx=10, pady=6, fill=BOTH, expand=True)
        self.frame_orig = LabelFrame(display, text="Ảnh gốc (click để phóng to)")
        self.frame_orig.pack(side=LEFT, padx=6, pady=6, fill=BOTH, expand=True)
        self.frame_proc = LabelFrame(display, text="Kết quả (click để phóng to)")
        self.frame_proc.pack(side=RIGHT, padx=6, pady=6, fill=BOTH, expand=True)
        self.orig_canvases = []
        for i in range(2):
            lbl = Label(self.frame_orig)
            lbl.pack(padx=6, pady=6)
            lbl.bind("<Button-1>", lambda e, idx=i: self.show_full_original(idx))
            self.orig_canvases.append(lbl)
        self.proc_canvases = []
        for i in range(3):
            lbl = Label(self.frame_proc)
            lbl.pack(padx=6, pady=6)
            lbl.bind("<Button-1>", lambda e, idx=i: self.show_full_processed(idx))
            self.proc_canvases.append(lbl)
        self.status = Label(root, text="Ready", anchor=W)
        self.status.pack(fill=X, padx=8, pady=4)

    def load_image1(self):
        path = filedialog.askopenfilename(title="Chọn ảnh 1", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Không đọc được file ảnh.")
            return
        self.img1 = img
        self.show_originals()
        self.status.config(text=f"Loaded image1: {path}")

    def load_image2(self):
        path = filedialog.askopenfilename(title="Chọn ảnh 2", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Không đọc được file ảnh.")
            return
        self.img2 = img
        self.show_originals()
        self.status.config(text=f"Loaded image2: {path}")

    def reset_images(self):
        self.img1 = None
        self.img2 = None
        self.proc_img = None
        self.show_originals()
        self.show_processed([])
        self.status.config(text="Reset images.")

    def save_result(self):
        if self.proc_img is None:
            messagebox.showinfo("Info", "Chưa có ảnh kết quả để lưu.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG","*.jpg")])
        if not path:
            return
        if isinstance(self.proc_img, tuple):
            # Save the hybrid image (third component)
            cv2.imwrite(path, self.proc_img[2])
        else:
            cv2.imwrite(path, self.proc_img)
        messagebox.showinfo("Saved", f"Đã lưu ảnh kết quả: {path}")

    def cv2_to_tk(self, img, maxsize=(400, 400)):
        bgr = img
        h, w = bgr.shape[:2]
        scale = min(maxsize[0] / w, maxsize[1] / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
        return ImageTk.PhotoImage(pil)

    def show_originals(self):
        for lbl in self.orig_canvases:
            lbl.config(image='', text='(empty)')
            lbl.image = None
        if self.img1 is not None:
            tk1 = self.cv2_to_tk(self.img1, maxsize=(380, 280))
            self.orig_canvases[0].config(image=tk1, text='')
            self.orig_canvases[0].image = tk1
        else:
            self.orig_canvases[0].config(text="Ảnh 1: (chưa chọn)")
        if self.img2 is not None:
            tk2 = self.cv2_to_tk(self.img2, maxsize=(380, 280))
            self.orig_canvases[1].config(image=tk2, text='')
            self.orig_canvases[1].image = tk2
        else:
            self.orig_canvases[1].config(text="Ảnh 2: (chưa chọn)")

    def show_processed(self, imgs):
        for i in range(3):
            lbl = self.proc_canvases[i]
            if i < len(imgs) and imgs[i] is not None:
                tk = self.cv2_to_tk(imgs[i], maxsize=(400, 300))
                lbl.config(image=tk, text='')
                lbl.image = tk
            else:
                lbl.config(image='', text='(empty)')
                lbl.image = None

    def show_full_original(self, idx):
        img = self.img1 if idx == 0 else self.img2
        if img is None:
            return
        self._popup_image(img, title=f"Original {idx+1}")

    def show_full_processed(self, idx):
        if self.proc_img is None:
            return
        if isinstance(self.proc_img, tuple):
            imgs = self.proc_img
            if idx < len(imgs) and imgs[idx] is not None:
                self._popup_image(imgs[idx], title=f"Result {idx+1}")
        else:
            if idx == 0:
                self._popup_image(self.proc_img, title="Result")

    def _popup_image(self, img, title="Image"):
        win = Toplevel(self.root)
        win.title(title)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(900 / w, 700 / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        pil = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(pil)
        lbl = Label(win, image=img_tk)
        lbl.image = img_tk
        lbl.pack()
        btn = Button(win, text="Đóng", command=win.destroy)
        btn.pack(pady=6)

    def apply_filter(self):
        if self.img1 is None:
            messagebox.showinfo("Info", "Vui lòng chọn ít nhất ảnh 1.")
            return
        self.progress['value'] = 0
        self.root.update()
        def update_progress(value):
            self.progress['value'] = value
            self.root.update()
        try:
            cutoff = max(1, int(self.cutoff_var.get()))
            cutoff_high = max(1, int(self.cutoff_high_var.get()))
            bw_order = max(1, int(self.bw_order_var.get()))
        except ValueError:
            messagebox.showerror("Lỗi", "Vui lòng nhập số nguyên hợp lệ cho Cutoff và Order.")
            return
        choice = self.filter_combo.get()
        img_float = to_float_img(self.img1)
        
        if choice.startswith("Ideal Lowpass"):
            out = process_color_by_channel(img_float, ideal_lowpass_mask, cutoff)
            self.proc_img = out
            self.show_processed([out])
            self.status.config(text=f"Ideal Lowpass áp dụng với D0={cutoff}")
            if self.show_spec_var.get():
                spec = self._spectrum_image(self.img1, ideal_lowpass_mask(self.img1.shape[:2], cutoff))
                self._show_spectrum_popup(spec, f"Ideal spectrum D0={cutoff}")
            update_progress(100)
        elif choice.startswith("Butterworth"):
            out = process_color_by_channel(img_float, butterworth_lowpass_mask, cutoff, order=bw_order)
            self.proc_img = out
            self.show_processed([out])
            self.status.config(text=f"Butterworth Lowpass D0={cutoff}, order={bw_order}")
            if self.show_spec_var.get():
                spec = self._spectrum_image(self.img1, butterworth_lowpass_mask(self.img1.shape[:2], cutoff, bw_order))
                self._show_spectrum_popup(spec, f"Butterworth spectrum D0={cutoff}, n={bw_order}")
            update_progress(100)
        elif choice.startswith("Gaussian Lowpass"):
            out = process_color_by_channel(img_float, gaussian_lowpass_mask, cutoff)
            self.proc_img = out
            self.show_processed([out])
            self.status.config(text=f"Gaussian Lowpass D0={cutoff}")
            if self.show_spec_var.get():
                spec = self._spectrum_image(self.img1, gaussian_lowpass_mask(self.img1.shape[:2], cutoff))
                self._show_spectrum_popup(spec, f"Gaussian spectrum D0={cutoff}")
            update_progress(100)
        elif choice == "Laplacian Filter (Freq)":
            out = process_color_by_channel(img_float, laplacian_freq_mask)
            self.proc_img = out
            self.show_processed([out])
            self.status.config(text="Laplacian Filter (Frequency Domain)")
            if self.show_spec_var.get():
                spec = self._spectrum_image(self.img1, laplacian_freq_mask(self.img1.shape[:2]))
                self._show_spectrum_popup(spec, "Laplacian spectrum")
            update_progress(100)
        elif choice == "Sharpening Filter (Freq)":
            out = sharpen_with_laplacian_freq(img_float, alpha=1)
            self.proc_img = out
            self.show_processed([out])
            self.status.config(text="Sharpening Filter (Frequency Domain)")
            if self.show_spec_var.get():
                H_lap = laplacian_freq_mask(self.img1.shape[:2])
                H_sharp = 1.0 - 0.8 * H_lap
                spec = self._spectrum_image(self.img1, H_sharp)
                self._show_spectrum_popup(spec, "Sharpening spectrum")
            update_progress(100)
        elif choice.startswith("Hybrid"):
            if self.img2 is None:
                messagebox.showinfo("Info", "Hybrid yêu cầu ảnh 2. Vui lòng chọn ảnh 2.")
                return
            method = 'gaussian'
            low, high, hybrid = make_hybrid(self.img1, self.img2, cutoff_low=cutoff,
                                           cutoff_high=cutoff_high,
                                         bw_order=bw_order,
                                           progress_callback=update_progress)
            self.proc_img = (low, high, hybrid)
            self.show_processed([hybrid])
            self.status.config(text=f"Hybrid created (low D0={cutoff}, high D0={cutoff_high})")
        else:
            messagebox.showwarning("Warning", "Lựa chọn không hợp lệ.")
        self.progress['value'] = 100
        self.root.update()

    def _spectrum_image(self, img, H=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_float = to_float_img(gray)
        F = fft2c(img_float)
        mag = np.log1p(np.abs(F))
        mag = mag - mag.min()
        mag = mag / (mag.max() + 1e-8) * 255.0
        mag = mag.astype(np.uint8)
        mag_color = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
        if H is not None:
            # Chuẩn hóa mặt nạ cho hiển thị, đặc biệt xử lý mặt nạ âm (như Laplacian)
            if np.min(H) < 0:  # Xử lý mặt nạ âm (Laplacian)
                Hvis = np.abs(H)  # Lấy giá trị tuyệt đối để hiển thị tần số cao
                Hvis = (Hvis - Hvis.min()) / (Hvis.max() - Hvis.min() + 1e-8)
            else:
                Hvis = (H - H.min()) / (H.max() - H.min() + 1e-8)
            Hvis = (Hvis * 255).astype(np.uint8)
            Hcolor = cv2.applyColorMap(Hvis, cv2.COLORMAP_JET)
            over = cv2.addWeighted(mag_color, 0.6, Hcolor, 0.4, 0)
            return over
        return mag_color

    def _show_spectrum_popup(self, img, title="Spectrum"):
        win = Toplevel(self.root)
        win.title(title)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(900 / w, 700 / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        pil = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(pil)
        lbl = Label(win, image=img_tk)
        lbl.image = img_tk
        lbl.pack()
        Button(win, text="Đóng", command=win.destroy).pack(pady=6)

# ----------------------- MAIN -----------------------
if __name__ == "__main__":
    root = Tk()
    app = FilterGUI(root)
    root.mainloop()