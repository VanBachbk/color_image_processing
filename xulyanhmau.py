import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
class ColorProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Processing - Smoothing & Color Detection")
        self.root.geometry("1800x1000")
        # Color reference & thresholds
        self.ref_r = tk.IntVar(value=255)
        self.ref_g = tk.IntVar(value=20)
        self.ref_b = tk.IntVar(value=20)
        self.threshold_w1 = tk.IntVar(value=10000)
        self.threshold_w = tk.IntVar(value=10000) # D² for RGB, Min S for HSI
        self.gamma_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.BooleanVar(value=False)
        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        self.running = False
        self.click_bound_tab1 = False
        self.click_bound_tab3 = False
        self.setup_ui()
        self.update_frame()
    def setup_ui(self):
        # Control buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        self.name = tk.Label(self.root, text="Trần Văn Bách",font=("Times New Roman", 16, "bold"), fg="Red")
        self.name.pack(pady=5)
        self.mssv = tk.Label(self.root, text="MSSV: 20226253",font=("Times New Roman", 16, "bold"), fg="Red")
        self.mssv.pack(pady=2)
        tk.Button(btn_frame, text="Start Webcam", command=self.start_webcam).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Stop Webcam", command=self.stop_webcam).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.frame1 = ttk.Frame(self.notebook)
        self.frame2 = ttk.Frame(self.notebook)
        self.frame3 = ttk.Frame(self.notebook)
        self.frame4 = ttk.Frame(self.notebook)
        self.notebook.add(self.frame1, text="1. Color Space")
        self.notebook.add(self.frame2, text="2. Smoothing")
        self.notebook.add(self.frame3, text="3. Color Detection")
        self.notebook.add(self.frame4, text="4. Color Coding")
        self.setup_color_space_tab()
        self.setup_smoothing_tab()
        self.setup_detection_tab()
        self.setup_color_coding_tab()
        # Control buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Start Webcam", command=self.start_webcam).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Stop Webcam", command=self.stop_webcam).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        
    # ====================== TAB 1: Color Space ======================
    def setup_color_space_tab(self):
        self.color_combo = ttk.Combobox(self.frame1, values=[
            "BGR", "RGB", "HSV", "LAB", "Gray",
            "Color Complement", "Color Slicing", "Tonal Correction"
        ], state="readonly")
        self.color_combo.set("BGR")
        self.color_combo.pack(pady=5)
        self.color_combo.bind('<<ComboboxSelected>>', self.on_color_combo_selected)
        # Tonal Correction
        self.tonal_frame = tk.LabelFrame(self.frame1, text="Tonal Correction", font=("Arial", 10, "bold"))
        tk.Label(self.tonal_frame, text="Gamma:").grid(row=0, column=0, padx=5, pady=5)
        self.gamma_scale = tk.Scale(self.tonal_frame, from_=0.1, to=5.0, resolution=0.1,
                                    orient=tk.HORIZONTAL, variable=self.gamma_var, length=200,
                                    command=lambda v: self.update_converted())
        self.gamma_scale.grid(row=0, column=1, padx=5, pady=5)
        tk.Checkbutton(self.tonal_frame, text="Contrast Enhancement", variable=self.contrast_var,
                       command=self.update_converted).grid(row=1, column=0, columnspan=2, pady=5)
        tk.Button(self.tonal_frame, text="Reset", command=self.reset_tonal).grid(row=1, column=2, padx=5, pady=5)
        # Color Slicing
        self.slicing_frame = tk.LabelFrame(self.frame1, text="Color Slicing Controls", font=("Arial", 10, "bold"))
        tk.Label(self.slicing_frame, text="Reference R:").grid(row=0, column=0, padx=5, pady=5)
        self.ref_r_scale = tk.Scale(self.slicing_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                    variable=self.ref_r, length=200,
                                    command=lambda v: self.update_converted())
        self.ref_r_scale.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(self.slicing_frame, text="Reference G:").grid(row=1, column=0, padx=5, pady=5)
        self.ref_g_scale = tk.Scale(self.slicing_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                    variable=self.ref_g, length=200,
                                    command=lambda v: self.update_converted())
        self.ref_g_scale.grid(row=1, column=1, padx=5, pady=5)
        tk.Label(self.slicing_frame, text="Reference B:").grid(row=2, column=0, padx=5, pady=5)
        self.ref_b_scale = tk.Scale(self.slicing_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                    variable=self.ref_b, length=200,
                                    command=lambda v: self.update_converted())
        self.ref_b_scale.grid(row=2, column=1, padx=5, pady=5)
        tk.Label(self.slicing_frame, text="Threshold W:").grid(row=3, column=0, padx=5, pady=5)
        self.threshold_scale = tk.Scale(self.slicing_frame, from_=0, to=20000, orient=tk.HORIZONTAL,
                                        variable=self.threshold_w1, length=200,
                                        command=lambda v: self.update_converted())
        self.threshold_scale.grid(row=3, column=1, padx=5, pady=5)
        self.slicing_info = tk.Label(self.slicing_frame, text="Click on Original image to pick color", fg="red")
        self.slicing_info.grid(row=4, column=0, columnspan=2, pady=5)
        tk.Button(self.slicing_frame, text="Reset", command=self.reset_slicing).grid(row=3, column=2, padx=5, pady=5)
        # Image grid
        self.img_labels1 = []
        layout = [["Original", "Converted"], ["R", "G", "B"], ["C", "M", "Y"], ["H", "S", "I"]]
        for row in layout:
            row_frame = tk.Frame(self.frame1)
            row_frame.pack(pady=5)
            for txt in row:
                col_frame = tk.Frame(row_frame)
                col_frame.pack(side=tk.LEFT, padx=10)
                tk.Label(col_frame, text=txt, font=("Arial", 9, "bold")).pack()
                img_lbl = tk.Label(col_frame)
                img_lbl.pack()
                img_lbl.bind("<Button-1>", self.show_large_image)
                self.img_labels1.append(img_lbl)
    def on_color_combo_selected(self, event):
        color = self.color_combo.get()
        if color == "Tonal Correction":
            self.tonal_frame.pack(pady=5, fill=tk.X, after=self.color_combo)
        else:
            self.tonal_frame.pack_forget()
        if color == "Color Slicing":
            self.slicing_frame.pack(pady=5, fill=tk.X, after=self.color_combo)
            if not self.click_bound_tab1:
                self.img_labels1[0].bind("<Button-1>", self.on_image_click_tab1)
                self.click_bound_tab1 = True
        else:
            self.slicing_frame.pack_forget()
            if self.click_bound_tab1:
                self.img_labels1[0].unbind("<Button-1>")
                self.click_bound_tab1 = False
        self.update_converted()
    def on_image_click_tab1(self, event):
        self.on_image_click(event, is_tab1=True)
    def on_image_click_tab3(self, event):
        self.on_image_click(event, is_tab1=False)
    def on_image_click(self, event, is_tab1=True):
        if self.current_frame is None: return
        display_w, display_h = 150, 113
        h, w = self.current_frame.shape[:2]
        x = int(event.x * w / display_w)
        y = int(event.y * h / display_h)
        x, y = max(0, min(x, w-1)), max(0, min(y, h-1))
        b, g, r = self.current_frame[y, x]
        self.ref_b.set(b)
        self.ref_g.set(g)
        self.ref_r.set(r)
        if is_tab1:
            self.update_converted()
        else:
            self.process_detection()
    def reset_tonal(self):
        self.gamma_var.set(1.0)
        self.contrast_var.set(False)
        self.update_converted()
    def reset_slicing(self):
        self.ref_r.set(255)
        self.ref_g.set(20)
        self.ref_b.set(20)
        self.threshold_w1.set(50)
        self.update_converted()
    def apply_tonal_correction(self, frame):
        img = frame.astype(np.float32) / 255.0
        if self.contrast_var.get():
            min_val, max_val = np.min(img), np.max(img)
            if max_val > min_val:
                img = (img - min_val) / (max_val - min_val)
        gamma = self.gamma_var.get()
        img = np.power(img, gamma)
        return (img * 255).astype(np.uint8)
    # ====================== TAB 2: Smoothing ======================
    def setup_smoothing_tab(self):
        tk.Label(self.frame2, text="Kernel Size:").pack()
        self.kernel_scale = tk.Scale(self.frame2, from_=1, to=31, orient=tk.HORIZONTAL, length=200,
                                     command=lambda v: self.process_tab2())
        self.kernel_scale.set(5)
        self.kernel_scale.pack(pady=5)
        self.filter_combo = ttk.Combobox(self.frame2, values=["Average", "Gaussian", "Median"], state="readonly")
        self.filter_combo.set("Gaussian")
        self.filter_combo.pack(pady=5)
        self.filter_combo.bind('<<ComboboxSelected>>', lambda e: self.process_tab2())
        self.img_labels2 = []
        layout = [["Original", "Method 1 (Per-plane)", "Method 2 (HSI I)", "Difference"]]
        for row in layout:
            row_frame = tk.Frame(self.frame2)
            row_frame.pack(pady=5)
            for txt in row:
                col_frame = tk.Frame(row_frame)
                col_frame.pack(side=tk.LEFT, padx=10)
                tk.Label(col_frame, text=txt, font=("Arial", 9, "bold")).pack()
                img_lbl = tk.Label(col_frame)
                img_lbl.pack()
                img_lbl.bind("<Button-1>", self.show_large_image)
                self.img_labels2.append(img_lbl)
    def process_tab2(self):
        if self.current_frame is None: return
        frame = self.current_frame.copy()
        k = self.kernel_scale.get()
        if k % 2 == 0: k += 1
        filter_type = self.filter_combo.get()
        def apply_filter(ch):
            if filter_type == "Average": return cv2.blur(ch, (k, k))
            if filter_type == "Gaussian": return cv2.GaussianBlur(ch, (k, k), 0)
            return cv2.medianBlur(ch, k)
        method1 = np.zeros_like(frame)
        for c in range(3):
            method1[:, :, c] = apply_filter(frame[:, :, c])
        h, s, i = self.bgr_to_hsi(frame)
        i_smooth = apply_filter(i)
        method2 = self.hsi_to_bgr(h, s, i_smooth)
        diff = cv2.absdiff(method1, method2)
        size = (150, 113)
        photos = [
            ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(size)),
            ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(method1, cv2.COLOR_BGR2RGB)).resize(size)),
            ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(method2, cv2.COLOR_BGR2RGB)).resize(size)),
            ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)).resize(size))
        ]
        for i, photo in enumerate(photos):
            self.img_labels2[i].config(image=photo)
            self.img_labels2[i].image = photo
    # ====================== TAB 3: Color Detection ======================
    def setup_detection_tab(self):
        self.detection_combo = ttk.Combobox(self.frame3, values=["HSI Segmentation", "RGB Vector Segmentation"], state="readonly")
        self.detection_combo.set("HSI Segmentation")
        self.detection_combo.pack(pady=5)
        self.detection_combo.bind('<<ComboboxSelected>>', lambda e: self.update_detection_controls())
        self.detection_frame = tk.LabelFrame(self.frame3, text="Detection Controls", font=("Arial", 10, "bold"))
        self.detection_frame.pack(pady=5, fill=tk.X)
        # Reference color (shared)
        tk.Label(self.detection_frame, text="Reference R:").grid(row=0, column=0, padx=5, pady=5)
        self.ref_r_scale_tab3 = tk.Scale(self.detection_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                         variable=self.ref_r, length=200,
                                         command=lambda v: self.process_detection())
        self.ref_r_scale_tab3.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(self.detection_frame, text="Reference G:").grid(row=1, column=0, padx=5, pady=5)
        self.ref_g_scale_tab3 = tk.Scale(self.detection_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                         variable=self.ref_g, length=200,
                                         command=lambda v: self.process_detection())
        self.ref_g_scale_tab3.grid(row=1, column=1, padx=5, pady=5)
        tk.Label(self.detection_frame, text="Reference B:").grid(row=2, column=0, padx=5, pady=5)
        self.ref_b_scale_tab3 = tk.Scale(self.detection_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                         variable=self.ref_b, length=200,
                                         command=lambda v: self.process_detection())
        self.ref_b_scale_tab3.grid(row=2, column=1, padx=5, pady=5)
        # Threshold label (dynamic)
        self.threshold_label = tk.Label(self.detection_frame, text="Threshold W (D²):")
        self.threshold_label.grid(row=3, column=0, padx=5, pady=5)
        self.threshold_scale_tab3 = tk.Scale(self.detection_frame, from_=0, to=200000, orient=tk.HORIZONTAL,
                                             variable=self.threshold_w, length=200,
                                             command=lambda v: self.process_detection())
        self.threshold_scale_tab3.grid(row=3, column=1, padx=5, pady=5)
        self.detection_info = tk.Label(self.detection_frame, text="Click on Original image to pick target color", fg="red")
        self.detection_info.grid(row=4, column=0, columnspan=2, pady=5)
        tk.Button(self.detection_frame, text="Reset", command=self.reset_detection).grid(row=3, column=2, padx=5, pady=5)
        # Image grid
        self.img_labels3 = []
        layout = [["Original", "Mask"], ["R", "G", "B"], ["C", "M", "Y"], ["H", "S", "I"]]
        for row in layout:
            row_frame = tk.Frame(self.frame3)
            row_frame.pack(pady=5)
            for txt in row:
                col_frame = tk.Frame(row_frame)
                col_frame.pack(side=tk.LEFT, padx=10)
                tk.Label(col_frame, text=txt, font=("Arial", 9, "bold")).pack()
                img_lbl = tk.Label(col_frame)
                img_lbl.pack()
                img_lbl.bind("<Button-1>", self.show_large_image)
                self.img_labels3.append(img_lbl)
        self.img_labels3[0].bind("<Button-1>", self.on_image_click_tab3)
        self.update_detection_controls()
    def update_detection_controls(self):
        method = self.detection_combo.get()
        if method == "HSI Segmentation":
            self.threshold_label.config(text="Min S (0-255):")
            self.threshold_scale_tab3.config(from_=0, to=255)
            self.threshold_w.set(50)
        else:
            self.threshold_label.config(text="Threshold W (D²):")
            self.threshold_scale_tab3.config(from_=0, to=200000)
            self.threshold_w.set(10000)
        self.process_detection()
    def reset_detection(self):
        self.ref_r.set(255)
        self.ref_g.set(20)
        self.ref_b.set(20)
        self.update_detection_controls()
    def process_detection(self):
        if self.current_frame is None: return
        frame = self.current_frame.copy()
        method = self.detection_combo.get()
        size = (150, 113)
        if method == "RGB Vector Segmentation":
            mask = self.rgb_vector_segmentation(frame)
        else:
            mask = self.hsi_segmentation(frame)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_photo = ImageTk.PhotoImage(Image.fromarray(mask_rgb).resize(size))
        self.img_labels3[1].config(image=mask_photo)
        self.img_labels3[1].image = mask_photo
        # Display channels
        b, g, r = cv2.split(frame)
        r_col = np.zeros_like(frame); r_col[:,:,2] = r
        g_col = np.zeros_like(frame); g_col[:,:,1] = g
        b_col = np.zeros_like(frame); b_col[:,:,0] = b
        c_col, m_col, y_col = self.bgr_to_cmy(frame)
        h, s, i = self.bgr_to_hsi(frame)
        h_img = np.stack([h]*3, axis=-1); s_img = np.stack([s]*3, axis=-1); i_img = np.stack([i]*3, axis=-1)
        channels = [r_col, g_col, b_col, c_col, m_col, y_col, h_img, s_img, i_img]
        channel_photos = []
        for ch in channels:
            if len(ch.shape) == 3:
                ch_rgb = cv2.cvtColor(ch, cv2.COLOR_BGR2RGB)
            else:
                ch_rgb = np.stack([ch]*3, axis=-1)
            photo = ImageTk.PhotoImage(Image.fromarray(ch_rgb).resize(size))
            channel_photos.append(photo)
        for j, photo in enumerate(channel_photos, 2):
            self.img_labels3[j].config(image=photo)
            self.img_labels3[j].image = photo
    def rgb_vector_segmentation(self, frame):
        a_r, a_g, a_b, T2 = self.ref_r.get(), self.ref_g.get(), self.ref_b.get(), self.threshold_w.get()
        diff_r = frame[:, :, 2].astype(np.int32) - a_r
        diff_g = frame[:, :, 1].astype(np.int32) - a_g
        diff_b = frame[:, :, 0].astype(np.int32) - a_b
        dist_sq = diff_r**2 + diff_g**2 + diff_b**2
        mask = (dist_sq <= T2).astype(np.uint8) * 255
        return mask
    def hsi_segmentation(self, frame):
        h, s, i = self.bgr_to_hsi(frame)
        ref_color = np.uint8([[[self.ref_b.get(), self.ref_g.get(), self.ref_r.get()]]])
        ref_hsv = cv2.cvtColor(ref_color, cv2.COLOR_BGR2HSV)
        ref_h = ref_hsv[0][0][0]
        delta_h = 20
        min_s = self.threshold_w.get()
        ref_h = int(ref_h)
        delta_h = int(delta_h)
        h = h.astype(np.uint8)
        mask_h = cv2.inRange(h, max(0, ref_h - delta_h), min(179, ref_h + delta_h))
        mask_s = cv2.inRange(s, min_s, 255)
        mask = cv2.bitwise_and(mask_h, mask_s)
        return mask
    # ====================== TAB 4: Color Coding ======================
    def setup_color_coding_tab(self):
        top_frame = tk.Frame(self.frame4)
        top_frame.pack(pady=10)
        tk.Label(top_frame, text="Pseudocolor Map:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.colormap_combo = ttk.Combobox(top_frame, values=["Jet", "Hot", "Cool", "Rainbow", "HSV", "Custom"], state="readonly")
        self.colormap_combo.set("Jet")
        self.colormap_combo.pack(side=tk.LEFT, padx=5)
        self.colormap_combo.bind('<<ComboboxSelected>>', lambda e: self.update_color_coding())
        img_frame = tk.Frame(self.frame4)
        img_frame.pack(pady=10)
        self.gray_label = tk.Label(img_frame, text="Gray Input", font=("Arial", 9, "bold"))
        self.gray_label.pack(side=tk.LEFT, padx=20)
        self.gray_label.bind("<Button-1>", self.show_large_image)
        self.colorized_label = tk.Label(img_frame)
        self.colorized_label.pack(side=tk.LEFT, padx=20)
        self.colorized_label.bind("<Button-1>", self.show_large_image)
        self.colorbar_frame = tk.Frame(self.frame4)
        self.colorbar_frame.pack(pady=5)
    def update_color_coding(self):
        if self.current_frame is None: return
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY) if len(self.current_frame.shape) == 3 else self.current_frame
        size = (200, 150)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        gray_photo = ImageTk.PhotoImage(Image.fromarray(gray_rgb).resize(size))
        self.gray_label.config(image=gray_photo); self.gray_label.image = gray_photo
        colorized = self.apply_colormap(gray, self.colormap_combo.get())
        colorized_photo = ImageTk.PhotoImage(Image.fromarray(colorized).resize(size))
        self.colorized_label.config(image=colorized_photo); self.colorized_label.image = colorized_photo
        self.update_colorbar(self.colormap_combo.get())
    # ====================== HỖ TRỢ ======================
    def color_complement(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h = (h.astype(np.int32) + 90) % 180
        return cv2.cvtColor(cv2.merge([h.astype(np.uint8), s, v]), cv2.COLOR_HSV2BGR)
    def color_slicing(self, bgr):
        if len(bgr.shape) != 3: return bgr.copy()
        a_r, a_g, a_b, W = self.ref_r.get(), self.ref_g.get(), self.ref_b.get(), self.threshold_w1.get()
        diff_r = bgr[:, :, 2].astype(np.int32) - a_r
        diff_g = bgr[:, :, 1].astype(np.int32) - a_g
        diff_b = bgr[:, :, 0].astype(np.int32) - a_b
        mask = (diff_r**2 + diff_g**2 + diff_b**2) > W
        result = bgr.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        result[mask] = cv2.merge([gray, gray, gray])[mask]
        return result
    def apply_colormap(self, gray, name):
        if name == "Custom":
            h = gray.astype(np.float32) / 255 * 180
            hsv = np.stack([h, np.full_like(h, 255), np.full_like(h, 255)], axis=-1).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return cv2.applyColorMap(gray, getattr(cv2, f"COLORMAP_{name.upper()}"))
    def update_colorbar(self, name):
        for widget in self.colorbar_frame.winfo_children(): widget.destroy()
        fig, ax = plt.subplots(figsize=(3, 0.5))
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.vstack((gradient, gradient))
        ax.imshow(gradient, aspect='auto', cmap='hsv' if name == "Custom" else name.lower())
        ax.set_axis_off()
        canvas = FigureCanvasTkAgg(fig, master=self.colorbar_frame)
        canvas.draw(); canvas.get_tk_widget().pack()
        plt.close(fig)
    def bgr_to_cmy(self, bgr):
        B, G, R = cv2.split(bgr.astype(np.float32) / 255.0)
        C, M, Y = 1-R, 1-G, 1-B
        C, M, Y = (C*255).astype(np.uint8), (M*255).astype(np.uint8), (Y*255).astype(np.uint8)
        C_col = np.zeros_like(bgr); C_col[:,:,0] = C; C_col[:,:,1] = C
        M_col = np.zeros_like(bgr); M_col[:,:,0] = M; M_col[:,:,2] = M
        Y_col = np.zeros_like(bgr); Y_col[:,:,1] = Y; Y_col[:,:,2] = Y
        return C_col, M_col, Y_col
    def bgr_to_hsi(self, bgr):
        B, G, R = cv2.split(bgr.astype(np.float32) / 255.0)
        I = (R + G + B) / 3
        min_rgb = np.minimum.reduce([R, G, B])
        S = np.where(R+G+B > 0, 1 - 3*min_rgb/(R+G+B+1e-6), 0)
        num = 0.5 * ((R-G) + (R-B))
        den = np.sqrt((R-G)**2 + (R-B)*(G-B)) + 1e-6
        theta = np.arccos(np.clip(num/den, -1, 1))
        H = theta.copy()
        H[B > G] = 2*np.pi - H[B > G]
        H = (H / (2*np.pi) * 179).astype(np.uint8)
        return H, (S*255).astype(np.uint8), (I*255).astype(np.uint8)
    def hsi_to_bgr(self, h, s, i):
        H = h.astype(np.float32) / 179 * 360
        S = s.astype(np.float32) / 255
        I = i.astype(np.float32) / 255
        R, G, B = np.zeros_like(I), np.zeros_like(I), np.zeros_like(I)
        m = (H >= 0) & (H < 120)
        B[m] = I[m] * (1 - S[m])
        R[m] = I[m] * (1 + S[m] * np.cos(np.deg2rad(H[m])) / np.cos(np.deg2rad(60 - H[m])))
        G[m] = 3*I[m] - (R[m] + B[m])
        m = (H >= 120) & (H < 240)
        Ht = H[m] - 120
        R[m] = I[m] * (1 - S[m])
        G[m] = I[m] * (1 + S[m] * np.cos(np.deg2rad(Ht)) / np.cos(np.deg2rad(60 - Ht)))
        B[m] = 3*I[m] - (R[m] + G[m])
        m = (H >= 240) & (H < 360)
        Ht = H[m] - 240
        G[m] = I[m] * (1 - S[m])
        B[m] = I[m] * (1 + S[m] * np.cos(np.deg2rad(Ht)) / np.cos(np.deg2rad(60 - Ht)))
        R[m] = 3*I[m] - (G[m] + B[m])
        return cv2.merge([(B*255).astype(np.uint8), (G*255).astype(np.uint8), (R*255).astype(np.uint8)])
    # ====================== HIỂN THỊ ======================
    def update_converted(self, event=None):
        if self.current_frame is None: return
        frame = self.current_frame.copy()
        color = self.color_combo.get()
        size = (150, 113)
        if color == "Tonal Correction": converted = self.apply_tonal_correction(frame)
        elif color == "Color Slicing": converted = self.color_slicing(frame)
        elif color == "RGB": converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif color == "HSV": converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif color == "LAB": converted = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        elif color == "Gray": gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); converted = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif color == "Color Complement": converted = self.color_complement(frame)
        else: converted = frame.copy()
        converted_rgb = cv2.cvtColor(converted, cv2.COLOR_BGR2RGB) if color != "Gray" else np.stack([converted[:,:,0]]*3, axis=-1)
        photo = ImageTk.PhotoImage(Image.fromarray(converted_rgb).resize(size))
        self.img_labels1[1].config(image=photo); self.img_labels1[1].image = photo
    def display_all(self, frame):
        size = (150, 113)
        original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_photo = ImageTk.PhotoImage(Image.fromarray(original_rgb).resize(size))
        color_space = self.color_combo.get()
        if color_space == "Tonal Correction": converted = self.apply_tonal_correction(frame)
        elif color_space == "Color Slicing": converted = self.color_slicing(frame)
        elif color_space == "RGB": converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif color_space == "HSV": converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif color_space == "LAB": converted = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        elif color_space == "Gray": gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); converted = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif color_space == "Color Complement": converted = self.color_complement(frame)
        else: converted = frame.copy()
        converted_rgb = cv2.cvtColor(converted, cv2.COLOR_BGR2RGB) if color_space != "Gray" else np.stack([converted[:,:,0]]*3, axis=-1)
        converted_photo = ImageTk.PhotoImage(Image.fromarray(converted_rgb).resize(size))
        b, g, r = cv2.split(frame)
        r_col = np.zeros_like(frame); r_col[:,:,2] = r
        g_col = np.zeros_like(frame); g_col[:,:,1] = g
        b_col = np.zeros_like(frame); b_col[:,:,0] = b
        c_col, m_col, y_col = self.bgr_to_cmy(frame)
        h, s, i = self.bgr_to_hsi(frame)
        h_img = np.stack([h]*3, axis=-1); s_img = np.stack([s]*3, axis=-1); i_img = np.stack([i]*3, axis=-1)
        channels = [r_col, g_col, b_col, c_col, m_col, y_col, h_img, s_img, i_img]
        channel_photos = [ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(ch, cv2.COLOR_BGR2RGB) if len(ch.shape)==3 else ch).resize(size)) for ch in channels]
        for img_labels in [self.img_labels1, self.img_labels3]:
            img_labels[0].config(image=original_photo); img_labels[0].image = original_photo
            img_labels[1].config(image=converted_photo); img_labels[1].image = converted_photo
            if color_space == "BGR":
                for j, photo in enumerate(channel_photos, 2):
                    img_labels[j].config(image=photo); img_labels[j].image = photo
            else:
                for j in range(2, len(img_labels)):
                    img_labels[j].config(image=''); img_labels[j].image = None
        self.update_converted()
    # ====================== PHÓNG TO ======================
    def show_large_image(self, event):
        if not hasattr(event.widget, 'image') or event.widget.image is None:
            return
        try:
            pil_img = ImageTk.getimage(event.widget.image)
        except:
            return
        large_window = tk.Toplevel(self.root)
        large_window.title("Large View - Click or ESC to close")
        large_window.configure(bg='black')
        large_window.focus_set()
        large_size = (pil_img.width * 4, pil_img.height * 4)
        large_img = pil_img.resize(large_size, Image.LANCZOS)
        large_photo = ImageTk.PhotoImage(large_img)
        label = tk.Label(large_window, image=large_photo, bg='black', cursor="hand2")
        label.image = large_photo
        label.pack(expand=True)
        label.bind("<Button-1>", lambda e: large_window.destroy())
        large_window.bind("<Escape>", lambda e: large_window.destroy())
        large_window.protocol("WM_DELETE_WINDOW", lambda: large_window.destroy())
    # ====================== WEBCAM & LOAD ======================
    def start_webcam(self):
        self.running = True
    def stop_webcam(self):
        self.running = False
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            img = cv2.imread(path)
            if img is None:
                messagebox.showerror("Error", "Cannot load image!")
                return
            self.cap.release()
            self.current_frame = img.copy()
            self.display_all(img)
            self.process_tab2()
            self.process_detection()
            self.update_color_coding()
    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.display_all(frame)
                self.process_tab2()
                self.process_detection()
                self.update_color_coding()
        self.root.after(30, self.update_frame)
    def on_closing(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()
if __name__ == "__main__":
    root = tk.Tk()
    app = ColorProcessingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()