import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull

# --- CẤU HÌNH MÀU SẮC & DATA MẶC ĐỊNH ---
COLORS = {'red': '#E74C3C', 'blue': '#3498DB', 'green': '#2ECC71', 'purple': '#9B59B6'}

# Dữ liệu mẫu ban đầu (để không phải nhập lại mỗi lần chạy test)
INIT_DATA = [
    {'GPA': 3.8, 'Activity': 90}, {'GPA': 3.6, 'Activity': 85},
    {'GPA': 3.9, 'Activity': 95}, {'GPA': 3.7, 'Activity': 50},
    {'GPA': 3.8, 'Activity': 45}, {'GPA': 2.5, 'Activity': 90},
    {'GPA': 2.2, 'Activity': 80}, {'GPA': 1.8, 'Activity': 40},
    {'GPA': 2.0, 'Activity': 50}, {'GPA': 3.2, 'Activity': 70}
]

class ClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PHẦN MỀM PHÂN CỤM SINH VIÊN (GUI)")
        self.root.geometry("1300x750")
        
        # Dữ liệu chính
        self.data = INIT_DATA.copy()
        self.df = pd.DataFrame()
        self.model_ready = False

        # --- TẠO GIAO DIỆN ---
        self.create_layout()
        self.update_table()
        
        # Vẽ biểu đồ mặc định
        self.perform_clustering()
        self.view_scatter()

    def create_layout(self):
        # 1. KHUNG TRÁI (NHẬP LIỆU & ĐIỀU KHIỂN)
        left_frame = tk.Frame(self.root, width=350, bg="#f0f0f0", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False) # Giữ cố định chiều rộng

        # Tiêu đề
        tk.Label(left_frame, text="NHẬP DỮ LIỆU", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=(0, 10))

        # Form nhập liệu
        input_frame = tk.LabelFrame(left_frame, text="Thông tin sinh viên", bg="#f0f0f0", padx=5, pady=5)
        input_frame.pack(fill=tk.X)

        tk.Label(input_frame, text="GPA (0 - 4.0):", bg="#f0f0f0").grid(row=0, column=0, sticky="w")
        self.entry_gpa = ttk.Entry(input_frame, width=15)
        self.entry_gpa.grid(row=0, column=1, padx=5, pady=5)
        self.entry_gpa.insert(0, "3.5") # Giá trị gợi ý

        tk.Label(input_frame, text="ĐRL (0 - 100):", bg="#f0f0f0").grid(row=1, column=0, sticky="w")
        self.entry_act = ttk.Entry(input_frame, width=15)
        self.entry_act.grid(row=1, column=1, padx=5, pady=5)
        self.entry_act.insert(0, "80")

        # Nút Thêm/Xóa
        btn_frame = tk.Frame(input_frame, bg="#f0f0f0")
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Thêm SV", command=self.add_student).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Xóa Hết", command=self.clear_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset Mẫu", command=self.reset_data).pack(side=tk.LEFT, padx=5)

        # Bảng hiển thị dữ liệu (Treeview)
        tk.Label(left_frame, text="Danh sách sinh viên:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=(15, 5))
        
        cols = ('ID', 'GPA', 'Activity')
        self.tree = ttk.Treeview(left_frame, columns=cols, show='headings', height=12)
        self.tree.heading('ID', text='ID')
        self.tree.heading('GPA', text='GPA')
        self.tree.heading('Activity', text='ĐRL')
        self.tree.column('ID', width=40, anchor='center')
        self.tree.column('GPA', width=60, anchor='center')
        self.tree.column('Activity', width=60, anchor='center')
        self.tree.pack(fill=tk.X)

        # Nút điều khiển vẽ biểu đồ
        ctrl_frame = tk.LabelFrame(left_frame, text="Chọn Biểu Đồ", bg="#f0f0f0", padx=5, pady=10)
        ctrl_frame.pack(fill=tk.X, pady=20)

        ttk.Button(ctrl_frame, text="1. Xem Dendrogram", command=self.view_dendrogram).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl_frame, text="2. Xem Phân Nhóm", command=self.view_scatter).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl_frame, text="3. Xem Boxplot", command=self.view_boxplot).pack(fill=tk.X, pady=2)
        
        ttk.Button(left_frame, text="PHÂN TÍCH LẠI", command=self.perform_clustering).pack(fill=tk.X, pady=10)

        # 2. KHUNG PHẢI (HIỂN THỊ BIỂU ĐỒ)
        right_frame = tk.Frame(self.root, bg="white")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- XỬ LÝ DỮ LIỆU ---
    def add_student(self):
        try:
            gpa = float(self.entry_gpa.get())
            act = int(self.entry_act.get())
            
            if not (0 <= gpa <= 4.0):
                messagebox.showerror("Lỗi", "GPA phải từ 0 đến 4.0")
                return
            if not (0 <= act <= 100):
                messagebox.showerror("Lỗi", "ĐRL phải từ 0 đến 100")
                return
                
            self.data.append({'GPA': gpa, 'Activity': act})
            self.update_table()
            self.perform_clustering() # Tự động chạy lại khi thêm
            self.view_scatter()       # Vẽ lại
            
        except ValueError:
            messagebox.showerror("Lỗi", "Vui lòng nhập đúng định dạng số!")

    def clear_data(self):
        self.data = []
        self.update_table()
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Chưa có dữ liệu", ha='center')
        self.canvas.draw()

    def reset_data(self):
        self.data = INIT_DATA.copy()
        self.update_table()
        self.perform_clustering()
        self.view_scatter()

    def update_table(self):
        # Xóa cũ
        for row in self.tree.get_children():
            self.tree.delete(row)
        # Thêm mới
        for i, item in enumerate(self.data):
            self.tree.insert("", "end", values=(i+1, item['GPA'], item['Activity']))

    # --- THUẬT TOÁN PHÂN CỤM ---
    def perform_clustering(self):
        if len(self.data) < 4:
            self.model_ready = False
            return

        self.df = pd.DataFrame(self.data)
        self.df['Student_ID'] = range(1, len(self.df) + 1)
        
        X = self.df[['GPA', 'Activity']].values
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)
        self.X_raw = X

        # Linkage Matrix
        self.linkage_matrix = sch.linkage(self.X_scaled, method='ward')

        # Clustering
        n_clusters = 4 # Cố định 4 nhóm
        try:
            hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
            self.y_hc = hc.fit_predict(self.X_scaled)
        except:
            hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
            self.y_hc = hc.fit_predict(self.X_scaled)
            
        self.df['Cluster'] = self.y_hc
        
        # Gán nhãn màu sắc
        self.cluster_info = {}
        means = self.df.groupby('Cluster')[['GPA', 'Activity']].mean()
        
        for cluster_id, row in means.iterrows():
            gpa = row['GPA']
            act = row['Activity']
            if gpa > 3.1 and act > 70: label, color = "Ưu tú", COLORS['red']
            elif gpa > 3.1 and act <= 70: label, color = "Mọt sách", COLORS['blue']
            elif gpa <= 3.1 and act > 70: label, color = "Năng động", COLORS['green']
            else: label, color = "Cần hỗ trợ", COLORS['purple']
            self.cluster_info[cluster_id] = {'label': label, 'color': color}
            
        self.model_ready = True

    # --- CÁC HÀM VẼ BIỂU ĐỒ ---
    def check_ready(self):
        if not self.model_ready:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Cần ít nhất 4 sinh viên để phân tích!", ha='center', fontsize=14, color='red')
            self.canvas.draw()
            return False
        return True

    def view_dendrogram(self):
        if not self.check_ready(): return
        self.ax.clear()
        
        max_d = max(self.linkage_matrix[:, 2])
        threshold = max_d * 0.55
        
        sch.dendrogram(self.linkage_matrix, ax=self.ax, color_threshold=threshold, above_threshold_color='#95A5A6')
        self.ax.axhline(y=threshold, color='black', linestyle='--', label='Ngưỡng cắt')
        
        self.ax.set_title("Biểu đồ cây (Dendrogram)", fontsize=14, fontweight='bold', color='navy')
        self.ax.set_ylabel("Khoảng cách Euclidean")
        self.ax.legend()
        self.canvas.draw()

    def view_scatter(self):
        if not self.check_ready(): return
        self.ax.clear()
        
        for i in range(4):
            if i not in self.cluster_info: continue
            info = self.cluster_info[i]
            # Vẽ điểm
            points = self.X_raw[self.y_hc == i]
            self.ax.scatter(points[:, 0], points[:, 1], s=100, c=info['color'], 
                            label=info['label'], edgecolors='white', zorder=2)
            
            # Vẽ vùng bao (Convex Hull)
            if len(points) > 2:
                hull = ConvexHull(points)
                pts = points
                for simplex in hull.simplices:
                    self.ax.plot(pts[simplex, 0], pts[simplex, 1], color=info['color'], alpha=0.3, lw=2, linestyle='--')
                self.ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color=info['color'], alpha=0.1)

        # Gán ID sinh viên (Xử lý chống đè chữ bằng cách offset nhẹ)
        for idx, row in self.df.iterrows():
            self.ax.annotate(str(row['Student_ID']), 
                             (row['GPA'], row['Activity']),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=9, fontweight='bold')

        # Đường tham chiếu
        self.ax.axhline(y=70, color='gray', linestyle=':', alpha=0.5)
        self.ax.axvline(x=3.1, color='gray', linestyle=':', alpha=0.5)
        
        self.ax.set_title("Kết quả Phân nhóm", fontsize=14, fontweight='bold', color='navy')
        self.ax.set_xlabel("GPA")
        self.ax.set_ylabel("Activity Score")
        self.ax.legend(loc='lower right')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.figure.tight_layout()
        self.canvas.draw()

    def view_boxplot(self):
        if not self.check_ready(): return
        self.ax.clear()
        
        sorted_ids = sorted(self.cluster_info.keys(), key=lambda k: self.cluster_info[k]['label'])
        
        data_plot = []
        colors_plot = []
        labels_plot = []
        
        for i in sorted_ids:
            subset = self.df[self.df['Cluster'] == i]['GPA'].values
            if len(subset) > 0:
                data_plot.append(subset)
                colors_plot.append(self.cluster_info[i]['color'])
                labels_plot.append(self.cluster_info[i]['label'])
        
        if not data_plot: return

        bplot = self.ax.boxplot(data_plot, patch_artist=True, labels=labels_plot, widths=0.5)
        
        for patch, color in zip(bplot['boxes'], colors_plot):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            
        # Jitter plot
        for i, d in enumerate(data_plot):
            y = d
            x = np.random.normal(i + 1, 0.04, size=len(y))
            self.ax.scatter(x, y, alpha=0.8, color=colors_plot[i], s=25, edgecolors='white')

        self.ax.set_title("Phân tích GPA chi tiết", fontsize=14, fontweight='bold', color='navy')
        self.ax.set_ylabel("GPA")
        self.ax.grid(True, linestyle='--', axis='y')
        self.figure.tight_layout()
        self.canvas.draw()

# --- CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    root = tk.Tk()
    try:
        # Cố gắng thiết lập độ phân giải cao cho màn hình HD
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    app = ClusteringApp(root)
    root.mainloop()