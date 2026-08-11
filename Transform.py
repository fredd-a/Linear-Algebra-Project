"""
3D Linear Transformation Lab
-----------------------------
A Tkinter desktop app for visualizing how linear transformation matrices
act on 3D shapes. Pick a shape, pick a transformation, drag the sliders
(or type a custom matrix), and hit Animate to watch the shape morph.

Run with:  python shape_transform_gui.py
Requires:  numpy, matplotlib  (both pip-installable; tkinter ships with
           most standard Python installs)
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import colorchooser

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# THEME

BG = "#0f1b2d"  # deep navy (page / figure background)
PANEL_BG = "#13253d"  # slightly lighter navy (control panel)
FIELD_BG = "#0c1523"  # input field background
ACCENT = "#4fd1e8"  # cyan - primary accent
ACCENT_2 = "#f2a65a"  # amber - animate / highlight accent
TEXT = "#34424d"
MUTED = "#7b93ac"
GRID_LINE = "#1e3a5f"
SHAPE_FACE = "#2f6f8f"
SHAPE_EDGE = "#6eaabb"
GHOST_EDGE = "#3c5a78"

FONT_LABEL = ("Consolas", 15)
FONT_LABEL_BOLD = ("Consolas", 15, "bold")
FONT_TITLE = ("Consolas", 20, "bold")
FONT_MATRIX = ("Consolas", 14, "bold")


# SHAPE GENERATORS
# Each returns (vertices: (N,3) float array, faces: list[list[int]])
# Shapes are centered at the origin and sized to roughly fit inside
# a radius-1 sphere so every shape shares consistent axis limits.

def make_cube():
    s = 0.75
    v = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)])
    # reorder into the 8 corners in a convenient winding order
    v = np.array(
        [
            [-s, -s, -s],
            [s, -s, -s],
            [s, s, -s],
            [-s, s, -s],
            [-s, -s, s],
            [s, -s, s],
            [s, s, s],
            [-s, s, s],
        ]
    )
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4],
    ]
    return v, faces


def make_tetrahedron():
    s = 0.8
    v = (
        s
        * np.array(
            [
                [1, 1, 1],
                [1, -1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
            ]
        )
        / np.sqrt(3)
    )
    faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    return v, faces


def make_octahedron():
    s = 0.95
    v = np.array(
        [
            [s, 0, 0],
            [-s, 0, 0],
            [0, s, 0],
            [0, -s, 0],
            [0, 0, s],
            [0, 0, -s],
        ]
    )
    faces = [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [0, 2, 5],
        [2, 1, 5],
        [1, 3, 5],
        [3, 0, 5],
    ]
    return v, faces


def make_sphere(n_lat=10, n_lon=16, r=0.85):
    thetas = np.linspace(0, np.pi, n_lat)
    phis = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    verts = []
    index = {}
    for i, th in enumerate(thetas):
        for j, ph in enumerate(phis):
            index[(i, j)] = len(verts)
            verts.append(
                [
                    r * np.sin(th) * np.cos(ph),
                    r * np.sin(th) * np.sin(ph),
                    r * np.cos(th),
                ]
            )
    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            faces.append(
                [index[(i, j)], index[(i, j2)], index[(i + 1, j2)], index[(i + 1, j)]]
            )
    return np.array(verts), faces


def make_cylinder(n=16, r=0.55, h=1.5):
    bottom = [
        [r * np.cos(a), r * np.sin(a), -h / 2]
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False)
    ]
    top = [
        [r * np.cos(a), r * np.sin(a), h / 2]
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False)
    ]
    verts = np.array(bottom + top)
    faces = []
    for j in range(n):
        j2 = (j + 1) % n
        faces.append([j, j2, n + j2, n + j])
    faces.append(list(range(n)))  # bottom cap
    faces.append(list(range(n, 2 * n)))  # top cap
    return verts, faces


def make_cone(n=16, r=0.7, h=1.5):
    base = [
        [r * np.cos(a), r * np.sin(a), -h / 2]
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False)
    ]
    apex = [0, 0, h / 2]
    verts = np.array(base + [apex])
    apex_idx = n
    faces = []
    for j in range(n):
        j2 = (j + 1) % n
        faces.append([j, j2, apex_idx])
    faces.append(list(range(n)))  # base cap
    return verts, faces


def make_torus(n_major=16, n_minor=10, R=0.6, r=0.22):
    verts = []
    index = {}
    for i, u in enumerate(np.linspace(0, 2 * np.pi, n_major, endpoint=False)):
        for j, v in enumerate(np.linspace(0, 2 * np.pi, n_minor, endpoint=False)):
            index[(i, j)] = len(verts)
            verts.append(
                [
                    (R + r * np.cos(v)) * np.cos(u),
                    (R + r * np.cos(v)) * np.sin(u),
                    r * np.sin(v),
                ]
            )
    faces = []
    for i in range(n_major):
        i2 = (i + 1) % n_major
        for j in range(n_minor):
            j2 = (j + 1) % n_minor
            faces.append(
                [index[(i, j)], index[(i2, j)], index[(i2, j2)], index[(i, j2)]]
            )
    return np.array(verts), faces


SHAPES = {
    "Cube": make_cube,
    "Tetrahedron": make_tetrahedron,
    "Octahedron": make_octahedron,
    "Sphere": make_sphere,
    "Cylinder": make_cylinder,
    "Cone": make_cone,
    "Torus": make_torus,
}

# TRANSFORMATION MATRIX BUILDERS

def rot_x(deg):
    t = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def rot_y(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def rot_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


TRANSFORMS = [
    "Scaling",
    "Rotation X",
    "Rotation Y",
    "Rotation Z",
    "Shear",
    "Reflection",
    "Custom Matrix",
]

# MAIN APPLICATION
class TransformLab:
    def __init__(self, root):
        self.root = root
        root.title("3D Linear Transformation Lab")
        root.configure(bg=BG)
        root.geometry("1430x720")

        self._setup_style()

        self.shape_name = tk.StringVar(value="Cube")
        self.transform_name = tk.StringVar(value="Scaling")
        self.param_vars = {}  # name -> tk.DoubleVar / tk.BooleanVar for current transform
        self.custom_entries = []  # 3x3 grid of Entry widgets for Custom Matrix mode
        self.animating = False

        self.shape_color = SHAPE_FACE  # current fill/wireframe color, hex string
        self.hollow = tk.BooleanVar(value=False)

        self._build_layout()
        self._load_shape()
        self._apply_appearance()
        self._rebuild_param_panel()
        self._apply_transform(np.eye(3))

    # Styling

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=PANEL_BG)
        style.configure("TLabel", background=PANEL_BG, foreground=TEXT, font=FONT_LABEL)
        style.configure(
            "Title.TLabel", background=PANEL_BG, foreground=ACCENT, font=FONT_TITLE
        )
        style.configure(
            "Section.TLabel",
            background=PANEL_BG,
            foreground=MUTED,
            font=FONT_LABEL_BOLD,
        )
        style.configure(
            "TButton",
            background=ACCENT,
            foreground=BG,
            font=FONT_LABEL_BOLD,
            borderwidth=0,
            focusthickness=0,
            padding=6,
        )
        style.map("TButton", background=[("active", "#7fe3f2")])
        style.configure(
            "Animate.TButton",
            background=ACCENT_2,
            foreground=BG,
            font=FONT_LABEL_BOLD,
            padding=6,
        )
        style.map("Animate.TButton", background=[("active", "#f7c088")])
        style.configure(
            "TCombobox",
            fieldbackground=FIELD_BG,
            background=FIELD_BG,
            foreground=TEXT,
            arrowcolor=ACCENT,
        )
        style.configure("Horizontal.TScale", background=PANEL_BG, troughcolor=FIELD_BG)
        style.configure(
            "TCheckbutton", background=PANEL_BG, foreground=TEXT, font=FONT_LABEL
        )
        style.map("TCheckbutton", background=[("active", PANEL_BG)])


    # Layout

    def _build_layout(self):
        # --- left control panel ---
        panel = ttk.Frame(self.root, padding=16, width=320)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        ttk.Label(panel, text="TRANSFORM LAB", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            panel,
            text="3D linear transformations, visualized",
            style="TLabel",
            foreground=MUTED,
        ).pack(anchor="w", pady=(0, 18))

        ttk.Label(panel, text="SHAPE", style="Section.TLabel").pack(anchor="w")
        shape_box = ttk.Combobox(
            panel,
            textvariable=self.shape_name,
            values=list(SHAPES.keys()),
            state="readonly",
            width=20,
            font=("Consolas", 16),
        )
        shape_box.pack(fill=tk.X, pady=(4, 16))
        shape_box.bind("<<ComboboxSelected>>", lambda e: self._on_shape_change())

        ttk.Label(panel, text="TRANSFORMATION", style="Section.TLabel").pack(anchor="w")
        transform_box = ttk.Combobox(
            panel,
            textvariable=self.transform_name,
            values=TRANSFORMS,
            state="readonly",
            font=("Consolas", 16),
        )
        transform_box.pack(fill=tk.X, pady=(4, 12))
        transform_box.bind(
            "<<ComboboxSelected>>", lambda e: self._rebuild_param_panel()
        )

        # container that gets rebuilt per-transform
        self.param_frame = ttk.Frame(panel)
        self.param_frame.pack(fill=tk.X, pady=(0, 16))

        btn_row = ttk.Frame(panel)
        btn_row.pack(fill=tk.X, pady=(4, 20))
        ttk.Button(btn_row, text="Reset", command=self._reset).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6)
        )
        ttk.Button(
            btn_row, text="Animate", style="Animate.TButton", command=self._animate
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # appearance controls: fill/wireframe color + hollow toggle
        ttk.Label(panel, text="APPEARANCE", style="Section.TLabel").pack(anchor="w")
        appearance_row = ttk.Frame(panel)
        appearance_row.pack(fill=tk.X, pady=(4, 4))

        self.color_swatch = tk.Button(
            appearance_row,
            bg=self.shape_color,
            width=3,
            relief=tk.FLAT,
            highlightbackground=GRID_LINE,
            highlightthickness=1,
            command=self._choose_color,
        )
        self.color_swatch.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(appearance_row, text="Color", style="TLabel").pack(side=tk.LEFT)

        ttk.Checkbutton(
            panel,
            text="Hollow (wireframe only)",
            variable=self.hollow,
            command=self._apply_appearance,
        ).pack(anchor="w", pady=(6, 16))

        # matrix readout ("spec sheet" card)
        ttk.Label(panel, text="MATRIX", style="Section.TLabel").pack(anchor="w")
        matrix_card = tk.Frame(
            panel, bg=FIELD_BG, highlightbackground=GRID_LINE, highlightthickness=1
        )
        matrix_card.pack(fill=tk.X, pady=(4, 10))
        self.matrix_labels = []
        for r in range(3):
            row_labels = []
            for c in range(3):
                lbl = tk.Label(
                    matrix_card,
                    text="0.00",
                    width=6,
                    bg=FIELD_BG,
                    fg=ACCENT,
                    font=FONT_MATRIX,
                )
                lbl.grid(row=r, column=c, padx=4, pady=4)
                row_labels.append(lbl)
            self.matrix_labels.append(row_labels)

        self.det_label = ttk.Label(
            panel, text="det(M) = 1.00", style="TLabel", foreground=MUTED
        )
        self.det_label.pack(anchor="w")

        # --- right plotting area ---
        plot_frame = tk.Frame(self.root, bg=BG)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(7, 7), facecolor=BG)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._style_axes()
        self._init_plot_objects()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _style_axes(self):
        self.ax.set_facecolor(BG)
        lim = 2.0
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((0.06, 0.11, 0.18, 1.0))
            axis.line.set_color(GRID_LINE)
            axis._axinfo["grid"]["color"] = GRID_LINE
        self.ax.tick_params(colors=MUTED)
        self.ax.set_xlabel("X", color=MUTED)
        self.ax.set_ylabel("Y", color=MUTED)
        self.ax.set_zlabel("Z", color=MUTED)

    # Shape loading

    def _init_plot_objects(self):
        # Created once and reused for the app's lifetime. Switching shapes
        # updates their vertex data in place (via set_verts) instead of
        # clearing the axes -- clearing mid-redraw is what caused a
        # matplotlib 3D-projection crash on rapid shape switches.
        #
        # Seeded with real geometry (not an empty list) because
        # FigureCanvasTkAgg triggers an initial draw as soon as it's
        # constructed -- an empty Poly3DCollection at that point crashes
        # matplotlib's 3D projection code.
        verts, faces = SHAPES[self.shape_name.get()]()
        self.base_verts = verts
        self.faces = faces
        init_faces = [[verts[i] for i in f] for f in faces]

        # facecolor=(0,0,0,0) (fully transparent RGBA) rather than the
        # string "none" -- "none" triggers a matplotlib 3D-projection bug
        # on some versions when the collection is actually rendered.
        self.ghost = Poly3DCollection(
            init_faces,
            facecolor=(0, 0, 0, 0),
            edgecolor=GHOST_EDGE,
            linewidths=0.6,
            alpha=0.35,
        )
        self.ax.add_collection3d(self.ghost)

        self.mesh = Poly3DCollection(
            init_faces,
            facecolor=SHAPE_FACE,
            edgecolor=SHAPE_EDGE,
            linewidths=0.8,
            alpha=0.85,
        )
        self.ax.add_collection3d(self.mesh)

    def _load_shape(self):
        verts, faces = SHAPES[self.shape_name.get()]()
        self.base_verts = verts
        self.faces = faces

        ghost_faces = [[verts[i] for i in f] for f in faces]
        self.ghost.set_verts(ghost_faces)
        self.mesh.set_verts(ghost_faces)
        self.ax.set_title(self.shape_name.get(), color=TEXT, fontfamily="monospace")

    def _on_shape_change(self):
        self._load_shape()
        self._apply_transform(self._current_matrix())

    # ---------------------------------------------------
    # Appearance: fill color + hollow (wireframe-only) toggle
    # ---------------------------------------------------
    def _choose_color(self):
        rgb, hex_color = colorchooser.askcolor(
            initialcolor=self.shape_color, title="Choose shape color"
        )
        if hex_color is None:  # user cancelled the dialog
            return
        self.shape_color = hex_color
        self.color_swatch.configure(bg=hex_color)
        self._apply_appearance()

    def _apply_appearance(self):
        if self.hollow.get():
            # fully transparent fill so only the edges (the wireframe) show,
            # drawn in the chosen color
            self.mesh.set_alpha(None)
            self.mesh.set_facecolor((0, 0, 0, 0))
            self.mesh.set_edgecolor(self.shape_color)
            self.mesh.set_linewidth(1.1)
        else:
            self.mesh.set_alpha(0.85)
            self.mesh.set_facecolor(self.shape_color)
            self.mesh.set_edgecolor(SHAPE_EDGE)
            self.mesh.set_linewidth(0.8)
        self.canvas.draw()

    # Parameter panel (rebuilt per transformation type)

    def _rebuild_param_panel(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        self.param_vars.clear()
        self.custom_entries = []

        mode = self.transform_name.get()

        def add_slider(label, key, frm, to, default):
            ttk.Label(self.param_frame, text=label, style="TLabel").pack(anchor="w")
            var = tk.DoubleVar(value=default)
            scale = ttk.Scale(
                self.param_frame,
                from_=frm,
                to=to,
                variable=var,
                command=lambda v: self._on_param_change(),
            )
            scale.pack(fill=tk.X, pady=(0, 10))
            self.param_vars[key] = var

        if mode == "Scaling":
            add_slider("Scale X", "sx", 0.1, 2.5, 1.0)
            add_slider("Scale Y", "sy", 0.1, 2.5, 1.0)
            add_slider("Scale Z", "sz", 0.1, 2.5, 1.0)

        elif mode == "Rotation X":
            add_slider("Angle (deg)", "angle", 0, 360, 0)

        elif mode == "Rotation Y":
            add_slider("Angle (deg)", "angle", 0, 360, 0)

        elif mode == "Rotation Z":
            add_slider("Angle (deg)", "angle", 0, 360, 0)

        elif mode == "Shear":
            add_slider("Shear XY (x += k*y)", "shxy", -1.5, 1.5, 0.0)
            add_slider("Shear XZ (x += k*z)", "shxz", -1.5, 1.5, 0.0)
            add_slider("Shear YZ (y += k*z)", "shyz", -1.5, 1.5, 0.0)

        elif mode == "Reflection":
            for label, key in (
                ("Reflect across YZ (flip X)", "rx"),
                ("Reflect across XZ (flip Y)", "ry"),
                ("Reflect across XY (flip Z)", "rz"),
            ):
                var = tk.BooleanVar(value=False)
                cb = ttk.Checkbutton(
                    self.param_frame,
                    text=label,
                    variable=var,
                    command=self._on_param_change,
                )
                cb.pack(anchor="w", pady=(0, 6))
                self.param_vars[key] = var

        elif mode == "Custom Matrix":
            grid = ttk.Frame(self.param_frame)
            grid.pack(pady=(4, 10))
            defaults = np.eye(3)
            for r in range(3):
                row_entries = []
                for c in range(3):
                    e = tk.Entry(
                        grid,
                        width=6,
                        bg=FIELD_BG,
                        fg=TEXT,
                        insertbackground=TEXT,
                        relief=tk.FLAT,
                        justify="center",
                        font=FONT_MATRIX,
                    )
                    e.insert(0, f"{defaults[r, c]:.2f}")
                    e.grid(row=r, column=c, padx=3, pady=3)
                    row_entries.append(e)
                self.custom_entries.append(row_entries)
            ttk.Button(
                self.param_frame, text="Apply Matrix", command=self._on_param_change
            ).pack(fill=tk.X, pady=(6, 0))

        self._on_param_change()


    # Matrix computation

    def _current_matrix(self):
        mode = self.transform_name.get()
        v = self.param_vars

        if mode == "Scaling":
            return np.diag([v["sx"].get(), v["sy"].get(), v["sz"].get()])

        if mode == "Rotation X":
            return rot_x(v["angle"].get())
        if mode == "Rotation Y":
            return rot_y(v["angle"].get())
        if mode == "Rotation Z":
            return rot_z(v["angle"].get())

        if mode == "Shear":
            M = np.eye(3)
            M[0, 1] = v["shxy"].get()
            M[0, 2] = v["shxz"].get()
            M[1, 2] = v["shyz"].get()
            return M

        if mode == "Reflection":
            return np.diag(
                [
                    -1.0 if v["rx"].get() else 1.0,
                    -1.0 if v["ry"].get() else 1.0,
                    -1.0 if v["rz"].get() else 1.0,
                ]
            )

        if mode == "Custom Matrix":
            M = np.eye(3)
            for r in range(3):
                for c in range(3):
                    try:
                        M[r, c] = float(self.custom_entries[r][c].get())
                    except ValueError:
                        M[r, c] = 0.0
            return M

        return np.eye(3)


    # Applying / animating the transform

    def _on_param_change(self):
        if self.animating:
            return
        M = self._current_matrix()
        self._update_matrix_readout(M)
        self._apply_transform(M)

    def _apply_transform(self, M):
        transformed = (M @ self.base_verts.T).T
        faces = [[transformed[i] for i in f] for f in self.faces]
        self.mesh.set_verts(faces)
        self.canvas.draw()

    def _update_matrix_readout(self, M):
        for r in range(3):
            for c in range(3):
                self.matrix_labels[r][c].config(text=f"{M[r, c]: .2f}")
        det = np.linalg.det(M)
        self.det_label.config(text=f"det(M) = {det: .3f}")

    def _reset(self):
        self._rebuild_param_panel()  # rebuilding restores default slider values
        self._apply_transform(np.eye(3))
        self._update_matrix_readout(np.eye(3))

    def _animate(self, n_frames=40, interval_ms=20):
        if self.animating:
            return
        M_target = self._current_matrix()
        self._update_matrix_readout(M_target)
        I = np.eye(3)
        self.animating = True

        def ease(t):
            return t * t * (3 - 2 * t)  # smoothstep

        def step(frame):
            if frame >= n_frames:
                self._apply_transform(M_target)
                self.animating = False
                return
            t = ease(frame / (n_frames - 1))
            M_t = (1 - t) * I + t * M_target
            self._apply_transform(M_t)
            self.root.after(interval_ms, step, frame + 1)

        step(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = TransformLab(root)
    root.mainloop()
