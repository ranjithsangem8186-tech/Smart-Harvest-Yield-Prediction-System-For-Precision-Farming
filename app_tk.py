import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np
import os

# Import Matplotlib for Tkinter integration
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Define Color Palette (Precision Farming theme)
COLOR_BG = "#F4F7F6"           # Light sage off-white
COLOR_CARD = "#FFFFFF"         # White card background
COLOR_PRIMARY = "#1B4332"      # Deep Forest Green (Header)
COLOR_SECONDARY = "#4A7C59"    # Sage Green (Accents / Highlights)
COLOR_GOLD = "#D4A373"         # Warm earth gold
COLOR_TEXT = "#2D312E"         # Charcoal text
COLOR_LIGHT_TEXT = "#FFFFFF"   # White text
COLOR_LIGHT_GREEN = "#E8F5E9"  # High yield green highlight
COLOR_LIGHT_ORANGE = "#FFF3E0" # Moderate yield orange highlight
COLOR_LIGHT_RED = "#FFEBEE"    # Low yield red highlight
COLOR_BORDER = "#E0E0E0"       # Light gray borders

# Crop Profiles (Optimal ranges for N-P-K, pH, temperature, rainfall, humidity)
CROP_PROFILES = {
    "Rice": {
        "opt_npk": (120, 60, 60),
        "opt_ph": (6.0, 7.5),
        "opt_temp": (25, 33),
        "opt_rain": (1000, 2000),
        "opt_humid": (70, 90),
        "avg_yield": 3.8,
        "tips": "Rice requires continuous water logging (2-5 cm depth) during growth. Maintain high nitrogen. Recommended sowing: June-July."
    },
    "Wheat": {
        "opt_npk": (120, 50, 40),
        "opt_ph": (6.2, 7.8),
        "opt_temp": (15, 23),
        "opt_rain": (80, 250),
        "opt_humid": (45, 65),
        "avg_yield": 4.0,
        "tips": "Wheat is a Rabi crop requiring cool weather. Control moisture during flowering. Apply urea in split doses. Sowing: November."
    },
    "Maize": {
        "opt_npk": (120, 60, 40),
        "opt_ph": (5.8, 7.5),
        "opt_temp": (20, 30),
        "opt_rain": (500, 1000),
        "opt_humid": (55, 80),
        "avg_yield": 3.2,
        "tips": "Maize requires good drainage. High nitrogen is critical. Avoid water logging. Crop is highly responsive to organic manure."
    },
    "Cotton": {
        "opt_npk": (100, 50, 50),
        "opt_ph": (6.0, 8.0),
        "opt_temp": (25, 35),
        "opt_rain": (600, 1200),
        "opt_humid": (50, 75),
        "avg_yield": 2.2,
        "tips": "Cotton grows best in deep black soils. Requires high heat. Keep soil moist but not wet. Apply balanced P & K for boll opening."
    },
    "Sugarcane": {
        "opt_npk": (150, 80, 80),
        "opt_ph": (6.5, 7.8),
        "opt_temp": (26, 38),
        "opt_rain": (1200, 2200),
        "opt_humid": (65, 85),
        "avg_yield": 78.0,
        "tips": "Sugarcane is a long-duration crop. Requires high irrigation and nitrogen. Ensure weed control in early stages (first 90 days)."
    },
    "Groundnut": {
        "opt_npk": (20, 40, 40),
        "opt_ph": (6.0, 7.5),
        "opt_temp": (22, 32),
        "opt_rain": (400, 800),
        "opt_humid": (50, 70),
        "avg_yield": 2.1,
        "tips": "Groundnut fixes its own nitrogen. Apply gypsum at pegging stage. Sandy/loamy soils are optimal for easy pod penetration."
    },
    "Tomato": {
        "opt_npk": (100, 60, 120),
        "opt_ph": (6.0, 7.0),
        "opt_temp": (20, 30),
        "opt_rain": (400, 700),
        "opt_humid": (50, 75),
        "avg_yield": 24.0,
        "tips": "Tomatoes have a high Potassium demand. Stake plants to prevent disease. Watch for blossom end rot (apply calcium if needed)."
    },
    "Potato": {
        "opt_npk": (120, 60, 120),
        "opt_ph": (5.2, 6.5),
        "opt_temp": (15, 22),
        "opt_rain": (150, 400),
        "opt_humid": (45, 65),
        "avg_yield": 20.0,
        "tips": "Potatoes prefer loose sandy loam and slightly acidic soils. Earthing up is essential at 30 & 45 days. Keep soil uniformly moist."
    }
}

# State-District Mapping (Matches dataset generator)
STATE_DISTRICTS = {
    "Andhra Pradesh": ["Guntur", "Krishna", "Kurnool", "Anantapur", "East Godavari"],
    "Telangana": ["Karimnagar", "Warangal", "Nalgonda", "Nizamabad", "Medak"],
    "Punjab": ["Ludhiana", "Amritsar", "Patiala", "Jalandhar", "Bathinda"],
    "Haryana": ["Karnal", "Hisar", "Sirsa", "Rohtak", "Sonipat"],
    "Uttar Pradesh": ["Meerut", "Bareilly", "Varanasi", "Aligarh", "Gorakhpur"],
    "Maharashtra": ["Nashik", "Pune", "Nagpur", "Ahmednagar", "Jalgaon"],
    "Karnataka": ["Belagavi", "Dharwad", "Tumakuru", "Mysuru", "Raichur"],
    "Tamil Nadu": ["Coimbatore", "Thanjavur", "Erode", "Salem", "Madurai"]
}

class PrecisionFarmingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🌾 Smart Harvest Yield Prediction System")
        self.geometry("1200x800")
        self.configure(bg=COLOR_BG)
        
        # Try loading models
        self.load_ml_models()
        
        # Build UI Elements
        self.create_header()
        self.create_tabs()
        
        # Populate Default Values
        self.reset_inputs()
        
        # Force initial prediction to show graphs
        if self.models_loaded:
            self.predict_yield()

    def load_ml_models(self):
        self.models_loaded = False
        try:
            self.model = joblib.load("yield_prediction_model.joblib")
            self.label_encoders = joblib.load("label_encoders.joblib")
            self.scaler = joblib.load("scaler.joblib")
            self.feature_columns = joblib.load("feature_columns.joblib")
            self.models_loaded = True
            print("Successfully loaded all ML joblib models.")
        except Exception as e:
            messagebox.showerror(
                "Model Loading Error", 
                f"Could not load machine learning model files.\n\nError: {e}\n\nMake sure to run the notebook.ipynb (or build_notebook.py) first to generate the joblib files."
            )

    def create_header(self):
        # Header banner frame
        header_frame = tk.Frame(self, bg=COLOR_PRIMARY, height=90)
        header_frame.pack(fill="x", side="top")
        header_frame.pack_propagate(False)
        
        # Header text
        title_label = tk.Label(
            header_frame, 
            text="🚜 PRECISION FARMING CONTROL CENTER", 
            font=("Segoe UI", 20, "bold"), 
            bg=COLOR_PRIMARY, 
            fg=COLOR_LIGHT_TEXT
        )
        title_label.pack(anchor="w", padx=20, pady=(15, 2))
        
        subtitle_label = tk.Label(
            header_frame, 
            text="🌾 Smart Harvest Yield Prediction & Soil Analysis Dashboard • Academic Final Year Project", 
            font=("Segoe UI", 10, "italic"), 
            bg=COLOR_PRIMARY, 
            fg=COLOR_GOLD
        )
        subtitle_label.pack(anchor="w", padx=20)

    def create_tabs(self):
        # Style Notebook Tabs
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab", 
            background="#D8E2DC", 
            foreground=COLOR_TEXT, 
            font=("Segoe UI", 11, "bold"), 
            padding=[20, 8]
        )
        style.map(
            "TNotebook.Tab", 
            background=[("selected", COLOR_SECONDARY)], 
            foreground=[("selected", COLOR_LIGHT_TEXT)]
        )
        
        # Notebook component
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Create Frames for Tabs
        self.tab_yield = tk.Frame(self.notebook, bg=COLOR_BG)
        self.tab_soil = tk.Frame(self.notebook, bg=COLOR_BG)
        self.tab_insights = tk.Frame(self.notebook, bg=COLOR_BG)
        self.tab_about = tk.Frame(self.notebook, bg=COLOR_BG)
        
        # Add to Notebook
        self.notebook.add(self.tab_yield, text="🌾 Yield Prediction")
        self.notebook.add(self.tab_soil, text="🌱 Soil Analysis")
        self.notebook.add(self.tab_insights, text="📊 Farming Insights")
        self.notebook.add(self.tab_about, text="🧠 About Project")
        
        # Set up content for each tab
        self.setup_tab_yield()
        self.setup_tab_soil()
        self.setup_tab_insights()
        self.setup_tab_about()

    # ==================== TAB 1: YIELD PREDICTION ====================
    def setup_tab_yield(self):
        # Left Panel (Inputs)
        input_panel = tk.LabelFrame(
            self.tab_yield, 
            text="🚜 Agronomic Parameter Inputs", 
            font=("Segoe UI", 11, "bold"), 
            bg=COLOR_CARD, 
            fg=COLOR_PRIMARY, 
            padx=15, 
            pady=15,
            bd=1,
            relief="solid",
            width=420
        )
        input_panel.pack(side="left", fill="both", expand=False, padx=(10, 5), pady=10)
        input_panel.pack_propagate(False)
        
        # Set Grid weights for spacing
        for r in range(13):
            input_panel.grid_rowconfigure(r, weight=1)
        input_panel.grid_columnconfigure(0, weight=1)
        input_panel.grid_columnconfigure(1, weight=2)
        
        # Input row trackers
        row = 0
        
        # 1. State
        tk.Label(input_panel, text="State:", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD, anchor="w").grid(row=row, column=0, sticky="ew", pady=3)
        self.var_state = tk.StringVar()
        self.combo_state = ttk.Combobox(input_panel, textvariable=self.var_state, state="readonly", font=("Segoe UI", 9))
        self.combo_state['values'] = sorted(list(STATE_DISTRICTS.keys()))
        self.combo_state.grid(row=row, column=1, sticky="ew", pady=3)
        self.combo_state.bind("<<ComboboxSelected>>", self.on_state_selected)
        row += 1
        
        # 2. District
        tk.Label(input_panel, text="District:", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD, anchor="w").grid(row=row, column=0, sticky="ew", pady=3)
        self.var_district = tk.StringVar()
        self.combo_district = ttk.Combobox(input_panel, textvariable=self.var_district, state="readonly", font=("Segoe UI", 9))
        self.combo_district.grid(row=row, column=1, sticky="ew", pady=3)
        row += 1
        
        # 3. Crop
        tk.Label(input_panel, text="Crop:", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD, anchor="w").grid(row=row, column=0, sticky="ew", pady=3)
        self.var_crop = tk.StringVar()
        self.combo_crop = ttk.Combobox(input_panel, textvariable=self.var_crop, state="readonly", font=("Segoe UI", 9))
        self.combo_crop['values'] = sorted(list(CROP_PROFILES.keys()))
        self.combo_crop.grid(row=row, column=1, sticky="ew", pady=3)
        self.combo_crop.bind("<<ComboboxSelected>>", self.on_crop_selected)
        row += 1
        
        # 4. Season
        tk.Label(input_panel, text="Season:", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD, anchor="w").grid(row=row, column=0, sticky="ew", pady=3)
        self.var_season = tk.StringVar()
        self.combo_season = ttk.Combobox(input_panel, textvariable=self.var_season, state="readonly", font=("Segoe UI", 9))
        self.combo_season['values'] = ["Kharif", "Rabi", "Summer", "Whole Year"]
        self.combo_season.grid(row=row, column=1, sticky="ew", pady=3)
        row += 1
        
        # 5. Area
        tk.Label(input_panel, text="Area (Hectares):", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD, anchor="w").grid(row=row, column=0, sticky="ew", pady=3)
        self.entry_area = ttk.Entry(input_panel, font=("Segoe UI", 9))
        self.entry_area.grid(row=row, column=1, sticky="ew", pady=3)
        row += 1
        
        # 6. Soil Type
        tk.Label(input_panel, text="Soil Type:", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD, anchor="w").grid(row=row, column=0, sticky="ew", pady=3)
        self.var_soil = tk.StringVar()
        self.combo_soil = ttk.Combobox(input_panel, textvariable=self.var_soil, state="readonly", font=("Segoe UI", 9))
        self.combo_soil['values'] = ["Clayey", "Loamy", "Sandy", "Black", "Red", "Alluvial"]
        self.combo_soil.grid(row=row, column=1, sticky="ew", pady=3)
        row += 1
        
        # Sliders/Entries for Weather & Nutrients
        # 7. Rainfall
        tk.Label(input_panel, text="Rainfall (mm):", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD, anchor="w").grid(row=row, column=0, sticky="ew", pady=2)
        self.scale_rain = tk.Scale(input_panel, from_=50, to_=2500, orient="horizontal", bg=COLOR_CARD, bd=0, highlightthickness=0, resolution=10, font=("Segoe UI", 8))
        self.scale_rain.grid(row=row, column=1, sticky="ew")
        row += 1
        
        # 8. Temperature
        tk.Label(input_panel, text="Temp (°C):", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD, anchor="w").grid(row=row, column=0, sticky="ew", pady=2)
        self.scale_temp = tk.Scale(input_panel, from_=10, to_=45, orient="horizontal", bg=COLOR_CARD, bd=0, highlightthickness=0, resolution=0.5, font=("Segoe UI", 8))
        self.scale_temp.grid(row=row, column=1, sticky="ew")
        row += 1
        
        # 9. Humidity
        tk.Label(input_panel, text="Humidity (%):", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD, anchor="w").grid(row=row, column=0, sticky="ew", pady=2)
        self.scale_humid = tk.Scale(input_panel, from_=20, to_=100, orient="horizontal", bg=COLOR_CARD, bd=0, highlightthickness=0, resolution=1, font=("Segoe UI", 8))
        self.scale_humid.grid(row=row, column=1, sticky="ew")
        row += 1
        
        # 10. pH
        tk.Label(input_panel, text="Soil pH:", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD, anchor="w").grid(row=row, column=0, sticky="ew", pady=2)
        self.scale_ph = tk.Scale(input_panel, from_=4.0, to_=9.5, orient="horizontal", bg=COLOR_CARD, bd=0, highlightthickness=0, resolution=0.1, font=("Segoe UI", 8))
        self.scale_ph.grid(row=row, column=1, sticky="ew")
        row += 1
        
        # 11. Nitrogen
        tk.Label(input_panel, text="Nitrogen (N kg/ha):", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD, anchor="w").grid(row=row, column=0, sticky="ew", pady=2)
        self.scale_n = tk.Scale(input_panel, from_=10, to_=200, orient="horizontal", bg=COLOR_CARD, bd=0, highlightthickness=0, resolution=5, font=("Segoe UI", 8))
        self.scale_n.grid(row=row, column=1, sticky="ew")
        row += 1
        
        # 12. Phosphorus
        tk.Label(input_panel, text="Phosphorus (P):", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD, anchor="w").grid(row=row, column=0, sticky="ew", pady=2)
        self.scale_p = tk.Scale(input_panel, from_=5, to_=120, orient="horizontal", bg=COLOR_CARD, bd=0, highlightthickness=0, resolution=5, font=("Segoe UI", 8))
        self.scale_p.grid(row=row, column=1, sticky="ew")
        row += 1
        
        # 13. Potassium
        tk.Label(input_panel, text="Potassium (K):", font=("Segoe UI", 9, "bold"), bg=COLOR_CARD, anchor="w").grid(row=row, column=0, sticky="ew", pady=2)
        self.scale_k = tk.Scale(input_panel, from_=5, to_=150, orient="horizontal", bg=COLOR_CARD, bd=0, highlightthickness=0, resolution=5, font=("Segoe UI", 8))
        self.scale_k.grid(row=row, column=1, sticky="ew")
        row += 1
        
        # Button frame
        btn_frame = tk.Frame(input_panel, bg=COLOR_CARD)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        
        btn_clear = tk.Button(
            btn_frame, 
            text="🔄 Reset Defaults", 
            font=("Segoe UI", 9, "bold"), 
            bg="#E0E0E0", 
            fg=COLOR_TEXT, 
            bd=0, 
            padx=10, 
            pady=6, 
            cursor="hand2",
            command=self.reset_inputs
        )
        btn_clear.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        btn_predict = tk.Button(
            btn_frame, 
            text="🌾 Predict Harvest Yield", 
            font=("Segoe UI", 10, "bold"), 
            bg=COLOR_PRIMARY, 
            fg=COLOR_LIGHT_TEXT, 
            bd=0, 
            padx=15, 
            pady=6, 
            cursor="hand2",
            command=self.predict_yield
        )
        btn_predict.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # Right Panel (Outputs & Visualizations)
        right_panel = tk.Frame(self.tab_yield, bg=COLOR_BG)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)
        
        # Output Metrics Cards (Top)
        cards_frame = tk.Frame(right_panel, bg=COLOR_BG)
        cards_frame.pack(fill="x", side="top", pady=(0, 10))
        
        # Yield Card
        self.card_yield = tk.Frame(cards_frame, bg=COLOR_CARD, bd=1, relief="solid", padx=10, pady=10)
        self.card_yield.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        tk.Label(self.card_yield, text="PREDICTED HARVEST YIELD", font=("Segoe UI", 9, "bold"), fg=COLOR_SECONDARY, bg=COLOR_CARD).pack(anchor="w")
        self.lbl_yield_val = tk.Label(self.card_yield, text="--.- t/ha", font=("Segoe UI", 24, "bold"), fg=COLOR_PRIMARY, bg=COLOR_CARD)
        self.lbl_yield_val.pack(anchor="w", pady=2)
        self.lbl_total_prod = tk.Label(self.card_yield, text="Total Production: -- tons", font=("Segoe UI", 9), fg=COLOR_TEXT, bg=COLOR_CARD)
        self.lbl_total_prod.pack(anchor="w")
        
        # Productivity Badge Card
        self.card_status = tk.Frame(cards_frame, bg=COLOR_CARD, bd=1, relief="solid", padx=10, pady=10)
        self.card_status.pack(side="left", fill="both", expand=True, padx=5)
        
        tk.Label(self.card_status, text="PRODUCTIVITY LEVEL", font=("Segoe UI", 9, "bold"), fg=COLOR_SECONDARY, bg=COLOR_CARD).pack(anchor="w")
        self.lbl_status_val = tk.Label(self.card_status, text="UNKNOWN", font=("Segoe UI", 16, "bold"), fg=COLOR_TEXT, bg=COLOR_CARD)
        self.lbl_status_val.pack(anchor="w", pady=5)
        self.lbl_status_desc = tk.Label(self.card_status, text="Enter farm inputs to calculate.", font=("Segoe UI", 8, "italic"), fg=COLOR_TEXT, bg=COLOR_CARD)
        self.lbl_status_desc.pack(anchor="w")
        
        # Fertilizer Status Quick Info
        self.card_nutrient = tk.Frame(cards_frame, bg=COLOR_CARD, bd=1, relief="solid", padx=10, pady=10)
        self.card_nutrient.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        tk.Label(self.card_nutrient, text="SOIL N-P-K NUTRIENT RATIO", font=("Segoe UI", 9, "bold"), fg=COLOR_SECONDARY, bg=COLOR_CARD).pack(anchor="w")
        self.lbl_nut_val = tk.Label(self.card_nutrient, text="-- : -- : --", font=("Segoe UI", 18, "bold"), fg=COLOR_TEXT, bg=COLOR_CARD)
        self.lbl_nut_val.pack(anchor="w", pady=2)
        self.lbl_nut_desc = tk.Label(self.card_nutrient, text="Optimal N-P-K: -- : -- : --", font=("Segoe UI", 9, "italic"), fg=COLOR_TEXT, bg=COLOR_CARD)
        self.lbl_nut_desc.pack(anchor="w")
        
        # Embedded Chart Frame (Bottom)
        chart_container = tk.LabelFrame(
            right_panel, 
            text="📊 Crop Yield Comparison under Selected Environmental Conditions", 
            font=("Segoe UI", 10, "bold"), 
            bg=COLOR_CARD, 
            fg=COLOR_PRIMARY,
            bd=1,
            relief="solid",
            padx=10,
            pady=10
        )
        chart_container.pack(fill="both", expand=True)
        
        # Create Matplotlib Figure
        self.fig_yield = plt.Figure(figsize=(5, 3.8), dpi=100)
        self.fig_yield.patch.set_facecolor(COLOR_CARD)
        self.ax_yield = self.fig_yield.add_subplot(111)
        self.canvas_yield = FigureCanvasTkAgg(self.fig_yield, master=chart_container)
        self.canvas_yield.get_tk_widget().pack(fill="both", expand=True)

    # ==================== TAB 2: SOIL ANALYSIS ====================
    def setup_tab_soil(self):
        # Left Panel - Nutrition details
        left_panel = tk.LabelFrame(
            self.tab_soil, 
            text="🌱 Soil Health Diagnosis & Nutrition Balances", 
            font=("Segoe UI", 11, "bold"), 
            bg=COLOR_CARD, 
            fg=COLOR_PRIMARY, 
            padx=15, 
            pady=15,
            bd=1,
            relief="solid"
        )
        left_panel.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)
        
        # Soil health cards/text
        tk.Label(left_panel, text="pH Scale Status:", font=("Segoe UI", 10, "bold"), bg=COLOR_CARD, fg=COLOR_TEXT).pack(anchor="w", pady=(0, 2))
        self.lbl_ph_status = tk.Label(left_panel, text="Neutral (Optimal for most crops)", font=("Segoe UI", 12, "bold"), bg=COLOR_CARD, fg=COLOR_SECONDARY)
        self.lbl_ph_status.pack(anchor="w", pady=(0, 10))
        
        tk.Label(left_panel, text="Nutrient Deficiencies & Fertilizer Recommendations:", font=("Segoe UI", 10, "bold"), bg=COLOR_CARD, fg=COLOR_TEXT).pack(anchor="w", pady=(5, 2))
        
        # Scrollable text frame for diagnostic details
        text_frame = tk.Frame(left_panel, bg=COLOR_CARD)
        text_frame.pack(fill="both", expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.txt_soil_report = tk.Text(
            text_frame, 
            font=("Consolas", 10), 
            bg="#F8F9FA", 
            fg="#212529", 
            bd=1, 
            relief="solid", 
            wrap="word", 
            yscrollcommand=scrollbar.set
        )
        self.txt_soil_report.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.txt_soil_report.yview)
        
        # Right Panel - Bar comparison
        right_panel = tk.LabelFrame(
            self.tab_soil, 
            text="📊 Soil Nutrient Comparison: Actual vs. Recommended", 
            font=("Segoe UI", 11, "bold"), 
            bg=COLOR_CARD, 
            fg=COLOR_PRIMARY, 
            padx=10, 
            pady=10,
            bd=1,
            relief="solid"
        )
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)
        
        self.fig_soil = plt.Figure(figsize=(5, 5), dpi=100)
        self.fig_soil.patch.set_facecolor(COLOR_CARD)
        self.ax_soil = self.fig_soil.add_subplot(111)
        self.canvas_soil = FigureCanvasTkAgg(self.fig_soil, master=right_panel)
        self.canvas_soil.get_tk_widget().pack(fill="both", expand=True)

    # ==================== TAB 3: FARMING INSIGHTS ====================
    def setup_tab_insights(self):
        # Left Panel - Insight texts
        left_panel = tk.LabelFrame(
            self.tab_insights, 
            text="🌾 Weather Conditions & Cultivation Best Practices", 
            font=("Segoe UI", 11, "bold"), 
            bg=COLOR_CARD, 
            fg=COLOR_PRIMARY, 
            padx=15, 
            pady=15,
            bd=1,
            relief="solid"
        )
        left_panel.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)
        
        tk.Label(left_panel, text="Climate Comfort Analysis:", font=("Segoe UI", 10, "bold"), bg=COLOR_CARD, fg=COLOR_TEXT).pack(anchor="w", pady=(0, 2))
        self.lbl_climate_status = tk.Label(left_panel, text="Optimal climate conditions for growth.", font=("Segoe UI", 12, "bold"), bg=COLOR_CARD, fg=COLOR_SECONDARY)
        self.lbl_climate_status.pack(anchor="w", pady=(0, 10))
        
        tk.Label(left_panel, text="Strategic Cultivation Guidance:", font=("Segoe UI", 10, "bold"), bg=COLOR_CARD, fg=COLOR_TEXT).pack(anchor="w", pady=(5, 2))
        
        text_frame = tk.Frame(left_panel, bg=COLOR_CARD)
        text_frame.pack(fill="both", expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.txt_insights_report = tk.Text(
            text_frame, 
            font=("Segoe UI", 10), 
            bg="#F8F9FA", 
            fg="#212529", 
            bd=1, 
            relief="solid", 
            wrap="word", 
            yscrollcommand=scrollbar.set
        )
        self.txt_insights_report.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.txt_insights_report.yview)

        # Right Panel - Comfort Zone Plot
        right_panel = tk.LabelFrame(
            self.tab_insights, 
            text="📊 Temperature & Humidity Crop Envelope Plot", 
            font=("Segoe UI", 11, "bold"), 
            bg=COLOR_CARD, 
            fg=COLOR_PRIMARY, 
            padx=10, 
            pady=10,
            bd=1,
            relief="solid"
        )
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)
        
        self.fig_insights = plt.Figure(figsize=(5, 5), dpi=100)
        self.fig_insights.patch.set_facecolor(COLOR_CARD)
        self.ax_insights = self.fig_insights.add_subplot(111)
        self.canvas_insights = FigureCanvasTkAgg(self.fig_insights, master=right_panel)
        self.canvas_insights.get_tk_widget().pack(fill="both", expand=True)

    # ==================== TAB 4: ABOUT PROJECT ====================
    def setup_tab_about(self):
        # Create a scrollable Canvas to hold about content
        canvas = tk.Canvas(self.tab_about, bg=COLOR_BG, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab_about, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLOR_BG)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        # Center about container frame
        about_card = tk.Frame(scrollable_frame, bg=COLOR_CARD, bd=1, relief="solid", padx=25, pady=25)
        about_card.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Project Description
        tk.Label(about_card, text="🌾 SMART HARVEST YIELD PREDICTION SYSTEM", font=("Segoe UI", 16, "bold"), fg=COLOR_PRIMARY, bg=COLOR_CARD).pack(anchor="w", pady=(0, 10))
        
        desc_text = (
            "This decision support system integrates Machine Learning with agricultural science to aid "
            "farmers in precision farming. It predicts crop yield (tons/hectare) based on geographic location, "
            "crop types, seasonal intervals, and real-world soil/weather parameters.\n\n"
            "By analyzing Nitrogen (N), Phosphorus (P), Potassium (K) levels, and pH balances, the system provides "
            "detailed fertilizer recommendations and agricultural advice to improve overall crop productivity."
        )
        tk.Label(about_card, text=desc_text, font=("Segoe UI", 10), bg=COLOR_CARD, justify="left", wraplength=700).pack(anchor="w", pady=(0, 15))
        
        # AI & Technical Specs
        tk.Label(about_card, text="🛠️ SYSTEM ARCHITECTURE & ALGORITHMS", font=("Segoe UI", 12, "bold"), fg=COLOR_SECONDARY, bg=COLOR_CARD).pack(anchor="w", pady=(10, 5))
        
        specs_text = (
            "• Machine Learning Engine: Random Forest Regressor (Ensemble of 150 Decision Trees)\n"
            "• Categorical Preprocessing: Scikit-Learn LabelEncoder\n"
            "• Numerical Scaling: Scikit-Learn StandardScaler\n"
            "• Data Source: Synthetic Indian Agricultural Dataset mapped with regional crop statistics (5,000 records)\n"
            "• Visualizations: Matplotlib embedded directly into Tkinter Canvas panels"
        )
        tk.Label(about_card, text=specs_text, font=("Segoe UI", 9), bg=COLOR_CARD, justify="left").pack(anchor="w", pady=(0, 15))
        
        # Viva Preparation Q&A Section
        tk.Label(about_card, text="🧠 ACADEMIC VIVA REVISION CARD (Q&A)", font=("Segoe UI", 12, "bold"), fg=COLOR_PRIMARY, bg=COLOR_CARD).pack(anchor="w", pady=(15, 8))
        
        viva_qas = [
            ("Q1: What is Precision Agriculture?", 
             "A: Precision agriculture is a modern crop management strategy that uses IT and machine learning to ensure "
             "crops and soil receive exactly what they need for optimum health and productivity (e.g. site-specific fertilizer dosing)."),
            
            ("Q2: Why use Random Forest Regressor for Yield Prediction?", 
             "A: Agricultural yields have highly non-linear, interactive relationships with weather and nutrients. Simple linear models "
             "fail to capture sweet spots (e.g., moderate rainfall is good, but dry drought or floods drop yields to zero). "
             "Random Forest is a robust ensemble of decision trees that excels at capturing complex non-linear splits without overfitting."),
            
            ("Q3: How are categoricals like crop type and state fed into the model?", 
             "A: We map categorical strings to unique integers using LabelEncoder. In python, the encoders are exported to "
             "`label_encoders.joblib` and loaded dynamically by the dashboard to encode GUI combobox selections before feeding them to the regressor."),
            
            ("Q4: What is the significance of the soil pH value in this system?", 
             "A: pH determines nutrient availability in the soil. Highly acidic (pH < 5.5) or highly alkaline (pH > 8.0) soils lock up critical nutrients "
             "like Nitrogen and Phosphorus, reducing root uptake. The system applies crop-specific pH checks and advises adding lime (for acid soils) or gypsum (for alkaline soils).")
        ]
        
        for q, a in viva_qas:
            q_label = tk.Label(about_card, text=q, font=("Segoe UI", 10, "bold"), fg=COLOR_TEXT, bg=COLOR_CARD)
            q_label.pack(anchor="w", pady=(5, 0))
            a_label = tk.Label(about_card, text=a, font=("Segoe UI", 9), fg="#4A4A4A", bg=COLOR_CARD, justify="left", wraplength=700)
            a_label.pack(anchor="w", pady=(0, 8), padx=(15, 0))

    # ==================== CONTROLLER LOGIC & EVENTS ====================
    def on_state_selected(self, event=None):
        state = self.var_state.get()
        if state in STATE_DISTRICTS:
            districts = STATE_DISTRICTS[state]
            self.combo_district['values'] = sorted(districts)
            self.combo_district.current(0)

    def on_crop_selected(self, event=None):
        crop = self.var_crop.get()
        if crop in CROP_PROFILES:
            profile = CROP_PROFILES[crop]
            # Set default values for the sliders based on optimal ranges for selected crop
            opt_n, opt_p, opt_k = profile["opt_npk"]
            self.scale_n.set(opt_n)
            self.scale_p.set(opt_p)
            self.scale_k.set(opt_k)
            
            ph_min, ph_max = profile["opt_ph"]
            self.scale_ph.set(round((ph_min + ph_max) / 2.0, 1))
            
            t_min, t_max = profile["opt_temp"]
            self.scale_temp.set(round((t_min + t_max) / 2.0, 1))
            
            r_min, r_max = profile["opt_rain"]
            self.scale_rain.set(int((r_min + r_max) / 2.0))
            
            h_min, h_max = profile["opt_humid"]
            self.scale_humid.set(int((h_min + h_max) / 2.0))
            
            # Select first season matching the crop config
            if crop == "Wheat" or crop == "Potato":
                self.combo_season.set("Rabi")
            elif crop == "Rice":
                self.combo_season.set("Kharif")
            elif crop == "Sugarcane":
                self.combo_season.set("Whole Year")
            else:
                self.combo_season.set("Kharif")

    def reset_inputs(self):
        # Set State & District defaults
        self.combo_state.current(0)
        self.on_state_selected()
        
        # Set Crop defaults
        self.combo_crop.current(0) # Rice
        self.on_crop_selected()
        
        self.entry_area.delete(0, tk.END)
        self.entry_area.insert(0, "100.0")
        
        self.combo_soil.set("Loamy")
        
        # Re-run prediction after resetting defaults
        if self.models_loaded:
            self.predict_yield()

    def predict_yield(self):
        if not self.models_loaded:
            messagebox.showwarning("Models Not Loaded", "Machine Learning models could not be found. Please check setup.")
            return
            
        try:
            # 1. Fetch values from GUI
            state = self.var_state.get()
            district = self.var_district.get()
            crop = self.var_crop.get()
            season = self.var_season.get()
            soil_type = self.var_soil.get()
            
            try:
                area = float(self.entry_area.get())
                if area <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Input Error", "Area must be a valid positive decimal number (in Hectares).")
                return
                
            rainfall = float(self.scale_rain.get())
            temperature = float(self.scale_temp.get())
            humidity = float(self.scale_humid.get())
            ph = float(self.scale_ph.get())
            n = float(self.scale_n.get())
            p = float(self.scale_p.get())
            k = float(self.scale_k.get())
            
            # 2. Engineered feature calculations
            npk_total = n + p + k
            temp_humid_ratio = temperature / (humidity + 1.0)
            rain_temp_ratio = rainfall / (temperature + 1.0)
            
            # 3. Build Input DataFrame (Align exactly with training column indexes)
            input_data = {
                "state": [state],
                "district": [district],
                "crop": [crop],
                "season": [season],
                "area": [area],
                "rainfall": [rainfall],
                "temperature": [temperature],
                "humidity": [humidity],
                "soil_type": [soil_type],
                "ph": [ph],
                "nitrogen": [n],
                "phosphorus": [p],
                "potassium": [k],
                "NPK_Total": [npk_total],
                "temp_humidity_ratio": [temp_humid_ratio],
                "rain_temp_ratio": [rain_temp_ratio]
            }
            
            df_in = pd.DataFrame(input_data)
            
            # 4. Perform Label Encoding
            df_encoded = df_in.copy()
            for col in ['state', 'district', 'crop', 'season', 'soil_type']:
                le = self.label_encoders[col]
                # Fallback in case of unseen values, default to first class
                try:
                    df_encoded[col] = le.transform(df_in[col])
                except ValueError:
                    df_encoded[col] = le.transform([le.classes_[0]])
            
            # Reorder columns to match training set
            df_encoded = df_encoded[self.feature_columns]
            
            # 5. Apply Scaling
            scaled_features = self.scaler.transform(df_encoded)
            
            # 6. Predict Yield
            predicted_yield = self.model.predict(scaled_features)[0]
            
            # Update GUI cards
            self.lbl_yield_val.config(text=f"{predicted_yield:.2f} t/ha")
            total_prod = predicted_yield * area
            self.lbl_total_prod.config(text=f"Total Production: {total_prod:.2f} tons")
            
            # N-P-K Value displays
            self.lbl_nut_val.config(text=f"{int(n)} : {int(p)} : {int(k)}")
            profile = CROP_PROFILES[crop]
            opt_n, opt_p, opt_k = profile["opt_npk"]
            self.lbl_nut_desc.config(text=f"Optimal N-P-K: {opt_n} : {opt_p} : {opt_k}")
            
            # Update Productivity Level status
            avg_crop_yield = profile["avg_yield"]
            yield_pct = (predicted_yield / avg_crop_yield) * 100
            
            if yield_pct >= 115:
                status_text = "HIGH PRODUCTIVITY"
                status_bg = COLOR_LIGHT_GREEN
                status_fg = "#2E7D32"
                desc_text = f"Yield is {yield_pct - 100:.1f}% above standard averages.\nExcellent conditions!"
            elif yield_pct >= 80:
                status_text = "MODERATE PRODUCTIVITY"
                status_bg = COLOR_LIGHT_ORANGE
                status_fg = "#EF6C00"
                desc_text = f"Yield is within nominal crop average range.\nGood balance."
            else:
                status_text = "LOW PRODUCTIVITY"
                status_bg = COLOR_LIGHT_RED
                status_fg = "#C62828"
                desc_text = f"Yield is {100 - yield_pct:.1f}% below crop expectations.\nCheck deficiencies!"
                
            # Apply styling changes to the badge card
            self.card_status.config(bg=status_bg)
            for widget in self.card_status.winfo_children():
                widget.config(bg=status_bg)
            self.lbl_status_val.config(text=status_text, fg=status_fg)
            self.lbl_status_desc.config(text=desc_text, fg=COLOR_TEXT)
            
            # Generate Insights Reports & Charts
            self.generate_soil_report(crop, n, p, k, ph)
            self.generate_insights_report(crop, temperature, rainfall, humidity)
            
            # Render Plots
            self.update_yield_plot(df_encoded.copy(), crop)
            self.update_soil_plot(crop, n, p, k)
            self.update_insights_plot(crop, temperature, humidity)
            
        except Exception as e:
            messagebox.showerror("Prediction Execution Failed", f"An error occurred while generating models prediction.\n\nError: {e}")

    # ==================== REPORTS GENERATION ====================
    def generate_soil_report(self, crop, n, p, k, ph):
        profile = CROP_PROFILES[crop]
        opt_n, opt_p, opt_k = profile["opt_npk"]
        opt_ph_min, opt_ph_max = profile["opt_ph"]
        
        self.txt_soil_report.config(state="normal")
        self.txt_soil_report.delete("1.0", tk.END)
        
        report = []
        report.append(f"📊 SOIL DIAGNOSTIC REPORT FOR: {crop.upper()}")
        report.append("="*50 + "\n")
        
        # pH Check
        report.append(f"1. Soil Reaction (pH Analysis):")
        report.append(f"   • Current soil pH: {ph:.2f}")
        report.append(f"   • Optimal range: {opt_ph_min} - {opt_ph_max}")
        if opt_ph_min <= ph <= opt_ph_max:
            report.append("   ✔ STATUS: Soil pH is in the OPTIMAL zone. Soil is healthy for nutrient absorption.")
            self.lbl_ph_status.config(text=f"Optimal ({ph:.1f} pH) - Healthy", fg="#2E7D32")
        else:
            if ph < opt_ph_min:
                report.append("   ⚠ STATUS: Soil is too ACIDIC.")
                report.append("     Remedy: Add agricultural lime (calcium carbonate) to raise pH.")
                self.lbl_ph_status.config(text=f"Acidic ({ph:.1f} pH) - Needs Lime", fg="#C62828")
            else:
                report.append("   ⚠ STATUS: Soil is too ALKALINE.")
                report.append("     Remedy: Add gypsum (calcium sulfate) or elemental sulfur to lower pH.")
                self.lbl_ph_status.config(text=f"Alkaline ({ph:.1f} pH) - Needs Gypsum", fg="#C62828")
        report.append("\n" + "-"*40 + "\n")
        
        # NPK Nutrient checks
        report.append("2. N-P-K Nutrient Analysis (in kg/ha):")
        
        # Nitrogen
        n_diff = n - opt_n
        n_pct = (n / opt_n) * 100
        report.append(f"   • Nitrogen (N): Current {int(n)}, Optimal {opt_n}")
        if n_pct >= 90 and n_pct <= 120:
            report.append("     ✔ STATUS: Optimal nitrogen content.")
        elif n_pct < 90:
            report.append(f"     ⚠ STATUS: Deficient in Nitrogen (Shortfall of {int(abs(n_diff))} kg/ha).")
            report.append(f"       Remedy: Apply urea (46% N) at ~{int(abs(n_diff) * 2.17)} kg/ha.")
        else:
            report.append(f"     ℹ STATUS: Excess Nitrogen (Surplus of {int(n_diff)} kg/ha).")
            report.append("       Warning: Excessive N leads to weak stems and high pest vulnerability.")
            
        # Phosphorus
        p_diff = p - opt_p
        p_pct = (p / opt_p) * 100
        report.append(f"   • Phosphorus (P): Current {int(p)}, Optimal {opt_p}")
        if p_pct >= 90 and p_pct <= 125:
            report.append("     ✔ STATUS: Optimal phosphorus content.")
        elif p_pct < 90:
            report.append(f"     ⚠ STATUS: Deficient in Phosphorus (Shortfall of {int(abs(p_diff))} kg/ha).")
            report.append(f"       Remedy: Apply Single Superphosphate (SSP) at ~{int(abs(p_diff) * 6.25)} kg/ha or DAP.")
        else:
            report.append(f"     ℹ STATUS: Excess Phosphorus (Surplus of {int(p_diff)} kg/ha).")
            
        # Potassium
        k_diff = k - opt_k
        k_pct = (k / opt_k) * 100
        report.append(f"   • Potassium (K): Current {int(k)}, Optimal {opt_k}")
        if k_pct >= 90 and k_pct <= 125:
            report.append("     ✔ STATUS: Optimal potassium content.")
        elif k_pct < 90:
            report.append(f"     ⚠ STATUS: Deficient in Potassium (Shortfall of {int(abs(k_diff))} kg/ha).")
            report.append(f"       Remedy: Apply Muriate of Potash (MOP) at ~{int(abs(k_diff) * 1.67)} kg/ha.")
        else:
            report.append(f"     ℹ STATUS: Excess Potassium (Surplus of {int(k_diff)} kg/ha).")
            
        self.txt_soil_report.insert(tk.END, "\n".join(report))
        self.txt_soil_report.config(state="disabled")

    def generate_insights_report(self, crop, temperature, rainfall, humidity):
        profile = CROP_PROFILES[crop]
        
        self.txt_insights_report.config(state="normal")
        self.txt_insights_report.delete("1.0", tk.END)
        
        report = []
        report.append(f"🌾 PRECISION CULTIVATION GUIDE: {crop.upper()}")
        report.append("="*50 + "\n")
        
        # 1. Weather analysis
        report.append("1. Ambient Climate Analysis:")
        
        # Temp checks
        t_min, t_max = profile["opt_temp"]
        report.append(f"   • Current Temperature: {temperature:.1f} °C (Optimal: {t_min} - {t_max} °C)")
        if t_min <= temperature <= t_max:
            report.append("     ✔ Temperature is highly favorable for plant physiological growth.")
        else:
            if temperature < t_min:
                report.append("     ⚠ Cold Stress: Current temperature is below optimal limits. May retard growth rates.")
            else:
                report.append("     ⚠ Heat Stress: High heat detected. Risk of water loss through high evapotranspiration.")
                
        # Rain checks
        r_min, r_max = profile["opt_rain"]
        report.append(f"   • Season Rainfall: {int(rainfall)} mm (Optimal: {r_min} - {r_max} mm)")
        if r_min <= rainfall <= r_max:
            report.append("     ✔ Rainfall is within ideal limits. Adequate natural hydration.")
            self.lbl_climate_status.config(text="Favorable Climate Balance", fg="#2E7D32")
        else:
            if rainfall < r_min:
                report.append("     ⚠ Moisture Deficit: Rainfall is below requirements. Dry spells expected.")
                report.append("       Advice: Provide supplementary drip or sprinkler irrigation.")
                self.lbl_climate_status.config(text="Dry Stress: Irrigation Required", fg="#EF6C00")
            else:
                report.append("     ⚠ Excess Water: Rainfall exceeds optimal requirements. Danger of waterlogging.")
                report.append("       Advice: Ensure effective field drainage to avoid root rot and collar rot.")
                self.lbl_climate_status.config(text="Wet Stress: Drainage Required", fg="#C62828")
                
        # Humidity checks
        h_min, h_max = profile["opt_humid"]
        report.append(f"   • Relative Humidity: {int(humidity)}% (Optimal: {h_min} - {h_max}%)")
        if h_min <= humidity <= h_max:
            report.append("     ✔ Humidity is comfortable for transpiration cycles.")
        else:
            if humidity > h_max:
                report.append("     ⚠ High Humidity Warning: High dampness increases risks of fungal spores and blight diseases.")
                report.append("       Advice: Keep spacing between rows and spray prophylactic fungicide.")
                
        report.append("\n" + "-"*40 + "\n")
        
        # 2. Farming Advice
        report.append("2. Strategic Crop Farming Recommendations:")
        report.append(f"   • {profile['tips']}")
        
        self.txt_insights_report.insert(tk.END, "\n".join(report))
        self.txt_insights_report.config(state="disabled")

    # ==================== MATPLOTLIB PLOT UPDATES ====================
    def update_yield_plot(self, current_features, selected_crop):
        self.ax_yield.clear()
        
        # Let's run a comparative yield prediction for all crops under the same inputs
        all_crops = list(CROP_PROFILES.keys())
        predicted_yields = []
        
        # Label Encoder for Crop
        crop_le = self.label_encoders['crop']
        
        # Iterate over each crop and predict what it would yield under these environment params
        for crop_name in all_crops:
            df_temp = current_features.copy()
            # Set the crop code for this specific crop
            df_temp['crop'] = crop_le.transform([crop_name])[0]
            
            # Predict
            scaled = self.scaler.transform(df_temp)
            yield_pred = self.model.predict(scaled)[0]
            predicted_yields.append(yield_pred)
            
        # Draw a beautiful horizontal bar chart
        colors = [COLOR_SECONDARY if crop == selected_crop else "#A8C3B1" for crop in all_crops]
        
        y_pos = np.arange(len(all_crops))
        bars = self.ax_yield.barh(y_pos, predicted_yields, align='center', color=colors, edgecolor='gray', height=0.6)
        self.ax_yield.set_yticks(y_pos)
        self.ax_yield.set_yticklabels(all_crops, fontdict={"fontsize": 9, "fontweight": "bold"})
        self.ax_yield.invert_yaxis()  # top-down crop display
        self.ax_yield.set_xlabel('Predicted Yield (Tons / Hectare)', fontsize=9, fontweight="bold")
        self.ax_yield.set_title(f'Comparison of Crop Yields for Selected Soil & Weather', fontsize=10, fontweight="bold", pad=12, color=COLOR_PRIMARY)
        
        # Value annotations
        for bar in bars:
            width = bar.get_width()
            self.ax_yield.text(
                width + (max(predicted_yields) * 0.01), 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', 
                va='center', 
                ha='left', 
                fontsize=8, 
                fontweight='bold'
            )
            
        # Highlight margins and styling
        self.ax_yield.spines['top'].set_visible(False)
        self.ax_yield.spines['right'].set_visible(False)
        self.ax_yield.set_xlim(0, max(predicted_yields) * 1.15)
        self.fig_yield.tight_layout()
        self.canvas_yield.draw()

    def update_soil_plot(self, crop, n, p, k):
        self.ax_soil.clear()
        
        # Fetch optimal values
        opt_n, opt_p, opt_k = CROP_PROFILES[crop]["opt_npk"]
        
        categories = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
        actual_vals = [n, p, k]
        optimal_vals = [opt_n, opt_p, opt_k]
        
        x = np.arange(len(categories))
        width = 0.35
        
        # Plot double bar charts
        rects1 = self.ax_soil.bar(x - width/2, actual_vals, width, label='Current Level', color='#A2D2FF', edgecolor='gray')
        rects2 = self.ax_soil.bar(x + width/2, optimal_vals, width, label='Optimal Level', color=COLOR_SECONDARY, edgecolor='gray')
        
        self.ax_soil.set_ylabel('Soil Nutrient Indexes (kg/ha)', fontsize=9, fontweight="bold")
        self.ax_soil.set_title(f'NPK Nutrient Discrepancy for {crop}', fontsize=10, fontweight="bold", color=COLOR_PRIMARY)
        self.ax_soil.set_xticks(x)
        self.ax_soil.set_xticklabels(categories, fontsize=9, fontweight="bold")
        self.ax_soil.legend(fontsize=9)
        
        # Annotate actual vs optimal values on top of bars
        for r in rects1:
            h = r.get_height()
            self.ax_soil.text(r.get_x() + r.get_width()/2., h + 3, f'{int(h)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        for r in rects2:
            h = r.get_height()
            self.ax_soil.text(r.get_x() + r.get_width()/2., h + 3, f'{int(h)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
        self.ax_soil.spines['top'].set_visible(False)
        self.ax_soil.spines['right'].set_visible(False)
        self.ax_soil.set_ylim(0, max(max(actual_vals), max(optimal_vals)) * 1.2)
        
        self.fig_soil.tight_layout()
        self.canvas_soil.draw()

    def update_insights_plot(self, crop, temperature, humidity):
        self.ax_insights.clear()
        
        # Fetch crop envelope profile
        profile = CROP_PROFILES[crop]
        t_min, t_max = profile["opt_temp"]
        h_min, h_max = profile["opt_humid"]
        
        # Draw optimal comfort box (rectangle)
        rect = plt.Rectangle(
            (t_min, h_min), 
            t_max - t_min, 
            h_max - h_min, 
            facecolor='#D8F3DC', 
            alpha=0.6, 
            edgecolor='#2D6A4F', 
            linestyle='--', 
            linewidth=1.5,
            label='Optimal Growth Envelope'
        )
        self.ax_insights.add_patch(rect)
        
        # Draw current input weather point (Red Star)
        self.ax_insights.scatter(temperature, humidity, color='#D32F2F', marker='*', s=150, zorder=5, label='Current Climate Point')
        
        # Settings
        self.ax_insights.set_xlabel('Temperature (°C)', fontsize=9, fontweight="bold")
        self.ax_insights.set_ylabel('Humidity (%)', fontsize=9, fontweight="bold")
        self.ax_insights.set_title(f'{crop} Weather Envelope & Climate Suitability', fontsize=10, fontweight="bold", color=COLOR_PRIMARY)
        
        # Set bounds dynamically around the comfort box
        self.ax_insights.set_xlim(min(10, t_min - 5), max(45, t_max + 5))
        self.ax_insights.set_ylim(min(20, h_min - 10), max(100, h_max + 10))
        self.ax_insights.legend(loc='lower right', fontsize=8)
        self.ax_insights.grid(True, linestyle=':', alpha=0.6)
        
        self.fig_insights.tight_layout()
        self.canvas_insights.draw()

if __name__ == "__main__":
    app = PrecisionFarmingApp()
    app.mainloop()
