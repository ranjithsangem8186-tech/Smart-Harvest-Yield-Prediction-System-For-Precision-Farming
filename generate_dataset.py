import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Total records to generate
NUM_RECORDS = 5000

# Mapping of states to districts
state_district_map = {
    "Andhra Pradesh": ["Guntur", "Krishna", "Kurnool", "Anantapur", "East Godavari"],
    "Telangana": ["Karimnagar", "Warangal", "Nalgonda", "Nizamabad", "Medak"],
    "Punjab": ["Ludhiana", "Amritsar", "Patiala", "Jalandhar", "Bathinda"],
    "Haryana": ["Karnal", "Hisar", "Sirsa", "Rohtak", "Sonipat"],
    "Uttar Pradesh": ["Meerut", "Bareilly", "Varanasi", "Aligarh", "Gorakhpur"],
    "Maharashtra": ["Nashik", "Pune", "Nagpur", "Ahmednagar", "Jalgaon"],
    "Karnataka": ["Belagavi", "Dharwad", "Tumakuru", "Mysuru", "Raichur"],
    "Tamil Nadu": ["Coimbatore", "Thanjavur", "Erode", "Salem", "Madurai"]
}

states = list(state_district_map.keys())

# Crops and their attributes
crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Groundnut", "Tomato", "Potato"]

# Crop configurations
crop_configs = {
    "Rice": {
        "seasons": ["Kharif", "Summer"],
        "soils": ["Clayey", "Loamy", "Alluvial"],
        "base_yield": 3.8,  # tons/ha
        "opt_temp": (25, 33),
        "opt_rain": (1000, 2000),
        "opt_humid": (70, 90),
        "opt_ph": (6.0, 7.5),
        "opt_npk": (120, 60, 60)  # N, P, K
    },
    "Wheat": {
        "seasons": ["Rabi"],
        "soils": ["Loamy", "Alluvial", "Clayey"],
        "base_yield": 4.0,
        "opt_temp": (15, 23),
        "opt_rain": (80, 250),
        "opt_humid": (45, 65),
        "opt_ph": (6.2, 7.8),
        "opt_npk": (120, 50, 40)
    },
    "Maize": {
        "seasons": ["Kharif", "Rabi"],
        "soils": ["Loamy", "Red", "Alluvial"],
        "base_yield": 3.2,
        "opt_temp": (20, 30),
        "opt_rain": (500, 1000),
        "opt_humid": (55, 80),
        "opt_ph": (5.8, 7.5),
        "opt_npk": (120, 60, 40)
    },
    "Cotton": {
        "seasons": ["Kharif"],
        "soils": ["Black", "Clayey", "Alluvial"],
        "base_yield": 2.2,
        "opt_temp": (25, 35),
        "opt_rain": (600, 1200),
        "opt_humid": (50, 75),
        "opt_ph": (6.0, 8.0),
        "opt_npk": (100, 50, 50)
    },
    "Sugarcane": {
        "seasons": ["Whole Year"],
        "soils": ["Clayey", "Alluvial", "Loamy", "Black"],
        "base_yield": 78.0,
        "opt_temp": (26, 38),
        "opt_rain": (1200, 2200),
        "opt_humid": (65, 85),
        "opt_ph": (6.5, 7.8),
        "opt_npk": (150, 80, 80)
    },
    "Groundnut": {
        "seasons": ["Kharif", "Rabi"],
        "soils": ["Sandy", "Loamy", "Red"],
        "base_yield": 2.1,
        "opt_temp": (22, 32),
        "opt_rain": (400, 800),
        "opt_humid": (50, 70),
        "opt_ph": (6.0, 7.5),
        "opt_npk": (20, 40, 40)
    },
    "Tomato": {
        "seasons": ["Kharif", "Rabi", "Summer"],
        "soils": ["Loamy", "Sandy", "Red", "Alluvial"],
        "base_yield": 24.0,
        "opt_temp": (20, 30),
        "opt_rain": (400, 700),
        "opt_humid": (50, 75),
        "opt_ph": (6.0, 7.0),
        "opt_npk": (100, 60, 120)
    },
    "Potato": {
        "seasons": ["Rabi"],
        "soils": ["Sandy", "Loamy", "Alluvial"],
        "base_yield": 20.0,
        "opt_temp": (15, 22),
        "opt_rain": (150, 400),
        "opt_humid": (45, 65),
        "opt_ph": (5.2, 6.5),
        "opt_npk": (120, 60, 120)
    }
}

# Soil types
soil_types = ["Clayey", "Loamy", "Sandy", "Black", "Red", "Alluvial"]

data_list = []

for _ in range(NUM_RECORDS):
    # Select state & district
    state = np.random.choice(states)
    district = np.random.choice(state_district_map[state])
    
    # Select crop
    crop = np.random.choice(crops)
    config = crop_configs[crop]
    
    # Select season valid for the crop
    season = np.random.choice(config["seasons"])
    
    # Area (in hectares)
    area = np.random.uniform(50, 4500)
    
    # Generate Weather based on crop requirements (mostly reasonable with some extreme conditions)
    # Rainfall
    rain_range = config["opt_rain"]
    if np.random.rand() < 0.85:
        # 85% in normal range
        rainfall = np.random.uniform(rain_range[0] * 0.8, rain_range[1] * 1.2)
    else:
        # Drought or excess rain
        rainfall = np.random.uniform(50, 2800)
        
    # Temperature
    temp_range = config["opt_temp"]
    if np.random.rand() < 0.85:
        temperature = np.random.uniform(temp_range[0] - 2, temp_range[1] + 2)
    else:
        temperature = np.random.uniform(10, 42)
        
    # Humidity
    humid_range = config["opt_humid"]
    if np.random.rand() < 0.85:
        humidity = np.random.uniform(humid_range[0] - 5, humid_range[1] + 5)
    else:
        humidity = np.random.uniform(25, 95)
        
    # Soil type
    soil_type = np.random.choice(soil_types)
    
    # pH value
    ph_range = config["opt_ph"]
    if np.random.rand() < 0.90:
        ph = np.random.uniform(ph_range[0] - 0.5, ph_range[1] + 0.5)
    else:
        ph = np.random.uniform(4.0, 9.0)
        
    # N-P-K Levels
    opt_n, opt_p, opt_k = config["opt_npk"]
    nitrogen = np.random.uniform(max(10, opt_n - 40), opt_n + 40)
    phosphorus = np.random.uniform(max(5, opt_p - 25), opt_p + 25)
    potassium = np.random.uniform(max(5, opt_k - 30), opt_k + 30)
    
    # Yield calculation based on factors
    base_yield = config["base_yield"]
    
    # 1. Soil type multiplier
    soil_mult = 1.1 if soil_type in config["soils"] else 0.85
    
    # 2. Weather multipliers
    # Temperature multiplier
    t_min, t_max = config["opt_temp"]
    if t_min <= temperature <= t_max:
        temp_mult = 1.05
    else:
        deviation = min(abs(temperature - t_min), abs(temperature - t_max))
        temp_mult = max(0.65, 1.0 - 0.05 * deviation)
        
    # Rainfall multiplier
    r_min, r_max = config["opt_rain"]
    if r_min <= rainfall <= r_max:
        rain_mult = 1.05
    else:
        # Drought or flooding effects
        if rainfall < r_min:
            rain_mult = max(0.6, 1.0 - 0.4 * ((r_min - rainfall) / r_min))
        else:
            rain_mult = max(0.7, 1.0 - 0.2 * ((rainfall - r_max) / r_max))
            
    # Humidity multiplier
    h_min, h_max = config["opt_humid"]
    if h_min <= humidity <= h_max:
        humid_mult = 1.02
    else:
        deviation = min(abs(humidity - h_min), abs(humidity - h_max))
        humid_mult = max(0.8, 1.0 - 0.01 * deviation)
        
    # 3. Nutrient deviation multiplier
    n_dev = abs(nitrogen - opt_n) / opt_n
    p_dev = abs(phosphorus - opt_p) / opt_p
    k_dev = abs(potassium - opt_k) / opt_k
    nutrient_dev_avg = (n_dev + p_dev + k_dev) / 3.0
    nut_mult = max(0.7, 1.08 - 0.3 * nutrient_dev_avg)
    
    # 4. pH multiplier
    ph_min, ph_max = config["opt_ph"]
    if ph_min <= ph <= ph_max:
        ph_mult = 1.05
    else:
        deviation = min(abs(ph - ph_min), abs(ph - ph_max))
        ph_mult = max(0.7, 1.0 - 0.2 * deviation)
        
    # Calculate final yield (Base Yield * Mults) + Noise
    yld = base_yield * soil_mult * temp_mult * rain_mult * humid_mult * nut_mult * ph_mult
    
    # Add random noise (around 5% of base yield)
    noise = np.random.normal(0, 0.04 * base_yield)
    yld += noise
    
    # Ensure yield doesn't go below a sensible minimum (e.g. 10% of base yield)
    yld = max(base_yield * 0.1, yld)
    
    # Keep decimal places neat
    data_list.append({
        "state": state,
        "district": district,
        "crop": crop,
        "season": season,
        "area": round(area, 2),
        "rainfall": round(rainfall, 2),
        "temperature": round(temperature, 2),
        "humidity": round(humidity, 2),
        "soil_type": soil_type,
        "ph": round(ph, 2),
        "nitrogen": round(nitrogen, 2),
        "phosphorus": round(phosphorus, 2),
        "potassium": round(potassium, 2),
        "yield": round(yld, 2)
    })

# Save to CSV
df = pd.DataFrame(data_list)
df.to_csv("dataset.csv", index=False)
print("dataset.csv successfully generated inside:", os.getcwd())
