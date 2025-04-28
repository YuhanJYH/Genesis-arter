friction_coefficients = {
    "road": {
        "dry_asphalt_concrete": {
            "min": 0.7,
            "max": 0.9,
            "notes": "Can be higher for racing tires (up to 1.8+)",
            "peak_values": [0.8, 0.9]
        },
        "wet_asphalt_concrete": {
            "min": 0.4,
            "max": 0.7,
            "notes": "Water significantly reduces friction.",
            "alt_range": [0.45, 0.6]
        },
        "snow_hard_packed": {
            "min": 0.2,
            "max": 0.3
        },
        "ice": {
            "min": 0.05,
            "max": 0.2,
            "notes": "Very low friction.",
            "alt_range": [0.07, 0.1]
        },
        "gravel": {
            "min": 0.55,
            "max": 0.6,
            "notes": "Dry gravel"
        },
        "earth_road_dry": {
            "min": 0.65,
            "max": 0.68
        },
        "earth_road_wet": {
            "min": 0.4,
            "max": 0.5
        },
        "loose_gravel":{
            "min": 0.35,
            "max": 0.35, # Corrected to be the same, as only one value was provided.
            "notes": "Significantly reduced friction compared to dry gravel. Original dry gravel value was 0.8 which drops to this."
        }
    },
    "sand": {
        "solid_sand": {
            "rolling_friction_min": 0.04,
            "rolling_friction_max": 0.08
        },
        "loose_sand": {
            "rolling_friction_min": 0.2,
            "rolling_friction_max": 0.4,
            "alt_range": [0.5, 0.7] #Added alternative range
        }
    }
}