"""
Dataset Enhancement Script
Enhances the existing dataset with realistic seasonal patterns covering multiple years
"""

import pandas as pd
import numpy as np
import os
from backend.utils.fwi import FWICalculator

def enhance_dataset_with_seasonal_data(input_path='data/indian_forest_fires.csv', 
                                        output_path='data/indian_forest_fires_enhanced.csv',
                                        years=10):
    """
    Enhance the existing dataset with realistic seasonal patterns over multiple years
    
    Args:
        input_path: Path to existing dataset
        output_path: Path to save enhanced dataset
        years: Number of years to generate data for
    """
    print("=" * 70)
    print("ENHANCING DATASET WITH REALISTIC SEASONAL PATTERNS")
    print("=" * 70)
    
    # Load existing dataset
    if os.path.exists(input_path):
        df_existing = pd.read_csv(input_path)
        print(f"\nLoaded existing dataset: {len(df_existing)} records")
    else:
        print(f"\nExisting dataset not found at {input_path}")
        df_existing = pd.DataFrame()
    
    # Initialize FWI calculator
    fwi_calculator = FWICalculator()
    
    # Define regions with their climate characteristics
    regions = [
        {'name': 'Uttarakhand', 'base_temp': 22, 'temp_range': 15, 'base_rh': 65, 'monsoon_rain': True},
        {'name': 'Himachal Pradesh', 'base_temp': 18, 'temp_range': 20, 'base_rh': 60, 'monsoon_rain': True},
        {'name': 'Madhya Pradesh', 'base_temp': 26, 'temp_range': 18, 'base_rh': 55, 'monsoon_rain': True},
        {'name': 'Maharashtra', 'base_temp': 28, 'temp_range': 12, 'base_rh': 50, 'monsoon_rain': True},
        {'name': 'Karnataka', 'base_temp': 27, 'temp_range': 10, 'base_rh': 58, 'monsoon_rain': True},
        {'name': 'Odisha', 'base_temp': 29, 'temp_range': 12, 'base_rh': 62, 'monsoon_rain': True},
        {'name': 'Rajasthan', 'base_temp': 30, 'temp_range': 20, 'base_rh': 40, 'monsoon_rain': False},
        {'name': 'Jammu and Kashmir', 'base_temp': 15, 'temp_range': 25, 'base_rh': 55, 'monsoon_rain': False},
        {'name': 'Kerala', 'base_temp': 28, 'temp_range': 8, 'base_rh': 75, 'monsoon_rain': True},
        {'name': 'Tamil Nadu', 'base_temp': 30, 'temp_range': 10, 'base_rh': 60, 'monsoon_rain': True},
    ]
    
    all_data = []
    
    # Generate data for multiple years
    start_year = 2014
    end_year = start_year + years - 1
    
    print(f"\nGenerating data for {years} years ({start_year} to {end_year})...")
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Sample 5 days per month to capture variation
            for day in [1, 8, 15, 22, 29]:
                if day > 31:
                    continue
                    
                for region in regions:
                    # Seasonal temperature adjustment
                    # Summer (Mar-May): Higher temps
                    # Monsoon (Jun-Sep): Moderate temps
                    # Winter (Dec-Feb): Lower temps
                    
                    if month in [3, 4, 5]:
                        temp_adj = 5
                        rh_adj = -15
                    elif month in [6, 7, 8, 9]:
                        temp_adj = 0
                        rh_adj = 20
                    elif month in [12, 1, 2]:
                        temp_adj = -8
                        rh_adj = 10
                    else:  # Oct, Nov
                        temp_adj = 2
                        rh_adj = -5
                    
                    # Add some random variation
                    temp = region['base_temp'] + temp_adj + np.random.normal(0, 3)
                    temp = max(0, min(50, temp))
                    
                    rh = region['base_rh'] + rh_adj + np.random.normal(0, 10)
                    rh = max(10, min(100, rh))
                    
                    # Wind speed with seasonal variation
                    wind = 8 + np.random.normal(0, 4)
                    wind = max(0, min(40, wind))
                    
                    # Rainfall based on season and region
                    if region['monsoon_rain'] and month in [6, 7, 8, 9]:
                        # Monsoon season - heavy rain
                        rain = np.random.exponential(5) if np.random.random() < 0.7 else 0
                    elif month in [12, 1, 2]:
                        # Winter - light rain
                        rain = np.random.exponential(1) if np.random.random() < 0.3 else 0
                    elif not region['monsoon_rain'] and month in [3, 4, 5]:
                        # Dry season for arid regions
                        rain = np.random.exponential(0.5) if np.random.random() < 0.2 else 0
                    else:
                        # Normal conditions
                        rain = np.random.exponential(2) if np.random.random() < 0.4 else 0
                    
                    rain = max(0, min(50, rain))
                    
                    # Calculate FWI components
                    fwi_result = fwi_calculator.compute_all(
                        temp=temp,
                        rh=rh,
                        wind=wind,
                        rain=rain,
                        month=month
                    )
                    
                    ffmc = fwi_result['FFMC']
                    dmc = fwi_result['DMC']
                    dc = fwi_result['DC']
                    isi = fwi_result['ISI']
                    bui = fwi_result['BUI']
                    fwi = fwi_result['FWI']
                    
                    # Determine fire class based on FWI with realistic distribution
                    # This ensures we get all categories
                    if fwi < 1:
                        fire_class = 'not fire'
                    elif fwi < 5:
                        fire_class = 'not fire'
                    elif fwi < 10:
                        # Mix of fire and not fire based on conditions
                        fire_class = 'fire' if np.random.random() < 0.3 else 'not fire'
                    elif fwi < 18:
                        fire_class = 'fire'
                    else:
                        fire_class = 'fire'
                    
                    row = {
                        'day': day,
                        'month': month,
                        'year': year,
                        'Temperature': round(temp, 1),
                        'RH': round(rh, 1),
                        'Ws': round(wind, 1),
                        'Rain': round(rain, 2),
                        'FFMC': round(ffmc, 1),
                        'DMC': round(dmc, 1),
                        'DC': round(dc, 1),
                        'ISI': round(isi, 1),
                        'BUI': round(bui, 1),
                        'FWI': round(fwi, 1),
                        'Classes': fire_class,
                        'Region': region['name']
                    }
                    
                    all_data.append(row)
    
    # Create DataFrame
    df_enhanced = pd.DataFrame(all_data)
    
    # Combine with existing data if available
    if not df_existing.empty:
        df_combined = pd.concat([df_existing, df_enhanced], ignore_index=True)
    else:
        df_combined = df_enhanced
    
    # Save enhanced dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    
    print(f"\nEnhanced dataset saved to {output_path}")
    print(f"Total records: {len(df_combined)}")
    print(f"Years covered: {df_combined['year'].min()} to {df_combined['year'].max()}")
    print(f"Regions: {df_combined['Region'].nunique()}")
    
    print("\nFire class distribution:")
    print(df_combined['Classes'].value_counts())
    
    print("\nFWI distribution:")
    print(f"Min FWI: {df_combined['FWI'].min():.2f}")
    print(f"Max FWI: {df_combined['FWI'].max():.2f}")
    print(f"Mean FWI: {df_combined['FWI'].mean():.2f}")
    print(f"Median FWI: {df_combined['FWI'].median():.2f}")
    
    # Seasonal analysis
    df_combined['season'] = df_combined['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Monsoon', 10: 'Monsoon', 11: 'Monsoon'
    })
    
    print("\nTemperature by season:")
    print(df_combined.groupby('season')['Temperature'].agg(['min', 'max', 'mean']))
    
    print("\nFWI by season:")
    print(df_combined.groupby('season')['FWI'].agg(['min', 'max', 'mean']))
    
    print("\nFire class by season:")
    season_fire = df_combined.groupby('season')['Classes'].value_counts(normalize=True).unstack()
    print(season_fire)
    
    print("\n" + "=" * 70)
    print("DATASET ENHANCEMENT COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    enhance_dataset_with_seasonal_data()
