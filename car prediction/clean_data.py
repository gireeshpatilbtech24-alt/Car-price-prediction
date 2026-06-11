import os
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'templates', 'quikr_car.csv')

# Load CSV
df = pd.read_csv(DATA_PATH)

# Filter: keep only rows with sensible company names (exclude numbers, fragments, etc.)
valid_companies = {
    'Audi', 'BMW', 'Chevrolet', 'Datsun', 'Fiat', 'Force', 'Ford',
    'Hindustan', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Mahindra',
    'Maruti', 'Mercedes', 'Mini', 'Mitsubishi', 'Nissan', 'Renault',
    'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'
}

df = df[df['company'].isin(valid_companies)]

# Save cleaned CSV
df.to_csv(DATA_PATH, index=False)
print(f'Cleaned CSV: {len(df)} rows saved to {DATA_PATH}')
