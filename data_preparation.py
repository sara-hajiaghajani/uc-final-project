"""
KNMI Weather Data Preparation
Processes raw KNMI station data into unified hourly format for model training.
Combines Schiphol and Maastricht stations into single CSV with station labels.
"""

import pandas as pd
import os


def process_knmi_data():
    """Process raw KNMI weather station files into unified format."""

    stations = {
        'schiphol': 'data/Schiphol_240_2011-2020.txt',
        'maastricht': 'data/Maastricht_380_2011-2020.txt'
    }

    output_file = 'data/knmi_hourly.csv'
    all_data = []

    for station_name, input_file in stations.items():
        print(f"Processing {station_name}...")

        # Find where data starts (after metadata header)
        with open(input_file, 'r') as f:
            lines = f.readlines()

        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('# STN,YYYYMMDD'):
                data_start = i + 2
                break

        # Read raw data
        df = pd.read_csv(input_file, skiprows=data_start, header=None, skipinitialspace=True)

        # KNMI column names (standard format)
        columns = ['STN', 'YYYYMMDD', 'HH', 'DD', 'FH', 'FF', 'FX', 'T', 'T10N', 'TD',
                   'SQ', 'Q', 'DR', 'RH', 'P', 'VV', 'N', 'U', 'WW', 'IX', 'M', 'R', 'S', 'O', 'Y']
        df.columns = columns

        # Parse datetime (KNMI uses hour 1-24, convert to 0-23)
        df['HH'] = pd.to_numeric(df['HH'], errors='coerce').fillna(1).astype(int) - 1
        df['HH'] = df['HH'].clip(0, 23)
        df['YYYYMMDD'] = pd.to_numeric(df['YYYYMMDD'], errors='coerce').astype(int)
        df['datetime'] = pd.to_datetime(df['YYYYMMDD'].astype(str), format='%Y%m%d') + \
                         pd.to_timedelta(df['HH'], unit='h')

        # Convert KNMI units to standard units
        # T: temperature in 0.1 degrees Celsius -> degrees Celsius
        # FH: hourly mean wind speed in 0.1 m/s -> m/s
        # U: relative humidity in percentage (already correct)
        # P: air pressure in 0.1 hPa -> hPa
        df['temperature'] = pd.to_numeric(df['T'], errors='coerce') / 10.0
        df['wind_speed'] = pd.to_numeric(df['FH'], errors='coerce') / 10.0
        df['humidity'] = pd.to_numeric(df['U'], errors='coerce')
        df['pressure'] = pd.to_numeric(df['P'], errors='coerce') / 10.0

        # Add station identifier
        df['station'] = station_name

        # Keep only relevant columns and remove incomplete records
        df = df[['datetime', 'station', 'temperature', 'wind_speed', 'humidity', 'pressure']]
        df = df.dropna()

        all_data.append(df)
        print(f"  Loaded {len(df):,} records ({df['datetime'].min()} to {df['datetime'].max()})")

    # Combine both stations
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(['datetime', 'station']).reset_index(drop=True)

    # Save to CSV
    os.makedirs('data', exist_ok=True)
    combined_df.to_csv(output_file, index=False)

    # Summary
    print(f"\nCombined {len(combined_df):,} records from {len(stations)} stations")
    print(f"Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
    print(f"Saved to: {output_file}")
    print("\nRecords per station:")
    for station, count in combined_df['station'].value_counts().items():
        print(f"  {station}: {count:,}")


if __name__ == "__main__":
    process_knmi_data()