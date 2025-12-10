import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================
# CONFIG
# ============================================================

N_MACHINES = 50
N_DAYS = 60
FREQ = "10min"  # time resolution

np.random.seed(42)

# ============================================================
# BUILD GLOBAL TIME INDEX
# ============================================================

start_date = datetime(2024, 1, 1)
end_date = start_date + timedelta(days=N_DAYS)
time_index = pd.date_range(start=start_date, end=end_date, freq=FREQ)
N_STEPS = len(time_index)

print(f"Simulating {N_MACHINES} machines, {N_STEPS} time steps each.")


# ============================================================
# SIMULATE ONE MACHINE
# ============================================================

def simulate_machine(machine_id: int) -> pd.DataFrame:
    """
    Simulate sensor data and a degradation-based health_score
    for a single machine over the global time_index.
    """

    # Baseline ranges
    base_temp = np.random.uniform(45, 55)          # °C
    base_vibration = np.random.uniform(0.8, 1.2)   # m/s²
    base_pressure = np.random.uniform(260, 320)    # psi
    base_load = np.random.uniform(50, 70)          # %
    base_rpm = np.random.uniform(1500, 1900)       # rpm

    # Slow drifts to simulate wear
    temp_drift = np.linspace(0, np.random.uniform(8, 18), N_STEPS)
    vib_drift = np.linspace(0, np.random.uniform(0.3, 0.7), N_STEPS)

    # Random noise
    temp_noise = np.random.normal(0, 1.5, N_STEPS)
    vib_noise = np.random.normal(0, 0.2, N_STEPS)
    pressure_noise = np.random.normal(0, 5, N_STEPS)
    load_noise = np.random.normal(0, 5, N_STEPS)
    rpm_noise = np.random.normal(0, 40, N_STEPS)

    # Sensor signals
    temp = base_temp + temp_drift + temp_noise
    vibration = base_vibration + vib_drift + vib_noise
    pressure = base_pressure + pressure_noise
    load_pct = np.clip(base_load + load_noise, 10, 110)
    rpm = np.clip(base_rpm + rpm_noise, 800, 2200)

    # Health score: starts at 1.0, decreases over time due to wear + stress
    health_score = np.ones(N_STEPS)

    # Base wear rate (so every machine degrades even under mild stress)
    base_wear_rate = np.random.uniform(0.00015, 0.0003)  # per time step

    for t in range(1, N_STEPS):
        stress = 0.0

        # Temperature contribution
        if temp[t] > 65:
            stress += 0.005
        if temp[t] > 75:
            stress += 0.010
        if temp[t] > 85:
            stress += 0.015

        # Vibration contribution
        if vibration[t] > 1.8:
            stress += 0.007
        if vibration[t] > 2.2:
            stress += 0.012
        if vibration[t] > 2.6:
            stress += 0.018

        # Load contribution
        if load_pct[t] > 80:
            stress += 0.005
        if load_pct[t] > 90:
            stress += 0.010

        # Total wear: base + stress
        total_wear = base_wear_rate + stress

        # Decrease health; clamp to [0, 1]
        health_score[t] = max(0.0, health_score[t - 1] - total_wear)

    df = pd.DataFrame({
        "machine_id": machine_id,
        "timestamp": time_index,
        "temp_c": temp,
        "vibration_ms2": vibration,
        "pressure_psi": pressure,
        "load_pct": load_pct,
        "rpm": rpm,
        "health_score": health_score
    })

    return df


# ============================================================
# SIMULATE ALL MACHINES
# ============================================================

all_data = []

for m in range(N_MACHINES):
    df_m = simulate_machine(machine_id=m)
    all_data.append(df_m)

data = pd.concat(all_data, ignore_index=True)
data.sort_values(["machine_id", "timestamp"], inplace=True)
data.reset_index(drop=True, inplace=True)

print("Sample rows (before defining failures):")
print(data.head())

# ============================================================
# DEFINE FAILURE EVENTS FROM HEALTH_SCORE
# ============================================================

# Critical health threshold: when health_score drops below this
# for the first time in a period, we mark a failure event.
HEALTH_THRESHOLD = 0.3  # since health_score ∈ [0, 1]

# Flag low-health states
data["low_health"] = data["health_score"] < HEALTH_THRESHOLD

# For each machine, look at transitions from not-low → low
data["low_health_prev"] = (
    data.groupby("machine_id")["low_health"]
        .shift(1)
        .fillna(False)
)

# Failure event: crossing into low-health
data["failed"] = 0
data.loc[
    (data["low_health"] == True) & (data["low_health_prev"] == False),
    "failed"
] = 1

# Clean up helper columns
data.drop(columns=["low_health", "low_health_prev"], inplace=True)

print("\nFailure counts (event-level):")
print(data["failed"].value_counts())
print("\nFailures per machine:")
print(data.groupby("machine_id")["failed"].sum())

# ============================================================
# SAVE TO CSV
# ============================================================

output_path = "synthetic_sensor_data.csv"
data.to_csv(output_path, index=False)
print(f"\nSaved synthetic sensor data to: {output_path}")