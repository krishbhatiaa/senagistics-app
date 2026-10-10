"""
SenaGistics - Synthetic depot consumption data simulator.

Generates realistic daily fuel & ammunition consumption logs for multiple
defence depots: weekly usage cycles, a slow upward trend, random noise,
and a handful of injected anomalies (spikes) for testing anomaly detection.
"""

import numpy as np
import pandas as pd

DEPOTS = [
    ("Depot-Alpha", 500, 800, 1),
    ("Depot-Bravo", 650, 950, 2),
    ("Depot-Charlie", 420, 700, 3),
]


def simulate_depot(depot_name, days=240, base_fuel=500, base_ammo=800, seed=0, n_anomalies=4):
    rng = np.random.default_rng(seed)
    t = np.arange(days)

    weekly = 40 * np.sin(2 * np.pi * t / 7) + 40
    trend = 0.15 * t
    noise_fuel = rng.normal(0, 25, days)
    noise_ammo = rng.normal(0, 35, days)

    fuel = base_fuel + weekly + trend + noise_fuel
    ammo = base_ammo + 0.8 * weekly + trend * 1.3 + noise_ammo

    fuel = np.clip(fuel, 50, None)
    ammo = np.clip(ammo, 50, None)

    is_anomaly = np.zeros(days, dtype=int)
    if n_anomalies > 0:
        anomaly_days = rng.choice(days, size=n_anomalies, replace=False)
        for d in anomaly_days:
            fuel[d] *= rng.uniform(2.0, 3.0)
            ammo[d] *= rng.uniform(2.0, 3.0)
            is_anomaly[d] = 1

    df = pd.DataFrame({
        "day": t,
        "depot": depot_name,
        "fuel_litres": fuel.round(1),
        "ammo_rounds": ammo.round(0).astype(int),
        "is_anomaly": is_anomaly,
    })
    return df


def simulate_all_depots(days=240):
    frames = [
        simulate_depot(name, days=days, base_fuel=bf, base_ammo=ba, seed=s)
        for name, bf, ba, s in DEPOTS
    ]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    data = simulate_all_depots()
    data.to_csv("depot_consumption.csv", index=False)
    print(f"Simulated {len(data)} rows across {len(DEPOTS)} depots -> depot_consumption.csv")
    print(data.head())
