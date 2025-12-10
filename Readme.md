# **README.md — Predictive Maintenance System (End-to-End ML Pipeline)**

## **Overview**

This project implements an **end-to-end Predictive Maintenance System** that forecasts machinery failures using synthetic industrial sensor data. The goal is to predict whether a machine will fail **within the next 24 hours**, enabling maintenance teams to reduce unplanned downtime, optimize maintenance scheduling, and improve equipment lifespan.

The system includes:

* Synthetic dataset generation
* Feature engineering (6h & 24h windows)
* Failure-horizon label creation
* Model training (Random Forest)
* Class-imbalance handling
* Performance evaluation
* FastAPI service for real-time inference

---

## **1. Dataset Generation**

### **Synthetic Data**

We simulate **50 machines** over **60 days** at **10-minute intervals**, generating realistic time-series sensor data:

* Temperature (°C)
* Vibration (m/s²)
* Pressure (psi)
* Load (%)
* RPM (rotational speed)
* Health score (0–1)

Each machine gradually **degrades** due to stress, noise, and long-term drift.

### **Failure Definition**

A machine is considered to **fail** when its `health_score` falls below a critical threshold.

We detect failure events by identifying transitions:

```
health_score >= threshold  →  health_score < threshold
```

Each transition represents a **failure event**.

Output of generator (`synthetic_sensor_data.csv`) contains:

| machine_id | timestamp | temp_c | vibration_ms2 | pressure_psi | load_pct | rpm | health_score | failed |
| ---------- | --------- | ------ | ------------- | ------------ | -------- | --- | ------------ | ------ |

---

## **2. Creating the Prediction Target (`fail_within_24h`)**

Predictive maintenance requires forecasting **future failure**, not detecting current failure.

We convert event labels into a prediction target:

```
fail_within_24h = 1  
if the machine will fail within the next 24 hours
```

This is done by:

* Finding the **next failure timestamp** for each row
* Calculating the **time difference**
* Assigning label **1** if failure occurs within 24 hours

This expands each failure event into many positive training examples, making model learning possible.

---

## **3. Feature Engineering**

Failures are rarely caused by a single sensor reading.
They emerge from **patterns over time**.

We compute **rolling window features**:

### **6-hour window (short-term behaviour)**

Captures recent instability:

* Sudden vibration spikes
* Temperature jumps
* Load fluctuations

### **24-hour window (long-term behaviour)**

Captures gradual degradation:

* Slow temperature rise
* Sustained stress
* Long-term vibration increase

For each window, we compute:

* Mean
* Standard deviation
* Max
* Delta (change over time)
* Count of high-temperature events

These features form the final model input.

---

## **4. Model Training**

### **Handling class imbalance**

Only about ~50 raw failure events exist, but `fail_within_24h` expands this to ~7,000+ positives.
We perform:

* **Stratified train-test split**
* **Undersampling** of the majority class (`0`)
* Controlled ratio (e.g., 5:1 negatives to positives)

### **Model Used**

A **Random Forest Classifier** is trained on:

* Rolling features
* Failure horizon label
* Undersampled training set

We also adjust the prediction threshold (`0.20`) to prioritize **recall**, because:

> Missing a failure is more costly than a false alarm.

---

## **5. Evaluation**

Evaluation includes:

* Precision, Recall, and F1-score
* Confusion matrix
* Probability outputs (`predict_proba`)
* Identification of high-risk windows

Example result:

* Model successfully identifies windows leading up to failures
* Reasonable recall on the failure class
* Good interpretability and stable prediction behaviour

---

## **6. FastAPI Inference Service**

The trained model is deployed via a **FastAPI service** that:

### **Input:**

A JSON payload containing engineered features:

```json
{
  "features": {
    "temp_c_mean_24h": 60.2,
    "vibration_ms2_mean_24h": 2.1,
    ...
  }
}
```

### **Output:**

Failure probability and maintenance recommendation:

```json
{
  "failure_probability_24h": 0.89,
  "recommendation": "High risk: schedule maintenance as soon as possible."
}
```

The API also supports:

* Low-risk predictions
* Moderate-risk predictions
* High-risk predictions extracted from the test set

---

## **7. End-to-End Pipeline Summary**

```
┌────────────────────────────────┐
│  Synthetic Sensor Generation   │
└──────────────┬─────────────────┘
               ↓
┌────────────────────────────────┐
│   Failure Event Detection      │
└──────────────┬─────────────────┘
               ↓
┌────────────────────────────────┐
│  Create fail_within_24h Label │
└──────────────┬─────────────────┘
               ↓
┌────────────────────────────────┐
│    Rolling Feature Engineering │
└──────────────┬─────────────────┘
               ↓
┌────────────────────────────────┐
│  Train Model (RF + imbalance) │
└──────────────┬─────────────────┘
               ↓
┌────────────────────────────────┐
│ FastAPI Real-Time Prediction  │
└────────────────────────────────┘
```

---

## **8. How to Run**

### **Generate dataset**

```bash
python generate_data.py
```

### **Train the model**

Run the training notebook or script:

```bash
python train_model.py
```

This outputs:

```
pd_24h_model.joblib
```

### **Run FastAPI**

```bash
uvicorn api:app --reload
```

Send POST requests to:

```
POST /predict_24h
```

---

## **9. Key Benefits of This System**

* Predicts failures **before** they occur
* Helps avoid unplanned downtime
* Uses explainable features (rolling windows)
* Reproduces real industrial patterns
* Demonstrates full ML + MLOps flow

---

## **10. Future Improvements**

* Multi-horizon prediction (24h / 48h / 72h)
* SHAP-based explainability