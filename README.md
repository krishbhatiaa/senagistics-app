# SenaGistics — Working Prototype

**TensorFlow Autonomous Agents for Secure, Real-Time Fuel and Ammunition Consumables Management**
Group No. 27 — Amisha Valecha (123), Krish Bhatia (127), Anish Dharnidhar (129), Viren Jagiasi (131)

This is a real, runnable implementation of every component described in the presentation —
not a mockup. Everything here actually executes: the models train, the agent decides, the
audit log encrypts and hash-chains, and the dashboard is live and interactive.

## What's actually implemented

| Component | File | What it does |
|---|---|---|
| Synthetic data | `data_sim.py` | Simulates 240 days of fuel/ammo consumption across 3 depots, with weekly patterns, trend, noise, and injected anomalies |
| Forecasting model | `models.py` | LSTM (`Input → LSTM(32) → Dense(16) → Dense(1)`) — predicts tomorrow's consumption, doubles as the anomaly detector |
| Micro-LLM (intent classifier) | `models.py` | Embedding+LSTM text classifier — reads a message, classifies it as STATUS_CHECK / PROCUREMENT_REQUEST / ANOMALY_REPORT / RESUPPLY_ALERT |
| Security & audit layer | `audit.py` | Real encryption (Fernet/AES), role-based access control (RBAC), and a hash-chained tamper-evident audit log |
| Autonomous agent | `agent.py` | Ties it all together: sense → predict → reason → recommend, writing every decision to the audit log |
| Live dashboard | `app.py` | Streamlit UI — depot charts, live forecast/anomaly view, a text box to test the intent classifier, and an audit trail viewer |

## Honest scope — what this is, and isn't

- **Data is synthetic.** Real defence consumption data isn't accessible to us, so `data_sim.py`
  generates realistic-shaped data with NumPy. Say this plainly if asked.
- **Models are small on purpose.** Both models together are under 20,000 parameters — this is
  intentional, matching the "small, offline-capable" design philosophy, not a limitation we're
  hiding.
- **The audit trail is a real hash-chained, encrypted log**, not just a claim — `audit.py`'s
  `verify_chain()` genuinely detects tampering. It's not blockchain in the distributed-ledger
  sense (no consensus, no multiple nodes) — it's a single-node tamper-evident log, which is an
  honest and correct thing to call it.
- **The agent is a custom Python orchestrator**, not built on LangGraph/LangChain. That's a
  deliberate, explainable choice for the prototype; LangGraph is the named path for scaling it.
- **RBAC is enforced for real** — try switching roles in the sidebar and attempting an action
  the role isn't permitted; it will visibly reject it.

## How to run it

### Option A — Google Colab
1. Upload this whole folder (or just `data_sim.py`, `models.py`, `audit.py`, `agent.py`, `app.py`,
   `requirements.txt`) to a Colab session, or clone from GitHub if you push this there.
2. `!pip install -r requirements.txt`
3. Models/data are already trained and saved in `saved_models/` and `depot_consumption.csv` —
   you don't need to retrain. If you want to regenerate them: `!python models.py`
4. Colab can't run Streamlit directly in-browser easily; for Colab, run the model/agent scripts
   directly instead (see Option C below) — or use `localtunnel`/`ngrok` to expose the Streamlit
   app if you want the full dashboard from Colab.

### Option B — Locally / on your own machine (recommended for the live dashboard)

On Windows, use the same Python version for installing dependencies and launching
Streamlit. Python 3.13 is recommended; Python 3.12 is also supported.

```bash
py -3.13 -m pip install -r requirements.txt
py -3.13 -m streamlit run app.py
```

For Python 3.12, replace `-3.13` with `-3.12` in both commands. The requirements
select a compatible TensorFlow version for each Python version. If TensorFlow
still fails to import, reinstall it in that same interpreter with
`py -3.12 -m pip install --force-reinstall "tensorflow==2.16.2"` (or use
`py -3.13` and `tensorflow==2.21.0`).
This opens the full interactive dashboard in your browser at `http://localhost:8501`.

### Option C — Just run the pipeline without the UI (works anywhere, including Colab)
```bash
python models.py   # trains + saves both models (only needed once)
python agent.py     # runs 4 demo scenarios through the full agent loop
python audit.py     # demonstrates the encrypted, hash-chained audit log + RBAC
```

## Demo script for the viva

1. Open the dashboard (`streamlit run app.py`).
2. **Depot Overview tab** — show the consumption charts across depots.
3. **Forecast & Anomaly tab** — show the live prediction, then tick "Force an anomaly" in the
   sidebar and show the status flip to "FLAGGED".
4. **Agent & Intent Classifier tab** — type different messages (a status question, then
   "fuel stock will run out in three days") and click "Run agent cycle" — watch the
   recommendation change live, in front of the panel.
5. **Audit Trail tab** — show the hash-chained log filling up with every action you just took,
   and point out the chain-integrity check at the top.
6. Switch the role in the sidebar to `depot_clerk` and try to trigger an action that role can't
   do, to show RBAC actually blocking it.

## File map
```
senagistics_app/
├── data_sim.py           # synthetic depot data generator
├── models.py             # LSTM forecaster + intent classifier (train/save/load)
├── audit.py              # encrypted, hash-chained, RBAC-enforced audit log
├── agent.py              # the autonomous sense→reason→predict→recommend loop
├── app.py                # Streamlit dashboard (run this for the live demo)
├── requirements.txt
├── saved_models/          # pre-trained model weights (forecast + intent)
└── depot_consumption.csv  # pre-generated synthetic dataset
```
