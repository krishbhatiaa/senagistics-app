"""
SenaGistics - Autonomous agent orchestration layer.

Implements the Sense -> Reason -> Predict -> Recommend loop:
  1. Sense     : pull the latest consumption window for a depot.
  2. Predict   : forecast tomorrow's usage & estimate days-to-depletion.
  3. Reason    : check for anomalies, and classify any incoming message's intent.
  4. Recommend : decide one of four actions, and write the decision to the
                 tamper-evident audit log.

High-stakes actions (raising a procurement request) are RECOMMENDATIONS only
-- a human (logistics_command role) must approve them. This keeps a human in
the loop by design, not as an afterthought.
"""

from dataclasses import dataclass

import numpy as np

from models import WINDOW, predict_next_day, is_anomalous, classify_intent
from audit import AuditLog


@dataclass
class AgentDecision:
    depot: str
    predicted_tomorrow: float
    current_stock: float
    days_to_depletion: float
    anomaly_flagged: bool
    anomaly_deviation: float
    message: str
    intent: str
    intent_confidence: float
    recommendation: str
    rationale: str


ACTIONS = {
    "RAISE_PROCUREMENT": "Raise procurement request",
    "ESCALATE": "Escalate to duty officer for verification",
    "FORWARD": "Forward procurement request to logistics command",
    "NO_ACTION": "No action required",
}


class SenaGisticsAgent:
    def __init__(self, forecast_model, forecast_stats, intent_model, intent_tokenizer,
                 audit_log: AuditLog, depletion_threshold_days: float = 5.0):
        self.forecast_model = forecast_model
        self.forecast_stats = forecast_stats
        self.intent_model = intent_model
        self.intent_tokenizer = intent_tokenizer
        self.audit_log = audit_log
        self.depletion_threshold_days = depletion_threshold_days

    def run_cycle(self, depot_name: str, recent_series: np.ndarray, current_stock: float,
                  incoming_message: str, user: str = "system_agent", role: str = "duty_officer",
                  force_anomaly: bool = False) -> AgentDecision:
        assert len(recent_series) == WINDOW, f"recent_series must have exactly {WINDOW} days"

        # 1 & 2. Sense + Predict
        predicted_tomorrow = predict_next_day(self.forecast_model, self.forecast_stats, recent_series)
        avg_recent = float(recent_series.mean())
        days_to_depletion = current_stock / avg_recent if avg_recent > 0 else float("inf")

        # 3a. Anomaly check
        last_actual = recent_series[-1] if not force_anomaly else predicted_tomorrow * 2.8
        deviation = abs(float(last_actual) - predicted_tomorrow)
        anomaly_flagged = bool(is_anomalous(last_actual, predicted_tomorrow, self.forecast_stats))

        # 3b. Reason over incoming message
        intent, confidence = classify_intent(self.intent_model, self.intent_tokenizer, incoming_message)

        # 4. Recommend
        if days_to_depletion <= self.depletion_threshold_days or intent == "RESUPPLY_ALERT":
            action = "RAISE_PROCUREMENT"
            rationale = f"Stock projected critical within {days_to_depletion:.1f} days."
        elif anomaly_flagged or intent == "ANOMALY_REPORT":
            action = "ESCALATE"
            rationale = f"Consumption anomaly detected (deviation {deviation:.1f}L). Needs human verification."
        elif intent == "PROCUREMENT_REQUEST":
            action = "FORWARD"
            rationale = "Routine procurement request forwarded for approval."
        else:
            action = "NO_ACTION"
            rationale = "Status normal, no intervention required."

        decision = AgentDecision(
            depot=depot_name,
            predicted_tomorrow=round(predicted_tomorrow, 1),
            current_stock=current_stock,
            days_to_depletion=round(days_to_depletion, 1),
            anomaly_flagged=anomaly_flagged,
            anomaly_deviation=round(deviation, 1),
            message=incoming_message,
            intent=intent,
            intent_confidence=round(confidence, 3),
            recommendation=ACTIONS[action],
            rationale=rationale,
        )

        # Write to the tamper-evident audit log
        self.audit_log.append(
            role=role,
            user=user,
            action="log_consumption",
            payload={
                "depot": depot_name,
                "predicted_tomorrow": decision.predicted_tomorrow,
                "current_stock": decision.current_stock,
                "days_to_depletion": decision.days_to_depletion,
                "anomaly_flagged": decision.anomaly_flagged,
                "message": decision.message,
                "intent": decision.intent,
                "recommendation": decision.recommendation,
                "rationale": decision.rationale,
            },
        )

        return decision


if __name__ == "__main__":
    from models import load_forecast_model, load_intent_model
    from data_sim import simulate_all_depots

    fmodel, fstats = load_forecast_model()
    imodel, tok = load_intent_model()
    log = AuditLog()

    agent = SenaGisticsAgent(fmodel, fstats, imodel, tok, log)

    data = simulate_all_depots()
    alpha_series = data[data.depot == "Depot-Alpha"]["fuel_litres"].values.astype("float32")[-WINDOW:]

    scenarios = [
        ("Depot-Alpha", 8000, "how much fuel do we currently have at alpha", False),
        ("Depot-Alpha", 1500, "fuel stock will run out in three days at this rate", False),
        ("Depot-Alpha", 8000, "fuel readings look strange today, please check", True),
        ("Depot-Alpha", 8000, "please raise a purchase order for 5.56mm rounds", False),
    ]

    for depot, stock, msg, force_anom in scenarios:
        decision = agent.run_cycle(depot, alpha_series, stock, msg, force_anomaly=force_anom)
        print(f"\n--- {depot} | stock={stock}L | msg='{msg}' ---")
        print(f"  Predicted tomorrow: {decision.predicted_tomorrow} L | Days to depletion: {decision.days_to_depletion}")
        print(f"  Anomaly: {decision.anomaly_flagged} (dev {decision.anomaly_deviation}L) | Intent: {decision.intent} ({decision.intent_confidence})")
        print(f"  -> {decision.recommendation} :: {decision.rationale}")

    valid, _ = log.verify_chain()
    print(f"\nAudit chain valid after {len(scenarios)} cycles: {valid}")
