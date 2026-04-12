import pandas as pd
import random
from pydantic import BaseModel
from typing import Optional
from sklearn.ensemble import RandomForestClassifier


# ── Observation ───────────────────────────────────────────────────────────────
class InvoiceObservation(BaseModel):
    invoice_amount: float
    payment_delay: float
    avg_invoice_amount: float
    risk_score: float          # float 0-1, used by task graders


# ── Step result ───────────────────────────────────────────────────────────────
class StepResult(BaseModel):
    observation: InvoiceObservation
    reward: float
    done: bool
    score: float               # strictly (0, 1)  ← key fix


# ── Task definitions (mirrors tasks/*.yaml) ───────────────────────────────────
TASKS = [
    {
        "id": "low_risk_business_detection",
        "name": "Low Risk Business Detection",
        "difficulty": "easy",
        "risk_level": "Low",
        "target_risk_score": 0.1,
        "passing_score": 0.5,
    },
    {
        "id": "medium_risk_business_detection",
        "name": "Medium Risk Business Detection",
        "difficulty": "medium",
        "risk_level": "Medium",
        "target_risk_score": 0.5,
        "passing_score": 0.6,
    },
    {
        "id": "high_risk_business_detection",
        "name": "High Risk Business Detection",
        "difficulty": "hard",
        "risk_level": "High",
        "target_risk_score": 0.9,
        "passing_score": 0.7,
    },
]
TASK_MAP = {t["id"]: t for t in TASKS}


# ── Main environment ──────────────────────────────────────────────────────────
class InvoiceFraudEnv:
    def __init__(self):
        df = pd.read_csv("data/Invoice.csv")
        df.columns = df.columns.str.lower()
        df["invoice_date"] = pd.to_datetime(df["date"])
        df["payment_date"] = [
            d + pd.to_timedelta(random.randint(1, 15), unit="D")
            for d in df["invoice_date"]
        ]
        df["payment_delay"] = (df["payment_date"] - df["invoice_date"]).dt.days
        df["avg_invoice_amount"] = df["amount"]

        # Continuous fraud probability (0.05-0.95) instead of hard 0/1
        delay_norm = (df["payment_delay"] - df["payment_delay"].min()) / (
            df["payment_delay"].max() - df["payment_delay"].min() + 1e-9
        )
        amount_norm = (df["amount"] - df["amount"].min()) / (
            df["amount"].max() - df["amount"].min() + 1e-9
        )
        df["fraud_prob"] = (0.6 * delay_norm + 0.4 * amount_norm).clip(0.05, 0.95)
        df["fraud_flag"] = (df["fraud_prob"] > 0.5).astype(int)

        self.X = df[["amount", "payment_delay", "avg_invoice_amount"]].rename(
            columns={"amount": "invoice_amount"}
        )
        self.y = df["fraud_flag"]
        self.fraud_prob = df["fraud_prob"].values

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X, self.y)

        self.index = 0
        self._current_task_id: Optional[str] = None
        self._steps_taken = 0
        self._episode_scores: list = []
        self._done = False
        self.state = None
        self.true_label = None
        self.risk_score = 0.5

    def _make_observation(self) -> dict:
        obs = self.state.to_dict()
        obs["risk_score"] = round(float(self.risk_score), 4)
        return obs

    def _score_for_action(self, action: float) -> float:
        """Score strictly in (0.01, 0.99): proximity of action to true fraud prob."""
        true_prob = float(self.risk_score)
        raw = 1.0 - abs(float(action) - true_prob)
        return round(max(0.01, min(0.99, raw)), 4)

    def reset(self, task_id: Optional[str] = None) -> dict:
        if self.index >= len(self.X):
            self.index = 0

        self.state = self.X.iloc[self.index]
        self.true_label = int(self.y.iloc[self.index])
        self.risk_score = float(self.fraud_prob[self.index])
        self.index += 1

        self._steps_taken = 0
        self._episode_scores = []
        self._done = False

        if task_id and task_id in TASK_MAP:
            self._current_task_id = task_id
        else:
            self._current_task_id = random.choice(list(TASK_MAP.keys()))

        task = TASK_MAP[self._current_task_id]
        return {
            "observation": self._make_observation(),
            "task": self._current_task_id,
            "task_name": task["name"],
            "difficulty": task["difficulty"],
            "risk_level": task["risk_level"],
        }

    def step(self, action) -> StepResult:
        action = float(action)
        score = self._score_for_action(action)
        self._episode_scores.append(score)
        self._steps_taken += 1

        task = TASK_MAP.get(self._current_task_id, TASKS[0])
        delay = float(self.state["payment_delay"])
        diff = task["difficulty"]

        if diff == "easy":
            reward = round(max(0.01, min(0.99, score * 0.9 + 0.05)), 4)
        elif diff == "medium":
            reward = round(max(0.01, min(0.99, score * 0.85 + 0.07)), 4)
        else:
            bonus = min(0.1, delay / 150)
            reward = round(max(0.01, min(0.99, score * 0.8 + bonus + 0.05)), 4)

        # Move to next invoice
        random_index = random.randint(0, len(self.X) - 1)
        self.state = self.X.iloc[random_index]
        self.true_label = int(self.y.iloc[random_index])
        self.risk_score = float(self.fraud_prob[random_index])

        done = self._steps_taken >= 10 or random.random() < 0.1
        self._done = done

        return StepResult(
            observation=InvoiceObservation(**self._make_observation()),
            reward=reward,
            score=score,
            done=done,
        )

    def grade(self) -> dict:
        if not self._episode_scores:
            return {"score": 0.5, "graded": False}
        avg = sum(self._episode_scores) / len(self._episode_scores)
        final = round(max(0.01, min(0.99, avg)), 4)
        task = TASK_MAP.get(self._current_task_id, TASKS[0])
        return {
            "score": final,
            "passed": final >= task["passing_score"],
            "task": self._current_task_id,
            "steps": self._steps_taken,
            "graded": True,
        }

    def state_info(self) -> dict:
        return {
            "observation": self._make_observation() if self.state is not None else {},
            "task": self._current_task_id,
            "steps_taken": self._steps_taken,
            "done": self._done,
        }

    def tasks_list(self) -> list:
        return TASKS
