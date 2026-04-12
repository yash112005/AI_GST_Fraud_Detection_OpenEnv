import os
import requests
from typing import List, Optional
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860")

MAX_STEPS = 10

TASKS = [
    "low_risk_business_detection",
    "medium_risk_business_detection",
    "high_risk_business_detection",
]

client = None
try:
    if API_BASE_URL and API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
except Exception as e:
    print(f"[DEBUG] Client init failed: {e}", flush=True)


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def choose_action(observation: dict, task_id: str) -> float:
    """
    Return a float strictly in (0.01, 0.99) representing predicted fraud risk.
    Uses LLM when available, otherwise falls back to heuristic.
    Score is graded as proximity to true fraud probability, so we want accuracy.
    """
    delay = float(observation.get("payment_delay", 0))
    amount = float(observation.get("invoice_amount", 0))
    avg = float(observation.get("avg_invoice_amount", 1))

    # Heuristic: compute a continuous fraud risk score
    delay_risk = min(delay / 15.0, 1.0)           # normalised 0-1
    amount_risk = min(amount / (avg * 2 + 1), 1.0) # normalised 0-1
    heuristic = round(0.6 * delay_risk + 0.4 * amount_risk, 4)
    heuristic = max(0.05, min(0.95, heuristic))   # keep in safe range

    # Try LLM for a better estimate
    if client:
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "system",
                    "content": (
                        "You are a GST fraud risk scoring assistant. "
                        "Return ONLY a single decimal number between 0.05 and 0.95 "
                        "representing fraud probability. No other text."
                    ),
                }, {
                    "role": "user",
                    "content": (
                        f"Task: {task_id}\n"
                        f"payment_delay={delay} days, "
                        f"invoice_amount={amount}, "
                        f"avg_invoice_amount={avg}.\n"
                        "Fraud risk score (0.05-0.95):"
                    ),
                }],
                temperature=0,
                max_tokens=10,
            )
            raw = resp.choices[0].message.content.strip()
            llm_score = float(raw)
            llm_score = max(0.05, min(0.95, llm_score))
            print(f"[DEBUG] LLM score: {llm_score}", flush=True)
            return round(llm_score, 4)
        except Exception as e:
            print(f"[DEBUG] LLM error: {e} — using heuristic", flush=True)

    return heuristic


def run_task(task_id: str) -> float:
    """Run one full episode for a task. Returns final grade score."""
    log_start(task=task_id, env=ENV_URL, model=MODEL_NAME or "heuristic")

    rewards: List[float] = []
    steps_taken = 0
    score = 0.5

    try:
        resp = requests.post(
            f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=10
        ).json()
        observation = resp.get("observation", {})
        current_task = resp.get("task", task_id)

        for step in range(1, MAX_STEPS + 1):
            action = choose_action(observation, current_task)

            result = requests.post(
                f"{ENV_URL}/step", json={"action": action}, timeout=10
            ).json()

            reward = float(result.get("reward", 0.5))
            done = bool(result.get("done", False))
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=str(action), reward=reward, done=done, error=None)

            observation = result.get("observation", observation)
            if done:
                break

        # Get official grade from /grade endpoint
        grade_resp = requests.get(f"{ENV_URL}/grade", timeout=10).json()
        score = float(grade_resp.get("score", 0.5))

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
        score = sum(rewards) / len(rewards) if rewards else 0.5

    success = score >= 0.5
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main():
    """Run all 3 tasks (required: at least 3 tasks with graders)."""
    all_scores = []
    for task_id in TASKS:
        print(f"\n{'='*50}", flush=True)
        print(f"[TASK] {task_id}", flush=True)
        s = run_task(task_id)
        all_scores.append(s)
        print(f"[TASK SCORE] {task_id} => {s:.4f}", flush=True)

    overall = sum(all_scores) / len(all_scores)
    print(f"\n[FINAL] overall_score={overall:.4f}", flush=True)


if __name__ == "__main__":
    main()
