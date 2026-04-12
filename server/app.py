from flask import Flask, request, jsonify
from flask_cors import CORS
from env import InvoiceFraudEnv

app = Flask(__name__)
CORS(app)
env = InvoiceFraudEnv()


@app.route("/")
def home():
    return jsonify({
        "project": "AI GST Fraud Detection Environment",
        "status": "running",
        "endpoints": {
            "reset": "/reset [POST/GET]",
            "step":  "/step  [POST]",
            "state": "/state [GET]",
            "grade": "/grade [GET]",
            "tasks": "/tasks [GET]",
        },
    })


@app.route("/reset", methods=["GET", "POST"])
def reset():
    task_id = None
    if request.method == "POST" and request.is_json:
        task_id = request.json.get("task_id")
    elif request.method == "GET":
        task_id = request.args.get("task_id")
    return jsonify(env.reset(task_id=task_id))


@app.route("/step", methods=["POST"])
def step():
    body = request.get_json(force=True)
    action = body.get("action", 0.5)
    result = env.step(action)
    return jsonify(result.dict())


@app.route("/state", methods=["GET"])
def state():
    return jsonify(env.state_info())


@app.route("/grade", methods=["GET"])
def grade():
    """Required by OpenEnv grader. Returns score strictly in (0, 1)."""
    return jsonify(env.grade())


@app.route("/tasks", methods=["GET"])
def tasks():
    return jsonify(env.tasks_list())


def main():
    app.run(host="0.0.0.0", port=7860, debug=False)


if __name__ == "__main__":
    main()
