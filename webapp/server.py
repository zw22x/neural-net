from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
import types
from urllib.parse import urlparse

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
STATIC_ROOT = ROOT / "static"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_moons_fallback(n_samples=100, noise=0.1, random_state=None):
    rng = np.random.default_rng(random_state)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    outer_theta = np.linspace(0, math.pi, n_outer)
    inner_theta = np.linspace(0, math.pi, n_inner)

    outer = np.column_stack([np.cos(outer_theta), np.sin(outer_theta)])
    inner = np.column_stack([1 - np.cos(inner_theta), 0.5 - np.sin(inner_theta)])

    X = np.vstack([outer, inner])
    y = np.concatenate([np.zeros(n_outer), np.ones(n_inner)])

    if noise:
        X = X + rng.normal(0, noise, X.shape)

    order = rng.permutation(n_samples)
    return X[order], y[order]


try:
    from sklearn.datasets import make_moons as _make_moons
except ModuleNotFoundError:
    sklearn_module = types.ModuleType("sklearn")
    datasets_module = types.ModuleType("sklearn.datasets")
    datasets_module.make_moons = make_moons_fallback
    sklearn_module.datasets = datasets_module
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.datasets"] = datasets_module
    _make_moons = make_moons_fallback

import neuralNet as nn


TESTS = {
    "and": {
        "title": "AND gate",
        "summary": "Only the top-right point should light up.",
        "epochs": 2500,
        "learning_rate": 0.12,
        "seed": 12,
        "init_scale": 0.1,
    },
    "or": {
        "title": "OR gate",
        "summary": "Every point except zero-zero should light up.",
        "epochs": 2200,
        "learning_rate": 0.12,
        "seed": 14,
        "init_scale": 0.1,
    },
    "nand": {
        "title": "NAND gate",
        "summary": "The opposite of AND: only the top-right point stays dark.",
        "epochs": 2600,
        "learning_rate": 0.12,
        "seed": 16,
        "init_scale": 0.1,
    },
    "nor": {
        "title": "NOR gate",
        "summary": "The opposite of OR: only zero-zero lights up.",
        "epochs": 2600,
        "learning_rate": 0.12,
        "seed": 18,
        "init_scale": 0.1,
    },
    "xor": {
        "title": "XOR gate",
        "summary": "The diagonal corners disagree, so the hidden layer has to help.",
        "epochs": 7000,
        "learning_rate": 0.1,
        "seed": 3,
        "init_scale": 0.1,
    },
    "noisy_xor": {
        "title": "Noisy XOR",
        "summary": "Four little clouds make the hidden layer solve XOR more visually.",
        "epochs": 9000,
        "learning_rate": 0.1,
        "seed": 23,
        "init_scale": 0.3,
    },
    "diagonal": {
        "title": "Diagonal split",
        "summary": "A simple line separates the two classes.",
        "epochs": 3000,
        "learning_rate": 0.1,
        "seed": 31,
        "init_scale": 0.2,
    },
    "circle": {
        "title": "Circle inside/outside",
        "summary": "The network learns to draw a soft bubble around the center.",
        "epochs": 12000,
        "learning_rate": 0.1,
        "seed": 37,
        "init_scale": 0.7,
    },
    "moons": {
        "title": "Two moons",
        "summary": "Two curved classes show the decision boundary bending.",
        "epochs": 10000,
        "learning_rate": 0.1,
        "seed": 42,
        "init_scale": 0.5,
    },
}


def reset_network(seed, scale):
    rng = np.random.default_rng(seed)
    nn.W1 = rng.normal(0, scale, (nn.input_size, nn.hidden_size))
    nn.b1 = np.zeros((1, nn.hidden_size))
    nn.W2 = rng.normal(0, scale, (nn.hidden_size, nn.output_size))
    nn.b2 = np.zeros((1, nn.output_size))


def make_noisy_xor(seed):
    rng = np.random.default_rng(seed)
    centers = [
        ((0.0, 0.0), 0),
        ((0.0, 1.0), 1),
        ((1.0, 0.0), 1),
        ((1.0, 1.0), 0),
    ]
    X_parts = []
    y_parts = []
    for center, label in centers:
        points = rng.normal(center, 0.08, (28, 2))
        X_parts.append(points)
        y_parts.append(np.full((points.shape[0], 1), label, dtype=float))
    X = np.vstack(X_parts)
    y = np.vstack(y_parts)
    order = rng.permutation(X.shape[0])
    return X[order], y[order]


def make_diagonal_split(seed):
    rng = np.random.default_rng(seed)
    points = []
    labels = []
    while len(points) < 160:
        point = rng.uniform(-1.0, 1.0, 2)
        margin = point[0] + point[1]
        if abs(margin) < 0.18:
            continue
        points.append(point)
        labels.append(1 if margin > 0 else 0)
    return np.array(points, dtype=float), np.array(labels, dtype=float).reshape(-1, 1)


def make_circle_split(seed):
    rng = np.random.default_rng(seed)
    points = []
    labels = []

    for _ in range(80):
        radius = rng.uniform(0.04, 0.42)
        angle = rng.uniform(0, math.tau)
        points.append([math.cos(angle) * radius, math.sin(angle) * radius])
        labels.append(1)

    for _ in range(120):
        radius = rng.uniform(0.72, 1.12)
        angle = rng.uniform(0, math.tau)
        points.append([math.cos(angle) * radius, math.sin(angle) * radius])
        labels.append(0)

    X = np.array(points, dtype=float)
    y = np.array(labels, dtype=float).reshape(-1, 1)
    order = rng.permutation(X.shape[0])
    return X[order], y[order]


def make_dataset(test_name, seed):
    if test_name == "and":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [0], [0], [1]], dtype=float)
        return X, y

    if test_name == "or":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [1]], dtype=float)
        return X, y

    if test_name == "nand":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[1], [1], [1], [0]], dtype=float)
        return X, y

    if test_name == "nor":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[1], [0], [0], [0]], dtype=float)
        return X, y

    if test_name == "xor":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)
        return X, y

    if test_name == "noisy_xor":
        return make_noisy_xor(seed)

    if test_name == "diagonal":
        return make_diagonal_split(seed)

    if test_name == "circle":
        return make_circle_split(seed)

    if test_name == "moons":
        X, y = _make_moons(n_samples=160, noise=0.1, random_state=seed)
        return X.astype(float), y.reshape(-1, 1).astype(float)

    raise ValueError(f"Unknown test: {test_name}")


def accuracy_for(y_true, y_pred):
    predicted = (y_pred >= 0.5).astype(float)
    return float(np.mean(predicted == y_true))


def bounds_for(X):
    x_min, y_min = np.min(X, axis=0)
    x_max, y_max = np.max(X, axis=0)
    x_pad = max((x_max - x_min) * 0.2, 0.35)
    y_pad = max((y_max - y_min) * 0.2, 0.35)
    return {
        "xMin": float(x_min - x_pad),
        "xMax": float(x_max + x_pad),
        "yMin": float(y_min - y_pad),
        "yMax": float(y_max + y_pad),
    }


def decision_grid(bounds, steps=42):
    xs = np.linspace(bounds["xMin"], bounds["xMax"], steps)
    ys = np.linspace(bounds["yMin"], bounds["yMax"], steps)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=float)
    predictions = nn.forward(grid)[1].reshape(-1)
    return [
        {"x": float(point[0]), "y": float(point[1]), "prediction": float(pred)}
        for point, pred in zip(grid, predictions)
    ]


def network_snapshot():
    return {
        "layers": [int(nn.input_size), int(nn.hidden_size), int(nn.output_size)],
        "W1": nn.W1.round(4).tolist(),
        "W2": nn.W2.round(4).tolist(),
    }


def train_test(test_name, epochs=None, learning_rate=None):
    if test_name not in TESTS:
        raise ValueError(f"Unknown test: {test_name}")

    config = TESTS[test_name]
    seed = int(config["seed"])
    epochs = int(epochs if epochs is not None else config["epochs"])
    learning_rate = float(
        learning_rate if learning_rate is not None else config["learning_rate"]
    )
    epochs = max(1, min(20000, epochs))
    learning_rate = max(0.001, min(1.0, learning_rate))

    reset_network(seed, float(config["init_scale"]))
    X, y = make_dataset(test_name, seed)
    sample_every = max(1, epochs // 80)
    history = []

    for epoch in range(epochs + 1):
        A1, A2, Z1, Z2 = nn.forward(X)
        loss = float(nn.mse_loss(y, A2))

        if epoch % sample_every == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "accuracy": accuracy_for(y, A2),
                }
            )

        if epoch == epochs:
            break

        gradients = nn.backward(X, y, A1, A2, Z1, Z2)
        nn.update_params(*gradients, learning_rate)

    final_predictions = nn.forward(X)[1]
    bounds = bounds_for(X)
    points = [
        {
            "x": float(row[0]),
            "y": float(row[1]),
            "label": int(label[0]),
            "prediction": float(pred[0]),
            "correct": bool((pred[0] >= 0.5) == bool(label[0])),
        }
        for row, label, pred in zip(X, y, final_predictions)
    ]
    final_loss = float(nn.mse_loss(y, final_predictions))
    final_accuracy = accuracy_for(y, final_predictions)

    return {
        "id": test_name,
        "title": config["title"],
        "summary": config["summary"],
        "epochs": epochs,
        "learningRate": learning_rate,
        "finalLoss": final_loss,
        "accuracy": final_accuracy,
        "passed": final_accuracy >= 0.95 and final_loss < 0.08,
        "points": points,
        "bounds": bounds,
        "decision": decision_grid(bounds),
        "history": history,
        "network": network_snapshot(),
    }


class AppHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/tests":
            tests = [
                {
                    "id": key,
                    "title": value["title"],
                    "summary": value["summary"],
                    "epochs": value["epochs"],
                    "learningRate": value["learning_rate"],
                }
                for key, value in TESTS.items()
            ]
            self.send_json({"tests": tests})
            return

        if parsed.path in {"/", "/index.html"}:
            self.path = "/static/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body or "{}")
            result = train_test(
                payload.get("test", "xor"),
                payload.get("epochs"),
                payload.get("learningRate"),
            )
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def translate_path(self, path):
        parsed = urlparse(path)
        clean_path = parsed.path.lstrip("/")
        if clean_path.startswith("static/"):
            return str(ROOT / clean_path)
        return str(STATIC_ROOT / "index.html")

    def send_json(self, payload, status=200):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    print(f"Neural net visualizer running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
