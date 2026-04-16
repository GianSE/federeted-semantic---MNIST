import json
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from app.core.classifier_utils import SimpleClassifier, predict_topk
from app.core.image_utils import (
    DATASET_META,
    apply_awgn_noise,
    apply_random_pixel_mask,
    dequantize_latent,
    get_latent_bytes,
    get_original_bytes,
    load_dataset,
    quantize_latent,
)
from app.core.model_utils import get_model
from app.core.config import LOGS_DIR, RESULTADOS_ROOT


class ClassifierOrchestrator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._latest_experiment_id: str | None = None
        self._progress = {"epoch": 0, "total_epochs": 0}

    def start(self, payload: dict) -> dict:
        with self._state_lock:
            if self._running:
                return {"status": "already_running"}
            self._running = True
            self._stop_event.clear()

        if self._lock.locked():
            return {"status": "already_running"}

        thread = threading.Thread(
            target=self._run_training,
            args=(payload,),
            daemon=True,
        )
        thread.start()
        return {"status": "started"}

    def status(self) -> dict:
        with self._state_lock:
            return {
                "running": self._running,
                "latest_experiment_id": self._latest_experiment_id,
                "progress": self._progress,
            }

    def stop(self) -> dict:
        with self._state_lock:
            if not self._running:
                return {"status": "not_running"}
            self._stop_event.set()
        self._emit("[stop] stop requested by user")
        return {"status": "stop_requested"}

    def clear_logs(self) -> dict:
        log_file = LOGS_DIR / "classifier_trainer.log"
        log_file.write_text("", encoding="utf-8")
        return {"status": "logs_cleared"}

    def stream(self):
        log_file = LOGS_DIR / "classifier_trainer.log"
        if not log_file.exists():
            log_file.write_text("", encoding="utf-8")

        with open(log_file, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    if not self._running:
                        time.sleep(1)
                        yield "[heartbeat] waiting for new logs..."
                        continue
                    time.sleep(0.5)
                    continue
                yield line.strip()

    def list_experiments(self) -> list[dict]:
        items = []
        root = RESULTADOS_ROOT / "classifier"
        if not root.exists():
            return items
        for exp_dir in sorted(root.glob("experimento_*"), reverse=True):
            summary_file = exp_dir / "metrics" / "final_summary.json"
            if not summary_file.exists():
                continue
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            items.append(
                {
                    "id": exp_dir.name,
                    "dataset": summary.get("dataset"),
                    "epochs": summary.get("epochs"),
                    "timestamp": summary.get("timestamp"),
                    "final_accuracy": summary.get("final_accuracy"),
                }
            )
        return items

    def latest_experiment(self) -> dict | None:
        items = self.list_experiments()
        if not items:
            return None
        return self.get_experiment(items[0]["id"])

    def get_experiment(self, experiment_id: str) -> dict | None:
        exp_dir = (RESULTADOS_ROOT / "classifier" / experiment_id)
        summary_file = exp_dir / "metrics" / "final_summary.json"
        history_file = exp_dir / "metrics" / "history.csv"
        eval_file = exp_dir / "metrics" / "evaluation.json"
        if not summary_file.exists():
            return None

        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if history_file.exists():
            lines = [line for line in history_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            header = lines[0].split(",") if lines else []
            history = []
            for line in lines[1:]:
                cols = line.split(",")
                row = {header[i]: cols[i] for i in range(min(len(header), len(cols)))}
                history.append(
                    {
                        "epoch": int(row.get("epoch", 0)),
                        "loss": float(row.get("loss", 0.0)),
                        "train_accuracy": float(row.get("train_accuracy", 0.0)),
                        "test_accuracy": float(row.get("test_accuracy", 0.0)),
                    }
                )
            summary["history"] = history
        if eval_file.exists():
            summary["evaluation"] = json.loads(eval_file.read_text(encoding="utf-8"))

        summary["figures"] = {
            "training_loss": f"/classifier/results/artifact/{experiment_id}/figures/training_loss.png",
            "training_accuracy": f"/classifier/results/artifact/{experiment_id}/figures/training_accuracy.png",
            "semantic": f"/classifier/results/artifact/{experiment_id}/figures/semantic_comparison.png",
            "snr": f"/classifier/results/artifact/{experiment_id}/figures/robustness_snr.png",
            "masking": f"/classifier/results/artifact/{experiment_id}/figures/robustness_masking.png",
        }
        return summary

    def artifact_path(self, experiment_id: str, relative_path: str) -> Path | None:
        base = (RESULTADOS_ROOT / "classifier" / experiment_id).resolve()
        candidate = (base / relative_path).resolve()
        if not str(candidate).startswith(str(base)):
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    def _emit(self, message: str) -> None:
        log_file = LOGS_DIR / "classifier_trainer.log"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    def _new_experiment_dir(self) -> tuple[str, Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"experimento_{timestamp}"
        root = RESULTADOS_ROOT / "classifier"
        experiment_dir = root / experiment_id
        for sub in ["config", "logs", "metrics", "figures", "tables", "models"]:
            (experiment_dir / sub).mkdir(parents=True, exist_ok=True)
        return experiment_id, experiment_dir

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        header = list(rows[0].keys())
        lines = [",".join(header)]
        for row in rows:
            lines.append(",".join(str(row.get(col, "")) for col in header))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _load_semantic_model(self, dataset: str, model_type: str, base_weights: str | None) -> tuple[torch.nn.Module, bool, str | None]:
        meta = DATASET_META.get(dataset)
        channels = meta["channels"]
        img_size = meta["height"]
        model = get_model(model_type, input_channels=channels, image_size=img_size)

        weights_dir = Path("/ml-data/weights")
        archive_dir = weights_dir / "archive"
        prefix = f"{dataset}_{model_type}"
        latest_path = weights_dir / f"{prefix}.pth"
        core_path = Path(f"app/core/{prefix}.pth")

        selected_path = None
        source = None
        if base_weights is None or base_weights in {"", "latest"}:
            if latest_path.exists():
                selected_path = latest_path
                source = "latest"
        elif base_weights in {"random", "none"}:
            selected_path = None
        else:
            safe_name = Path(base_weights).name
            candidate = weights_dir / safe_name
            archive_candidate = archive_dir / safe_name
            if candidate.exists():
                selected_path = candidate
                source = f"weights/{safe_name}"
            elif archive_candidate.exists():
                selected_path = archive_candidate
                source = f"archive/{safe_name}"

        if selected_path is None and core_path.exists():
            selected_path = core_path
            source = "core"

        weights_loaded = False
        if selected_path and selected_path.exists():
            model.load_state_dict(torch.load(selected_path, map_location="cpu", weights_only=True))
            weights_loaded = True
        model.eval()
        return model, weights_loaded, source

    def _classify_batch(
        self,
        classifier: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        top_k: int,
        min_confidence: float,
    ) -> dict:
        device = next(classifier.parameters()).device
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            top_indices, top_probs = predict_topk(classifier, images, top_k=top_k)
        top1 = top_indices[:, 0]
        top1_conf = top_probs[:, 0]
        correct_top1 = (top1 == labels)
        in_topk = top_indices.eq(labels.unsqueeze(1)).any(dim=1)
        recognized = (in_topk if top_k > 1 else correct_top1) & (top1_conf >= min_confidence)
        return {
            "recognized": recognized,
            "correct_top1": correct_top1,
        }

    def _evaluate_semantic(
        self,
        classifier: nn.Module,
        semantic_model: torch.nn.Module,
        dataset_name: str,
        bits: int,
        top_k: int,
        min_confidence: float,
        num_samples: int,
        snr_grid: list[float],
        masking_grid: list[float],
    ) -> dict:
        dataset_obj = load_dataset(dataset_name, train=False)
        indices = torch.randperm(len(dataset_obj))[:num_samples]
        classifier.eval()
        semantic_model.eval()

        def run_eval(awgn_snr: float | None, mask_drop: float | None) -> dict:
            hits_orig = 0
            hits_recv = 0
            hits_recon = 0
            hits_both = 0
            for idx in indices:
                original, label = dataset_obj[int(idx)]
                original = original.unsqueeze(0)
                label_tensor = torch.tensor([int(label)])

                received = original.clone()
                if mask_drop is not None:
                    received = apply_random_pixel_mask(received, mask_drop, 0.0)
                if awgn_snr is not None:
                    received = apply_awgn_noise(received, awgn_snr)

                with torch.no_grad():
                    encoded = semantic_model.encode(received)
                    if isinstance(encoded, tuple):
                        encoded = encoded[0]
                    q_latent, scale = quantize_latent(encoded, bits=bits)
                    dq_latent = dequantize_latent(q_latent, scale)
                    reconstructed = semantic_model.decode(dq_latent)

                pred_orig = self._classify_batch(classifier, original, label_tensor, top_k, min_confidence)
                pred_recv = self._classify_batch(classifier, received, label_tensor, top_k, min_confidence)
                pred_recon = self._classify_batch(classifier, reconstructed, label_tensor, top_k, min_confidence)

                hits_orig += int(pred_orig["recognized"].item())
                hits_recv += int(pred_recv["recognized"].item())
                hits_recon += int(pred_recon["recognized"].item())
                hits_both += int(pred_orig["recognized"].item() and pred_recon["recognized"].item())

            denom = max(1, num_samples)
            return {
                "accuracy_original": round(hits_orig / denom, 4),
                "accuracy_received": round(hits_recv / denom, 4),
                "accuracy_reconstructed": round(hits_recon / denom, 4),
                "semantic_preservation_rate": round(hits_recon / denom, 4),
                "semantic_preservation_given_original": round(
                    hits_both / max(1, hits_orig), 4
                ),
            }

        baseline = run_eval(None, None)
        snr_results = []
        for snr in snr_grid:
            snr_results.append({"snr_db": snr, **run_eval(float(snr), None)})

        masking_results = []
        for drop in masking_grid:
            masking_results.append({"drop_rate": drop, **run_eval(None, float(drop))})

        return {
            "baseline": baseline,
            "snr_curve": snr_results,
            "masking_curve": masking_results,
        }

    def _save_figures(self, experiment_dir: Path, history: list[dict], evaluation: dict) -> None:
        epochs = [h["epoch"] for h in history]
        loss = [h["loss"] for h in history]
        train_acc = [h["train_accuracy"] for h in history]
        test_acc = [h["test_accuracy"] for h in history]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, loss, color="#ffd166", linewidth=2, label="loss")
        ax.set_title("Treino do Classificador - Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(experiment_dir / "figures" / "training_loss.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, train_acc, color="#00f6a2", linewidth=2, label="train")
        ax.plot(epochs, test_acc, color="#489dff", linewidth=2, label="test")
        ax.set_title("Treino do Classificador - Acuracia")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(experiment_dir / "figures" / "training_accuracy.png", dpi=140)
        plt.close(fig)

        baseline = evaluation.get("baseline", {})
        labels = ["Original", "Recebida", "Reconstruida"]
        values = [
            baseline.get("accuracy_original", 0.0),
            baseline.get("accuracy_received", 0.0),
            baseline.get("accuracy_reconstructed", 0.0),
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, values, color=["#00f6a2", "#ffd166", "#489dff"])
        ax.set_ylim(0, 1.0)
        ax.set_title("Comparativo Semantico")
        ax.set_ylabel("Accuracy")
        fig.tight_layout()
        fig.savefig(experiment_dir / "figures" / "semantic_comparison.png", dpi=140)
        plt.close(fig)

        snr_curve = evaluation.get("snr_curve", [])
        if snr_curve:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            snr_vals = [row["snr_db"] for row in snr_curve]
            acc_recv = [row["accuracy_received"] for row in snr_curve]
            acc_recon = [row["accuracy_reconstructed"] for row in snr_curve]
            ax.plot(snr_vals, acc_recv, color="#ffd166", label="Recebida")
            ax.plot(snr_vals, acc_recon, color="#489dff", label="Reconstruida")
            ax.set_title("Robustez vs SNR")
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(experiment_dir / "figures" / "robustness_snr.png", dpi=140)
            plt.close(fig)

        masking_curve = evaluation.get("masking_curve", [])
        if masking_curve:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            drop_vals = [row["drop_rate"] for row in masking_curve]
            acc_recv = [row["accuracy_received"] for row in masking_curve]
            acc_recon = [row["accuracy_reconstructed"] for row in masking_curve]
            ax.plot(drop_vals, acc_recv, color="#ffd166", label="Recebida")
            ax.plot(drop_vals, acc_recon, color="#489dff", label="Reconstruida")
            ax.set_title("Robustez vs Masking")
            ax.set_xlabel("Drop rate")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(experiment_dir / "figures" / "robustness_masking.png", dpi=140)
            plt.close(fig)

    def _run_training(self, payload: dict) -> None:
        with self._lock:
            dataset = payload.get("dataset", "mnist")
            epochs = int(payload.get("epochs", 5))
            batch_size = int(payload.get("batch", 128))
            learning_rate = float(payload.get("lr", 1e-3))
            seed = int(payload.get("seed", 42))
            top_k = int(payload.get("top_k", 1))
            min_confidence = float(payload.get("min_confidence", 0.5))
            eval_samples = int(payload.get("eval_samples", 200))
            snr_grid = payload.get("snr_grid") or [5, 10, 15, 20, 25]
            masking_grid = payload.get("masking_grid") or [0.1, 0.25, 0.4, 0.6]
            bits = int(payload.get("bits", 8))
            semantic_model_type = payload.get("semantic_model", "cnn_vae")
            semantic_weights = payload.get("semantic_weights")

            meta = DATASET_META.get(dataset)
            if meta is None:
                self._emit(f"[error] dataset invalido: {dataset}")
                with self._state_lock:
                    self._running = False
                return

            torch.manual_seed(seed)
            np.random.seed(seed)

            self._progress = {"epoch": 0, "total_epochs": epochs}
            experiment_id, experiment_dir = self._new_experiment_dir()
            self._latest_experiment_id = experiment_id

            self._emit("=================================================")
            self._emit("  CLASSIFICADOR - TREINO E AVALIACAO SEMANTICA")
            self._emit("=================================================")
            self._emit(f"[init] dataset={dataset} | epochs={epochs} | batch={batch_size}")
            self._emit(f"[eval] top_k={top_k} | min_conf={min_confidence}")

            self._write_json(
                experiment_dir / "config" / "input_config.json",
                {
                    "dataset": dataset,
                    "epochs": epochs,
                    "batch": batch_size,
                    "lr": learning_rate,
                    "seed": seed,
                    "top_k": top_k,
                    "min_confidence": min_confidence,
                    "eval_samples": eval_samples,
                    "snr_grid": snr_grid,
                    "masking_grid": masking_grid,
                    "bits": bits,
                    "semantic_model": semantic_model_type,
                    "semantic_weights": semantic_weights,
                },
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_ds = load_dataset(dataset, train=True)
            test_ds = load_dataset(dataset, train=False)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

            model = SimpleClassifier(
                input_channels=meta["channels"],
                image_size=meta["height"],
                num_classes=meta.get("classes", 10),
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            history = []
            for epoch in range(1, epochs + 1):
                if self._stop_event.is_set():
                    self._emit("[stopped] treino interrompido")
                    break

                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for data, labels in train_loader:
                    data = data.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    logits = model(data)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    correct += int((preds == labels).sum().item())
                    total += labels.size(0)

                train_acc = correct / max(1, total)
                avg_loss = running_loss / max(1, len(train_loader))

                model.eval()
                eval_correct = 0
                eval_total = 0
                with torch.no_grad():
                    for data, labels in test_loader:
                        data = data.to(device)
                        labels = labels.to(device)
                        logits = model(data)
                        preds = logits.argmax(dim=1)
                        eval_correct += int((preds == labels).sum().item())
                        eval_total += labels.size(0)
                eval_acc = eval_correct / max(1, eval_total)

                history.append(
                    {
                        "epoch": epoch,
                        "loss": round(avg_loss, 6),
                        "train_accuracy": round(train_acc, 4),
                        "test_accuracy": round(eval_acc, 4),
                    }
                )
                self._progress = {"epoch": epoch, "total_epochs": epochs}
                self._emit(
                    f"[epoch {epoch}/{epochs}] loss={avg_loss:.4f} train_acc={train_acc:.3f} test_acc={eval_acc:.3f}"
                )

            weights_dir = Path("/ml-data/weights")
            weights_dir.mkdir(parents=True, exist_ok=True)
            latest_path = weights_dir / f"{dataset}_classifier.pth"
            torch.save(model.state_dict(), latest_path)
            archive_path = experiment_dir / "models" / f"{dataset}_classifier_{experiment_id}.pth"
            torch.save(model.state_dict(), archive_path)

            model = model.cpu()
            semantic_model, semantic_loaded, semantic_source = self._load_semantic_model(
                dataset, semantic_model_type, semantic_weights
            )

            evaluation = self._evaluate_semantic(
                classifier=model,
                semantic_model=semantic_model,
                dataset_name=dataset,
                bits=bits,
                top_k=top_k,
                min_confidence=min_confidence,
                num_samples=eval_samples,
                snr_grid=snr_grid,
                masking_grid=masking_grid,
            )

            summary = {
                "experiment_id": experiment_id,
                "dataset": dataset,
                "epochs": epochs,
                "batch": batch_size,
                "lr": learning_rate,
                "seed": seed,
                "final_accuracy": history[-1]["test_accuracy"] if history else 0.0,
                "timestamp": int(time.time()),
                "classifier_weights": str(latest_path),
                "semantic_model": semantic_model_type,
                "semantic_weights_loaded": semantic_loaded,
                "semantic_weights_source": semantic_source,
                "original_bytes": get_original_bytes(dataset),
                "latent_bytes": get_latent_bytes(torch.zeros(1, 32), bits),
                "top_k": top_k,
                "min_confidence": min_confidence,
            }

            self._write_json(experiment_dir / "metrics" / "final_summary.json", summary)
            self._write_json(experiment_dir / "metrics" / "evaluation.json", evaluation)
            if history:
                self._write_csv(experiment_dir / "metrics" / "history.csv", history)

            self._save_figures(experiment_dir, history, evaluation)

            log_file = LOGS_DIR / "classifier_trainer.log"
            if log_file.exists():
                (experiment_dir / "logs" / "trainer.log").write_text(
                    log_file.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )

            self._emit(f"[done] experimento salvo em {experiment_dir}")

        with self._state_lock:
            self._running = False
            self._stop_event.clear()


classifier_orchestrator = ClassifierOrchestrator()
