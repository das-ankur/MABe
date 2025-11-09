import os
import torch
from tqdm import tqdm
from typing import Dict, Any, Optional, Callable, Tuple, List
from torch.cuda.amp import autocast



def evaluate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    evaluator,
    device: torch.device,
    desc: str = "Eval",
    amp: bool = False,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    """Evaluation with padding mask support."""
    model.eval()
    losses = []
    logits_list, targets_list, masks_list = [], [], []

    torch.set_grad_enabled(False)
    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch in pbar:
        if isinstance(batch, dict):
            inputs, targets, mask = batch["bodyparts"], batch["actions"], batch["mask"]
        else:
            raise ValueError("Expected batch dict with keys: bodyparts, actions, mask")

        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)

        if amp:
            with autocast():
                logits = model(inputs, attn_mask=mask)
                loss_raw = loss_fn(logits, targets)
                masked_loss = (loss_raw * mask.unsqueeze(-1)).sum() / mask.sum()
        else:
            logits = model(inputs, attn_mask=mask)
            loss_raw = loss_fn(logits, targets)
            masked_loss = (loss_raw * mask.unsqueeze(-1)).sum() / mask.sum()

        losses.append(float(masked_loss.detach().cpu().item()))
        logits_list.append(logits.detach().cpu())
        targets_list.append(targets.detach().cpu())
        masks_list.append(mask.detach().cpu())

        pbar.set_postfix({"loss": f"{sum(losses)/len(losses):.4f}"})

    all_logits = torch.cat(logits_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    all_masks = torch.cat(masks_list, dim=0)  # (B, n)

    B, n, o = all_logits.shape
    mask_flat = all_masks.view(-1).bool()
    flat_logits = all_logits.view(B * n, o)[mask_flat]
    flat_targets = all_targets.view(B * n, o)[mask_flat]

    metrics = {
        "loss": float(sum(losses) / len(losses)) if losses else float("nan"),
        "accuracy": evaluator.accuracy(flat_logits, flat_targets),
        "f1_micro": evaluator.f1_score(flat_logits, flat_targets, average="micro"),
        "precision_micro": evaluator.precision(flat_logits, flat_targets, average="micro"),
        "recall_micro": evaluator.recall(flat_logits, flat_targets, average="micro"),
    }
    try:
        metrics["auroc_micro"] = evaluator.auroc(flat_logits, flat_targets, average="micro")
    except Exception:
        metrics["auroc_micro"] = float("nan")

    torch.set_grad_enabled(True)
    return metrics, flat_logits, flat_targets


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    evaluator,
    n_epochs: int = 10,
    grad_clip: Optional[float] = None,
    amp: bool = False,
    validate_every_n_epochs: int = 1,
    checkpoint_path: Optional[str] = None,
    checkpoint_monitor: str = "f1_micro",
    checkpoint_mode: str = "max",
) -> Dict[str, Any]:
    """Training loop with masking-aware loss and metrics."""

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"{device_name} is initialised to train the model.")

    model = model.to(device)
    if device_name == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    history = {"train": [], "val": []}
    best_monitor_val = None
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    epoch_pbar = tqdm(range(1, n_epochs + 1), desc="Epochs")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for epoch in epoch_pbar:
        model.train()
        train_losses = []
        logits_accum, targets_accum, masks_accum = [], [], []

        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
        for batch in train_pbar:
            if not isinstance(batch, dict):
                raise ValueError("Expected batch dict with keys: bodyparts, actions, mask")

            inputs, targets, mask = batch["bodyparts"].to(device), batch["actions"].to(device), batch["attention_mask"].to(device)

            optimizer.zero_grad()
            if amp:
                with autocast():
                    logits = model(inputs, attn_mask=mask)
                    valid_mask = mask.bool()
                    valid_logits = logits[valid_mask]
                    valid_targets = targets[valid_mask]
                    loss = loss_fn(valid_logits, valid_targets)
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs, attn_mask=mask)
                valid_mask = mask.bool()
                valid_logits = logits[valid_mask]
                valid_targets = targets[valid_mask]
                loss = loss_fn(valid_logits, valid_targets)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))
            logits_accum.append(logits.detach().cpu())
            targets_accum.append(targets.detach().cpu())
            masks_accum.append(mask.detach().cpu())

            train_pbar.set_postfix({
                "batch_loss": f"{train_losses[-1]:.4f}",
                "avg_loss": f"{sum(train_losses)/len(train_losses):.4f}"
            })

        # Epoch metrics (masked)
        all_logits = torch.cat(logits_accum, dim=0)
        all_targets = torch.cat(targets_accum, dim=0)
        all_masks = torch.cat(masks_accum, dim=0)
        B_t, n_t, o_t = all_logits.shape
        mask_flat = all_masks.view(-1).bool()
        flat_logits = all_logits.view(B_t * n_t, o_t)[mask_flat]
        flat_targets = all_targets.view(B_t * n_t, o_t)[mask_flat]

        train_metrics = {
            "loss": float(sum(train_losses)/len(train_losses)) if train_losses else float("nan"),
            "accuracy": evaluator.accuracy(flat_logits, flat_targets),
            "f1_micro": evaluator.f1_score(flat_logits, flat_targets, average="micro"),
            "precision_micro": evaluator.precision(flat_logits, flat_targets, average="micro"),
            "recall_micro": evaluator.recall(flat_logits, flat_targets, average="micro"),
        }
        try:
            train_metrics["auroc_micro"] = evaluator.auroc(flat_logits, flat_targets, average="micro")
        except Exception:
            train_metrics["auroc_micro"] = float("nan")

        history["train"].append(train_metrics)

        # Validation
        if (epoch % validate_every_n_epochs) == 0:
            val_metrics, _, _ = evaluate_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                evaluator=evaluator,
                device=device,
                desc=f"Val Epoch {epoch}",
                amp=amp,
            )
            history["val"].append(val_metrics)
            epoch_pbar.set_postfix({
                "train_loss": f"{train_metrics['loss']:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
                "val_f1": f"{val_metrics.get('f1_micro', float('nan')):.4f}"
            })

            # Checkpointing
            if checkpoint_path:
                monitor_value = val_metrics.get(checkpoint_monitor)
                if monitor_value is not None and not torch.isnan(torch.tensor(monitor_value)):
                    is_better = (
                        best_monitor_val is None or
                        (checkpoint_mode == "max" and monitor_value > best_monitor_val) or
                        (checkpoint_mode == "min" and monitor_value < best_monitor_val)
                    )
                    if is_better:
                        best_monitor_val = monitor_value
                        torch.save({
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "monitor": monitor_value,
                        }, checkpoint_path)
        else:
            epoch_pbar.set_postfix({"train_loss": f"{train_metrics['loss']:.4f}"})

    return history
