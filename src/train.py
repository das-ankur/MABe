import os
import torch
from tqdm import tqdm
from typing import Dict, Any, Optional, Callable, Tuple, List
from torch.cuda.amp import autocast



def evaluate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    evaluator,                  # instance of MultiLabelEvaluator
    device: torch.device,
    desc: str = "Eval",
    amp: bool = False,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    """
    Run evaluation over dataloader and return metrics, concatenated logits and targets.
    """
    model.eval()
    losses = []
    logits_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []

    torch.set_grad_enabled(False)
    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch in pbar:
        # Expect batch to be (inputs, targets) or dict-like; handle common cases
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        elif isinstance(batch, dict):
            inputs, targets = batch['bodyparts'], batch['actions']
        else:
            raise ValueError("Dataloader must return (inputs, targets) or a dict with 'bodyparts'/'actions'.")

        inputs = inputs.to(device)
        targets = targets.to(device)

        if amp:
            with autocast():
                logits = model(inputs)
                loss = loss_fn(logits, targets)
        else:
            logits = model(inputs)
            loss = loss_fn(logits, targets)

        losses.append(float(loss.detach().cpu().item()))
        logits_list.append(logits.detach().cpu())
        targets_list.append(targets.detach().cpu())

        pbar.set_postfix({"loss": f"{sum(losses)/len(losses):.4f}"})

    # concat across batches (shape: (N, n, n_outputs))
    all_logits = torch.cat(logits_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)

    # Flattening policy for evaluator: evaluator expects (B, n_outputs) logits/targets.
    # If model outputs (B, n, n_outputs), some metrics may be computed per-token label.
    # We'll flatten first two dims to treat every token as a sample: (B*n, n_outputs)
    B, n, o = all_logits.shape
    flat_logits = all_logits.view(B * n, o)
    flat_targets = all_targets.view(B * n, o)

    metrics = {}
    metrics['loss'] = float(sum(losses) / len(losses)) if losses else float('nan')
    # Use evaluator methods (they expect logits & targets as tensors)
    metrics['accuracy'] = evaluator.accuracy(flat_logits, flat_targets)
    metrics['f1_micro'] = evaluator.f1_score(flat_logits, flat_targets, average='micro')
    metrics['precision_micro'] = evaluator.precision(flat_logits, flat_targets, average='micro')
    metrics['recall_micro'] = evaluator.recall(flat_logits, flat_targets, average='micro')
    try:
        metrics['auroc_micro'] = evaluator.auroc(flat_logits, flat_targets, average='micro')
    except Exception:
        metrics['auroc_micro'] = float('nan')

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
    checkpoint_monitor: str = "f1_micro",  # metric to maximize on val
    checkpoint_mode: str = "max",          # max or min
) -> Dict[str, Any]:
    """
    Full training loop. Returns history dict with train & val metrics per epoch.
    """

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"{device_name} is initialised to train the model.")
    model.to(device)

    history = {
        "train": [],
        "val": []
    }

    best_monitor_val = None
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    # Outer epoch progressbar
    epoch_pbar = tqdm(range(1, n_epochs + 1), desc="Epochs")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for epoch in epoch_pbar:
        model.train()
        train_losses = []
        logits_accum: List[torch.Tensor] = []
        targets_accum: List[torch.Tensor] = []

        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
        for batch in train_pbar:
            # accept tuple/list or dict
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs, targets = batch['bodyparts'], batch['actions']
            else:
                raise ValueError("Dataloader must return (bodyparts, actions) or a dict with 'bodyparts'/'actions'.")

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            if amp:
                with autocast():
                    logits = model(inputs)                     # (B, n, n_outputs)
                    loss = loss_fn(logits, targets)
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))
            logits_accum.append(logits.detach().cpu())
            targets_accum.append(targets.detach().cpu())

            train_pbar.set_postfix({"batch_loss": f"{train_losses[-1]:.4f}", "avg_loss": f"{sum(train_losses)/len(train_losses):.4f}"})

        # End of training epoch; compute epoch-level train metrics
        all_train_logits = torch.cat(logits_accum, dim=0)   # (B', n, o)
        all_train_targets = torch.cat(targets_accum, dim=0)
        B_t, n_t, o_t = all_train_logits.shape
        flat_train_logits = all_train_logits.view(B_t * n_t, o_t)
        flat_train_targets = all_train_targets.view(B_t * n_t, o_t)

        train_metrics = {
            "loss": float(sum(train_losses)/len(train_losses)) if train_losses else float('nan'),
            "accuracy": evaluator.accuracy(flat_train_logits, flat_train_targets),
            "f1_micro": evaluator.f1_score(flat_train_logits, flat_train_targets, average='micro'),
            "precision_micro": evaluator.precision(flat_train_logits, flat_train_targets, average='micro'),
            "recall_micro": evaluator.recall(flat_train_logits, flat_train_targets, average='micro'),
        }
        try:
            train_metrics["auroc_micro"] = evaluator.auroc(flat_train_logits, flat_train_targets, average='micro')
        except Exception:
            train_metrics["auroc_micro"] = float('nan')

        history["train"].append(train_metrics)

        # Validate
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
            # Update epoch progressbar with concise summary
            epoch_pbar.set_postfix({
                "train_loss": f"{train_metrics['loss']:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
                "val_f1": f"{val_metrics.get('f1_micro', float('nan')):.4f}"
            })

            # Checkpointing: save best
            if checkpoint_path:
                monitor_value = val_metrics.get(checkpoint_monitor)
                if monitor_value is None or (isinstance(monitor_value, float) and (monitor_value != monitor_value)):  # nan check
                    # do not checkpoint on nan
                    pass
                else:
                    is_better = False
                    if best_monitor_val is None:
                        is_better = True
                    else:
                        if checkpoint_mode == "max" and monitor_value > best_monitor_val:
                            is_better = True
                        if checkpoint_mode == "min" and monitor_value < best_monitor_val:
                            is_better = True

                    if is_better:
                        best_monitor_val = monitor_value
                        # save state
                        torch.save({
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "monitor": monitor_value,
                        }, checkpoint_path)
        else:
            # still update epoch progressbar summary w/o val
            epoch_pbar.set_postfix({"train_loss": f"{train_metrics['loss']:.4f}"})

    return history
