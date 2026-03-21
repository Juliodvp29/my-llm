import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tokenizers import Tokenizer
import json
import time
import os
import math

from model.transformer import MiniGPT

CONFIG_SFT = {
    "vocab_size"         : 32000,
    "d_model"            : 1024,
    "n_heads"            : 16,
    "n_layers"           : 24,
    "d_ff"               : 4096,
    "max_len"            : 1024,
    "dropout"            : 0.05,
    "batch_size_per_gpu" : 1,
    "accumulation_steps" : 16,
    "epochs"             : 5,
    "lr"                 : 1e-5,
    "grad_clip"          : 1.0,
    "warmup_steps"       : 50,

    "dataset_path"       : "data/sft_dataset.jsonl",
    "tokenizer_path"     : "models/tokenizer.json",
    "pretrain_checkpoint": "models/checkpoints_sft/sft_best_model.pt",
    "checkpoint_dir"     : "models/checkpoints_sft",
}


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, is_main_process=True):
        self.examples = []
        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")
        pad_id = tokenizer.token_to_id("<pad>")

        if is_main_process:
            print(f"Cargando SFT dataset desde {path}...")

        with open(path, encoding="utf-8") as f:
            lineas = f.readlines()

        saltados = 0
        for linea in lineas:
            try:
                texto = json.loads(linea.strip())["text"]
            except (json.JSONDecodeError, KeyError):
                saltados += 1
                continue

            ids = [bos_id] + tokenizer.encode(texto).ids + [eos_id]

            if len(ids) < 8:
                saltados += 1
                continue
            if len(ids) > max_len + 1:
                ids = ids[:max_len] + [eos_id]

            ids = ids + [pad_id] * (max_len + 1 - len(ids))

            self.examples.append((
                torch.tensor(ids[:-1], dtype=torch.long),
                torch.tensor(ids[1:],  dtype=torch.long),
            ))

        if is_main_process:
            print(f"Ejemplos válidos: {len(self.examples):,} | Descartados: {saltados}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def entrenar_epoca(rank, model, loader, optimizer, scheduler, scaler, config,
                   epoca, total_epocas, pad_id, world_size, mejor_val_loss=float('inf')):
    model.train()
    perdida_total  = 0.0
    batches_vistos = 0
    t_inicio       = time.time()
    accumulation_steps = config["accumulation_steps"]
    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs  = inputs.to(rank, non_blocking=True)
        targets = targets.to(rank, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=True):
            logits = model(inputs)
            loss   = nn.functional.cross_entropy(
                logits.view(-1, config["vocab_size"]),
                targets.view(-1),
                ignore_index=pad_id,
                label_smoothing=0.05,
            )
            loss_scaled = loss / accumulation_steps

        scaler.scale(loss_scaled).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        perdida_total  += loss.item()
        batches_vistos += 1

        if rank == 0 and (batch_idx + 1) % 20 == 0:
            perdida_promedio = perdida_total / batches_vistos
            perplexity       = math.exp(min(perdida_promedio, 20))
            elapsed          = time.time() - t_inicio
            vram_used        = torch.cuda.max_memory_allocated(rank) / 1e9
            vram_total       = torch.cuda.get_device_properties(rank).total_memory / 1e9
            print(
                f"  Época {epoca}/{total_epocas} | "
                f"Batch {batch_idx+1:>4}/{len(loader)} | "
                f"Loss: {perdida_promedio:.4f} | "
                f"PPL: {perplexity:.1f} | "
                f"GPU{rank}: {vram_used:.1f}/{vram_total:.1f}GB"
            )

    loss_tensor = torch.tensor(perdida_total / max(1, batches_vistos)).to(rank)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    return loss_tensor.item() / world_size


def evaluar(rank, model, loader, config, pad_id, world_size):
    model.eval()
    perdida_total = 0.0
    batches       = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(rank, non_blocking=True)
            targets = targets.to(rank, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(inputs)
                loss   = nn.functional.cross_entropy(
                    logits.view(-1, config["vocab_size"]),
                    targets.view(-1),
                    ignore_index=pad_id,
                )
            perdida_total += loss.item()
            batches       += 1

    loss_tensor = torch.tensor(perdida_total / max(1, batches)).to(rank)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    return loss_tensor.item() / world_size


def main_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12356'  # puerto diferente al pre-entrenamiento
    backend = "nccl" if os.name != "nt" else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    if rank == 0:
        print("=" * 60)
        print("SFT — Supervised Fine-Tuning")
        print("=" * 60)
        os.makedirs(CONFIG_SFT["checkpoint_dir"], exist_ok=True)

    tokenizer = Tokenizer.from_file(CONFIG_SFT["tokenizer_path"])
    CONFIG_SFT["vocab_size"] = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")

    dataset = SFTDataset(
        CONFIG_SFT["dataset_path"], tokenizer,
        CONFIG_SFT["max_len"], is_main_process=(rank == 0)
    )

    val_size   = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG_SFT["batch_size_per_gpu"],
        sampler=train_sampler, drop_last=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG_SFT["batch_size_per_gpu"],
        sampler=val_sampler, drop_last=False, num_workers=0, pin_memory=True
    )

    if rank == 0:
        print(f"Train: {train_size} | Val: {val_size}")

    # Modelo — cargamos desde el pre-entrenamiento
    model = MiniGPT(**{k: CONFIG_SFT[k] for k in
                       ["vocab_size", "d_model", "n_heads", "n_layers", "d_ff", "max_len", "dropout"]})
    model = model.to(rank)

    # Cargar pesos del pre-entrenamiento
    ck = torch.load(CONFIG_SFT["pretrain_checkpoint"], map_location=f"cuda:{rank}")
    model.load_state_dict(ck["model_state"])
    if rank == 0:
        print(f"Pesos cargados desde: {CONFIG_SFT['pretrain_checkpoint']}")

    model = DDP(model, device_ids=[rank])

    # Optimizador con lr muy bajo para no destruir el pre-entrenamiento
    decay_params    = [p for n, p in model.module.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.module.named_parameters() if p.requires_grad and p.dim() < 2]

    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=CONFIG_SFT["lr"], betas=(0.9, 0.95), eps=1e-8)

    scaler = torch.amp.GradScaler(enabled=True)

    # Scheduler simple con warmup corto
    total_steps  = len(train_loader) * CONFIG_SFT["epochs"]
    warmup_steps = CONFIG_SFT["warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Entrenamiento
    mejor_val_loss = float('inf')
    t_total        = time.time()
    historial      = []

    for epoca in range(1, CONFIG_SFT["epochs"] + 1):
        train_sampler.set_epoch(epoca)
        t_ep = time.time()

        train_loss = entrenar_epoca(
            rank, model, train_loader, optimizer, scheduler, scaler,
            CONFIG_SFT, epoca, CONFIG_SFT["epochs"], pad_id, world_size, mejor_val_loss
        )
        val_loss = evaluar(rank, model, val_loader, CONFIG_SFT, pad_id, world_size)

        if rank == 0:
            val_ppl  = math.exp(min(val_loss, 20))
            duracion = time.time() - t_ep

            historial.append({
                "epoca": epoca, "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4), "val_ppl": round(val_ppl, 2),
            })

            print(f"\n{'─' * 60}")
            print(f"  Época {epoca} completada en {duracion/60:.1f} min")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {val_ppl:.1f}")

            if val_loss < mejor_val_loss:
                mejor_val_loss = val_loss
                torch.save({
                    "epoca": epoca, "config": CONFIG_SFT, "val_loss": val_loss,
                    "model_state": model.module.state_dict()
                }, os.path.join(CONFIG_SFT["checkpoint_dir"], "sft_best_model.pt"))
                print(f"  ★ Mejor modelo SFT guardado")

            torch.save({
                "epoca": epoca, "config": CONFIG_SFT, "val_loss": val_loss,
                "model_state": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            }, os.path.join(CONFIG_SFT["checkpoint_dir"], "sft_last_model.pt"))
            print(f"{'─' * 60}\n")

    if rank == 0:
        print(f"{'=' * 60}")
        print(f"SFT completado en {(time.time() - t_total)/60:.1f} minutos")
        print(f"Mejor val_loss: {mejor_val_loss:.4f}")
        print(f"{'=' * 60}")
        with open("models/historial_sft.json", "w") as f:
            json.dump(historial, f, indent=2)

    dist.destroy_process_group()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: se requiere GPU con CUDA.")
        exit(1)

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)