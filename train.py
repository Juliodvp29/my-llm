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

# CONFIGURACIÓN

CONFIG = {
    "vocab_size" : 32000,
    "d_model"    : 1024,
    "n_heads"    : 16,
    "n_layers"   : 24,
    "d_ff"       : 4096,
    "max_len"    : 512,
    "dropout"    : 0.1,
    "batch_size_per_gpu" : 4,
    "accumulation_steps" : 8,
    "epochs"             : 6,
    "lr"                 : 2e-4,
    "grad_clip"          : 1.0,
    "warmup_steps"       : 3000,

    # Rutas
  "dataset_path"  : "data/dataset.jsonl",
    "tokenizer_path": "models/tokenizer.json",
    "checkpoint_dir": "models/checkpoints",
}

# DATASET

def ensure_dataset_exists(path: str = "data/dataset.jsonl"):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ejemplo = "Este es un dataset de ejemplo para inicializar. Ejecutar data/prepare.py primero."
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": ejemplo}, ensure_ascii=False) + "\n")
    print(f"Archivo {path} creado temporalmente.")

class TextDataset(Dataset):
    def __init__(self, path: str, tokenizer: Tokenizer, max_len: int, is_main_process=True):
        if is_main_process:
            ensure_dataset_exists(path)

        self.examples = []
        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")
        pad_id = tokenizer.token_to_id("<pad>")

        if is_main_process:
            print(f"Cargando dataset desde {path}...")
            
        with open(path, encoding='utf-8') as f:
            lineas = f.readlines()

        if is_main_process:
            print(f"Tokenizando {len(lineas):,} fragmentos...")
            
        saltados = 0

        for linea in lineas:
            try:
                texto = json.loads(linea.strip())["text"]
            except (json.JSONDecodeError, KeyError):
                saltados += 1
                continue

            ids = tokenizer.encode(texto).ids
            ids = [bos_id] + ids + [eos_id]

            if len(ids) < 8:
                saltados += 1
                continue

            if len(ids) > max_len + 1:
                ids = ids[:max_len + 1]

            ids = ids + [pad_id] * (max_len + 1 - len(ids))

            self.examples.append((
                torch.tensor(ids[:-1], dtype=torch.long),
                torch.tensor(ids[1:],  dtype=torch.long),
            ))

        if is_main_process:
            print(f"Ejemplos validos: {len(self.examples):,} | Descartados: {saltados:,}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# ENTRENAMIENTO Y EVALUACION (DDP)

def entrenar_epoca(rank, model, loader, optimizer, scheduler, scaler, config,
                   epoca, total_epocas, pad_id, world_size, mejor_val_loss=float('inf')):
    model.train()

    perdida_total     = 0.0
    batches_vistos    = 0
    tokens_procesados = 0
    t_inicio          = time.time()

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
                label_smoothing=0.1,
            )
            loss_scaled = loss / accumulation_steps

        scaler.scale(loss_scaled).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["grad_clip"]
            )
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            optimizer.zero_grad()

        # Guardar checkpoint intermedio cada 1000 pasos (solo rank=0)
        if rank == 0 and (batch_idx + 1) % 1000 == 0:
            torch.save({
                "epoca"       : epoca,
                "batch_idx"   : batch_idx,
                "config"      : config,
                "val_loss"    : float('inf'),  # no tenemos val_loss todavía
                "mejor_val_loss": mejor_val_loss,
                "model_state" : model.module.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "scheduler"   : scheduler.state_dict(),
                "scaler"      : scaler.state_dict(),
            }, os.path.join(config["checkpoint_dir"], "last_model.pt"))
            print(f"  [Backup] Checkpoint intermedio guardado en paso {batch_idx + 1}")

        perdida_total     += loss.item()
        batches_vistos    += 1
        # Multiplicamos por world_size para estimar toc/s globales reales
        tokens_procesados += inputs.numel() * world_size 

        if rank == 0 and (batch_idx + 1) % 100 == 0:
            perdida_promedio = perdida_total / batches_vistos
            perplexity       = math.exp(min(perdida_promedio, 20))
            elapsed          = time.time() - t_inicio
            tok_per_sec      = tokens_procesados / elapsed

            # Memoria real (DDP mantiene balance 50/50 exacto)
            vram_used  = torch.cuda.max_memory_allocated(rank) / 1e9
            vram_total = torch.cuda.get_device_properties(rank).total_memory / 1e9
            vram_info  = f" | GPU{rank}: {vram_used:.1f}/{vram_total:.1f}GB"

            print(
                f"  Época {epoca}/{total_epocas} | "
                f"Batch {batch_idx+1:>5}/{len(loader)} | "
                f"Loss: {perdida_promedio:.4f} | "
                f"PPL: {perplexity:.1f} | "
                f"{tok_per_sec:.0f} tok/s"
                f"{vram_info}"
            )

    # Promediar el loss entre todas las GPUs de forma estricta
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
                loss = nn.functional.cross_entropy(
                    logits.view(-1, config["vocab_size"]),
                    targets.view(-1),
                    ignore_index=pad_id,
                    label_smoothing=0.15,
                )
            perdida_total += loss.item()
            batches       += 1

    loss_tensor = torch.tensor(perdida_total / max(1, batches)).to(rank)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    return loss_tensor.item() / world_size

def cargar_checkpoint_si_existe(rank, model, optimizer, scheduler, scaler, dir_path):
    ruta = os.path.join(dir_path, "last_model.pt")
    if not os.path.exists(ruta):
        if rank == 0: print("No se encontró checkpoint. Iniciando desde cero.\n")
        return 1, float('inf')

    if rank == 0: print(f"Checkpoint encontrado en {ruta}. Reanudando...")
    
    # Cargamos pasándolo directo al rank asignado de esta GPU
    ck = torch.load(ruta, map_location=f"cuda:{rank}")

    # En DDP, usamos model.module obligatoriamente para inyectar los pesos reales
    model.module.load_state_dict(ck["model_state"])
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    
    if "scaler" in ck:
        scaler.load_state_dict(ck["scaler"])

    epoca_inicio   = ck["epoca"] + 1
    mejor_val_loss = ck.get("mejor_val_loss", ck["val_loss"])

    if rank == 0:
        print(
            f"  Reanudando desde época {epoca_inicio} "
            f"(val_loss anterior: {ck['val_loss']:.4f}, "
            f"mejor: {mejor_val_loss:.4f})\n"
        )
    return epoca_inicio, mejor_val_loss

# WORKER PRINCIPAL DDP

def main_worker(rank, world_size):
    # Inicializar el grupo de procesos DDP
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    backend = "nccl" if os.name != "nt" else "gloo" # Kaggle(Linux)=nccl, Windows=gloo
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)

    if rank == 0:
        print("=" * 60)
        print("Comenzando — Entrenamiento !!")
        print("=" * 60)
        os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # ── Tokenizer
    tokenizer  = Tokenizer.from_file(CONFIG["tokenizer_path"])
    CONFIG["vocab_size"] = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")

    # ── Dataset (Previene colisión avisando con is_main_process)
    dataset = TextDataset(CONFIG["dataset_path"], tokenizer, CONFIG["max_len"], is_main_process=(rank==0))

    val_size   = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # Semilla rígida para sincronía perfecta
    )

    # DistributedSampler asegura que la GPU0 y la GPU1 no procesen los mismos datos idénticos
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    # num_workers=0 obliga a correrlos en el proceso principal de forma segura. (para kaggle)
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size_per_gpu"],
        sampler=train_sampler, drop_last=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size_per_gpu"],
        sampler=val_sampler, drop_last=True, num_workers=0, pin_memory=True
    )

    if rank == 0:
        print(f"Dataset dividido y balanceado!")
        print(f"  Batch distribuído: {CONFIG['batch_size_per_gpu']} por cada una de las {world_size} GPUs")
        print(f"  Batch total real procesándose de golpe: {CONFIG['batch_size_per_gpu'] * world_size}")
        print()

    # ── Modelo
    model = MiniGPT(**{k: CONFIG[k] for k in
                       ["vocab_size", "d_model", "n_heads", "n_layers", "d_ff", "max_len", "dropout"]})
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    if rank == 0:
        print(f"Modelo cargado ({(sum(p.numel() for p in model.parameters())):,} parámetros)\n")

    # ── Optimizador
    decay_params    = [p for n, p in model.module.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.module.named_parameters() if p.requires_grad and p.dim() < 2]

    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=CONFIG["lr"], betas=(0.9, 0.95), eps=1e-8)

    # ── Scheduler
    total_steps  = len(train_loader) * CONFIG["epochs"]
    warmup_steps = CONFIG["warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps: return float(step) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler(enabled=True)

    # ── Checkpoint
    epoca_inicio, mejor_val_loss = cargar_checkpoint_si_existe(
        rank, model, optimizer, scheduler, scaler, CONFIG["checkpoint_dir"]
    )

    if epoca_inicio > CONFIG["epochs"]:
        if rank == 0: print("Entrenamiento completado.")
        dist.destroy_process_group()
        return

    # ── Bucle de iteración
    t_total = time.time()
    historial = []

    for epoca in range(epoca_inicio, CONFIG["epochs"] + 1):
        train_sampler.set_epoch(epoca) # Obligatorio en DDP para revolver cada epoch
        t_ep = time.time()

        train_loss = entrenar_epoca(
            rank, model, train_loader, optimizer, scheduler, scaler,
            CONFIG, epoca, CONFIG["epochs"], pad_id, world_size, mejor_val_loss
        )
        val_loss = evaluar(rank, model, val_loader, CONFIG, pad_id, world_size)
        
        # Solo la GPU maestra maneja métricas visuales y guarda el .pt
        if rank == 0:
            val_ppl = math.exp(min(val_loss, 20))
            duracion = time.time() - t_ep
            
            historial.append({
                "epoca"      : epoca, "train_loss" : round(train_loss, 4),
                "val_loss"   : round(val_loss, 4), "val_ppl"    : round(val_ppl, 2),
                "duracion_s" : round(duracion, 1)
            })

            print(f"\n{'─' * 60}")
            print(f"  Época {epoca} completada en {duracion/60:.1f} min")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f}")

            if val_loss < mejor_val_loss:
                mejor_val_loss = val_loss
                torch.save({
                    "epoca": epoca, "config": CONFIG, "val_loss": val_loss,
                    "model_state": model.module.state_dict()
                }, os.path.join(CONFIG["checkpoint_dir"], "best_model.pt"))
                print(f"  ★ Mejor modelo guardado")

            torch.save({
                "epoca": epoca, "config": CONFIG, "val_loss": val_loss, "mejor_val_loss": mejor_val_loss,
                "model_state": model.module.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()
            }, os.path.join(CONFIG["checkpoint_dir"], "last_model.pt"))
            print(f"  Checkpoint guardado en época {epoca}")
            print(f"{'─' * 60}\n")

    if rank == 0:
        print(f"{'=' * 60}\nEntrenamiento completado en {(time.time() - t_total)/60:.1f} minutos\n{'=' * 60}")
        h_path = "models/historial.json"
        
        # Mezclamos el nuevo historiaL con el anterior archivo json si existe
        hist_prev = []
        if os.path.exists(h_path):
            with open(h_path, "r") as f: hist_prev = json.load(f)
        with open(h_path, "w") as f: json.dump(hist_prev + historial, f, indent=2)

    dist.destroy_process_group()

# INICIO DE PROCESOS

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: DDP requiere al menos 1 GPU con CUDA. Abortando.")
        exit(1)

    # Identificamos el número de GPUs detectadas
    world_size = torch.cuda.device_count()
    
    # mp.spawn iniciará `main_worker` una vez "nprocs" veces (uno para cada GPU)
    # y enviará por defecto el (rank, *args) al worker. Rank es el ID de GPU.
    mp.spawn(
        main_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )