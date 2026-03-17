"""
train.py — Entrenamiento de MiniGPT (~110M parámetros)
Optimizado para GPU (Colab T4) con Mixed Precision y Gradient Accumulation.
También funciona en CPU (tu laptop) de forma automática, más lento pero igual.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import json
import time
import os
import math

from model.transformer import MiniGPT


# ======================================================================
# DETECCIÓN DE DISPOSITIVO
# Se detecta automáticamente: usa GPU si hay una disponible, si no CPU.
# En Colab con GPU activada → "cuda", en tu laptop → "cpu"
# ======================================================================

DEVICE = (
    "cuda"  if torch.cuda.is_available()  else
    "mps"   if torch.backends.mps.is_available() else  # Mac con Apple Silicon
    "cpu"
)

# Mixed Precision solo tiene sentido en GPU CUDA
# En CPU se desactiva automáticamente para evitar errores
USE_AMP = (DEVICE == "cuda")

print(f"Dispositivo: {DEVICE.upper()}")
if USE_AMP:
    print("Mixed Precision (FP16): ACTIVADO  → ~2x más rápido, mitad de VRAM")
else:
    print("Mixed Precision (FP16): desactivado (solo disponible en GPU CUDA)")


# ======================================================================
# CONFIGURACIÓN
# ======================================================================

CONFIG = {
    # ── Arquitectura (~110M parámetros, igual que GPT-2 medium) ────────────
    # Cambia estos valores si quieres experimentar con tamaños distintos:
    #   Pequeño (~30M):  d_model=256, n_heads=8,  n_layers=6,  d_ff=1024
    #   Mediano (~60M):  d_model=512, n_heads=8,  n_layers=10, d_ff=2048
    #   Grande  (~110M): d_model=768, n_heads=12, n_layers=12, d_ff=3072  ← este
    "vocab_size" : 32000,   # se sobreescribe al cargar el tokenizer
    "d_model"    : 768,
    "n_heads"    : 12,
    "n_layers"   : 12,
    "d_ff"       : 3072,    # siempre 4 × d_model
    "max_len"    : 512,
    "dropout"    : 0.0,

    # ── Hiperparámetros de entrenamiento ───────────────────────────────────
    # batch_size=16 + accumulation_steps=4 → batch efectivo de 64
    # Si la VRAM se agota, baja batch_size a 8 (batch efectivo = 32)
    "batch_size"         : 8,
    "accumulation_steps" : 8,       # gradient accumulation
    "epochs"             : 3,       # 3 épocas caben bien en 12h de Colab
    "lr"                 : 3e-4,    # un poco más alto que antes para compensar
    "grad_clip"          : 1.0,
    "warmup_steps"       : 1000,     # más warmup para modelo más grande

    # ── Rutas ──────────────────────────────────────────────────────────────
    # En Colab estas rutas apuntan a Google Drive (se configuran en el notebook)
    # En tu laptop apuntan a las carpetas locales del proyecto
    "dataset_path"  : "data/dataset.jsonl",
    "tokenizer_path": "models/tokenizer.json",
    "checkpoint_dir": "models/checkpoints",
}


# ======================================================================
# DATASET
# ======================================================================

def ensure_dataset_exists(path: str = "data/dataset.jsonl"):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ejemplo = (
        "Este es un dataset de ejemplo para inicializar el proceso. "
        "Ejecuta data/prepare.py para generar un dataset completo."
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": ejemplo}, ensure_ascii=False) + "\n")
    print(f"Archivo {path} creado con ejemplo. Ejecuta 'python data/prepare.py' primero.")


class TextDataset(Dataset):
    def __init__(self, path: str, tokenizer: Tokenizer, max_len: int):
        ensure_dataset_exists(path)

        self.examples = []
        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")
        pad_id = tokenizer.token_to_id("<pad>")

        print(f"Cargando dataset desde {path}...")
        with open(path, encoding='utf-8') as f:
            lineas = f.readlines()

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

            # Truncar si es muy largo
            if len(ids) > max_len + 1:
                ids = ids[:max_len + 1]

            # Padding al final
            ids = ids + [pad_id] * (max_len + 1 - len(ids))

            self.examples.append((
                torch.tensor(ids[:-1], dtype=torch.long),
                torch.tensor(ids[1:],  dtype=torch.long),
            ))

        print(f"Ejemplos validos: {len(self.examples):,} | Descartados: {saltados:,}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ======================================================================
# ENTRENAMIENTO POR ÉPOCA — con Mixed Precision y Gradient Accumulation
# ======================================================================

def entrenar_epoca(model, loader, optimizer, scheduler, scaler, config,
                   epoca, total_epocas, pad_id):
    model.train()

    perdida_total   = 0.0
    batches_vistos  = 0
    tokens_procesados = 0
    t_inicio        = time.time()

    # Gradient accumulation: acumulamos gradientes durante N steps
    # y solo actualizamos los pesos cada accumulation_steps batches.
    # Esto simula un batch efectivo más grande sin usar más VRAM.
    accumulation_steps = config["accumulation_steps"]
    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Mover tensores al dispositivo (GPU o CPU)
        inputs  = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        # ── Forward con Mixed Precision ────────────────────────────────
        # torch.amp.autocast convierte automáticamente las operaciones
        # más pesadas (matmul, conv) a FP16, manteniendo FP32 donde
        # la precisión es crítica (softmax, layer norm).
        # Si USE_AMP=False (CPU), el autocast es un no-op transparente.
        with torch.amp.autocast(device_type=DEVICE, enabled=USE_AMP):
            logits = model(inputs)
            loss   = nn.functional.cross_entropy(
                logits.view(-1, config["vocab_size"]),
                targets.view(-1),
                ignore_index=pad_id,
                label_smoothing=0.1,
            )
            # Dividimos la loss por accumulation_steps para que el
            # gradiente acumulado sea equivalente al de un batch grande
            loss_scaled = loss / accumulation_steps

        # ── Backward con GradScaler ────────────────────────────────────
        # El GradScaler multiplica la loss por un factor grande antes
        # del backward para evitar underflow en FP16 (los gradientes
        # pequeños se volverían cero). Luego descala antes del optimizer.
        if USE_AMP:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # ── Actualización de pesos cada accumulation_steps ────────────
        if (batch_idx + 1) % accumulation_steps == 0:
            if USE_AMP:
                # Descalar gradientes, aplicar grad_clip y actualizar
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["grad_clip"]
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["grad_clip"]
                )
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        perdida_total     += loss.item()
        batches_vistos    += 1
        tokens_procesados += inputs.numel()

        if (batch_idx + 1) % 100 == 0:
            perdida_promedio = perdida_total / batches_vistos
            perplexity       = math.exp(min(perdida_promedio, 20))
            elapsed          = time.time() - t_inicio
            tok_per_sec      = tokens_procesados / elapsed

            # Mostrar uso de VRAM si estamos en GPU
            vram_info = ""
            if DEVICE == "cuda":
                vram_used = torch.cuda.memory_allocated() / 1e9
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                vram_info = f" | VRAM: {vram_used:.1f}/{vram_total:.1f}GB"

            print(
                f"  Época {epoca}/{total_epocas} | "
                f"Batch {batch_idx+1:>5}/{len(loader)} | "
                f"Loss: {perdida_promedio:.4f} | "
                f"PPL: {perplexity:.1f} | "
                f"{tok_per_sec:.0f} tok/s"
                f"{vram_info}"
            )

    return perdida_total / batches_vistos


# ======================================================================
# EVALUACIÓN
# ======================================================================

def evaluar(model, loader, config, pad_id):
    model.eval()
    perdida_total = 0.0
    batches       = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            with torch.amp.autocast(device_type=DEVICE, enabled=USE_AMP):
                logits = model(inputs)
                loss   = nn.functional.cross_entropy(
                    logits.view(-1, config["vocab_size"]),
                    targets.view(-1),
                    ignore_index=pad_id,
                )
            perdida_total += loss.item()
            batches       += 1

    return perdida_total / batches


# ======================================================================
# CHECKPOINT — reanudación desde el último estado guardado
# ======================================================================

def cargar_checkpoint_si_existe(model, optimizer, scheduler, scaler,
                                 checkpoint_dir):
    """
    Si existe last_model.pt, restaura todos los estados y devuelve
    (epoca_inicio, mejor_val_loss).
    Si no existe, devuelve (1, inf) — arranque fresco.
    """
    ruta = os.path.join(checkpoint_dir, "last_model.pt")
    if not os.path.exists(ruta):
        print("No se encontró checkpoint previo. Iniciando desde cero.\n")
        return 1, float('inf')

    print(f"Checkpoint encontrado en {ruta}. Reanudando...")
    # map_location="cpu" primero para evitar problemas si cambia de GPU a CPU
    ck = torch.load(ruta, map_location="cpu")

    model.load_state_dict(ck["model_state"])
    model.to(DEVICE)  # mover al dispositivo correcto después de cargar

    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])

    # El scaler solo existe cuando hay GPU, así que lo restauramos
    # solo si el checkpoint lo tiene y estamos en GPU
    if USE_AMP and "scaler" in ck:
        scaler.load_state_dict(ck["scaler"])

    epoca_inicio   = ck["epoca"] + 1
    mejor_val_loss = ck.get("mejor_val_loss", ck["val_loss"])

    print(
        f"  Reanudando desde época {epoca_inicio} "
        f"(val_loss anterior: {ck['val_loss']:.4f}, "
        f"mejor: {mejor_val_loss:.4f})\n"
    )
    return epoca_inicio, mejor_val_loss


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    # En CPU: 2 hilos para los P-cores del i7
    # En GPU: PyTorch gestiona los hilos automáticamente
    if DEVICE == "cpu":
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)

    print("=" * 60)
    print("MiniGPT — Entrenamiento")
    print("=" * 60)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────────────
    tokenizer  = Tokenizer.from_file(CONFIG["tokenizer_path"])
    vocab_size = tokenizer.get_vocab_size()
    CONFIG["vocab_size"] = vocab_size
    print(f"Tokenizer cargado | Vocabulario: {vocab_size:,} tokens\n")

    pad_id = tokenizer.token_to_id("<pad>")

    # ── Dataset y DataLoaders ──────────────────────────────────────────
    dataset = TextDataset(
        CONFIG["dataset_path"], tokenizer, CONFIG["max_len"]
    )

    val_size   = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # num_workers=2 en Colab (GPU) para cargar datos en paralelo
    # num_workers=0 en CPU para evitar problemas en Windows
    num_workers = 2 if DEVICE == "cuda" else 0

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),  # acelera transferencia CPU→GPU
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
    )

    print(f"Dataset dividido:")
    print(f"  Entrenamiento: {len(train_ds):,} ejemplos ({len(train_loader):,} batches)")
    print(f"  Validación:    {len(val_ds):,} ejemplos")
    print(f"  Batch efectivo: {CONFIG['batch_size'] * CONFIG['accumulation_steps']}")
    print()

    # ── Modelo ─────────────────────────────────────────────────────────
    model = MiniGPT(**{k: CONFIG[k] for k in
                       ["vocab_size", "d_model", "n_heads", "n_layers",
                        "d_ff", "max_len", "dropout"]})
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo inicializado | Parámetros: {total_params:,}\n")

    # ── Optimizador ────────────────────────────────────────────────────
    # Separamos los parámetros con y sin weight decay
    # Los biases y LayerNorm no deben tener decay (práctica estándar)
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=CONFIG["lr"],
        betas=(0.9, 0.95),  # betas ligeramente ajustados para modelos grandes
        eps=1e-8,
    )

    # ── Scheduler: Cosine con Warmup ───────────────────────────────────
    total_steps  = len(train_loader) * CONFIG["epochs"]
    warmup_steps = CONFIG["warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            # Warmup lineal: el LR sube gradualmente para estabilizar
            # las primeras actualizaciones del modelo grande
            return float(step) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        # Cosine decay: baja suavemente hasta el 10% del LR máximo
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── GradScaler para Mixed Precision ────────────────────────────────
    # Si USE_AMP=False (CPU), el scaler está desactivado y no hace nada
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    # ── Reanudar desde checkpoint si existe ────────────────────────────
    epoca_inicio, mejor_val_loss = cargar_checkpoint_si_existe(
        model, optimizer, scheduler, scaler, CONFIG["checkpoint_dir"]
    )

    if epoca_inicio > CONFIG["epochs"]:
        print("El entrenamiento ya estaba completo según el checkpoint.")
        exit(0)

    # ── Bucle principal ────────────────────────────────────────────────
    print("=" * 60)
    print(f"Iniciando entrenamiento en {DEVICE.upper()}")
    print(f"Épocas: {epoca_inicio} → {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}  |  "
          f"Acumulación: {CONFIG['accumulation_steps']}  |  "
          f"Batch efectivo: {CONFIG['batch_size'] * CONFIG['accumulation_steps']}")
    print(f"LR máximo: {CONFIG['lr']}  |  Warmup: {warmup_steps} steps")
    print("=" * 60)

    historial  = []
    t_total    = time.time()

    for epoca in range(epoca_inicio, CONFIG["epochs"] + 1):
        t_ep = time.time()

        train_loss = entrenar_epoca(
            model, train_loader, optimizer, scheduler, scaler,
            CONFIG, epoca, CONFIG["epochs"], pad_id
        )
        val_loss = evaluar(model, val_loader, CONFIG, pad_id)
        val_ppl  = math.exp(min(val_loss, 20))
        duracion = time.time() - t_ep

        historial.append({
            "epoca"      : epoca,
            "train_loss" : round(train_loss, 4),
            "val_loss"   : round(val_loss, 4),
            "val_ppl"    : round(val_ppl, 2),
            "duracion_s" : round(duracion, 1),
        })

        print(f"\n{'─' * 60}")
        print(f"  Época {epoca} completada en {duracion/60:.1f} min")
        print(f"  Train Loss: {train_loss:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  |  "
              f"Val Perplexity: {val_ppl:.1f}")

        # ── Guardar mejor modelo ───────────────────────────────────────
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            ruta_best = os.path.join(CONFIG["checkpoint_dir"], "best_model.pt")
            torch.save({
                "epoca"      : epoca,
                "config"     : CONFIG,
                "model_state": model.state_dict(),
                "val_loss"   : val_loss,
            }, ruta_best)
            print(f"  ★ Mejor modelo guardado (val_loss={val_loss:.4f})")

        # ── Guardar último estado completo (para reanudar) ────────────
        ruta_last = os.path.join(CONFIG["checkpoint_dir"], "last_model.pt")
        checkpoint = {
            "epoca"         : epoca,
            "config"        : CONFIG,
            "model_state"   : model.state_dict(),
            "optimizer"     : optimizer.state_dict(),
            "scheduler"     : scheduler.state_dict(),
            "val_loss"      : val_loss,
            "mejor_val_loss": mejor_val_loss,
        }
        if USE_AMP:
            checkpoint["scaler"] = scaler.state_dict()

        torch.save(checkpoint, ruta_last)
        print(f"  Checkpoint guardado en época {epoca}")
        print(f"{'─' * 60}\n")

    # ── Finalización ──────────────────────────────────────────────────
    tiempo_total = time.time() - t_total
    print(f"{'=' * 60}")
    print(f"Entrenamiento completado en {tiempo_total/60:.1f} minutos")
    print(f"Mejor val_loss: {mejor_val_loss:.4f} "
          f"(perplexity: {math.exp(min(mejor_val_loss, 20)):.1f})")
    print(f"{'=' * 60}")

    # ── Historial acumulativo ─────────────────────────────────────────
    historial_path = "models/historial.json"
    historial_previo = []
    if os.path.exists(historial_path):
        try:
            with open(historial_path, "r") as f:
                historial_previo = json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass

    with open(historial_path, "w") as f:
        json.dump(historial_previo + historial, f, indent=2)
    print("Historial exportado a models/historial.json")