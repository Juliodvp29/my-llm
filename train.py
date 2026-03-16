import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import json
import time
import os
import math

from model.transformer import MiniGPT


def ensure_dataset_exists(path: str = "data/dataset.jsonl"):
    if os.path.exists(path):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    ejemplo = (
        "Este es un dataset de ejemplo para inicializar el proceso. "
        "Ejecuta data/prepare.py para generar un dataset completo. "
        "Este texto se repite para asegurar que haya suficientes tokens."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": ejemplo}, ensure_ascii=False) + "\n")

    print(f"Archivo {path} creado con registro de ejemplo. Ejecutar 'python data/prepare.py' para generar dataset completo.")


# ----------------------------------------------------------------------
# Configuración del modelo y entrenamiento
# ----------------------------------------------------------------------

CONFIG = {
    # Arquitectura 
    "vocab_size" : 32000,
    "d_model"    : 512,
    "n_heads"    : 8,      # d_model % n_heads == 0
    "n_layers"   : 8,
    "d_ff"       : 2048,   # 4 * d_model
    "max_len"    : 512,
    "dropout"    : 0.1,

    # Hiperparámetros de entrenamiento
    "batch_size"    : 16,
    "epochs"        : 5,
    "lr"            : 2e-4,
    "grad_clip"     : 1.0,

    # Rutas
    "dataset_path"  : "data/dataset.jsonl",
    "tokenizer_path": "models/tokenizer.json",
    "checkpoint_dir": "models/checkpoints",
}

# ----------------------------------------------------------------------
# Dataset iterador causal
# Genera pares (input, target) desplazados en un paso de tiempo,
# prediciendo el token t+1 dado el contexto hasta t.
# ----------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, path: str, tokenizer: Tokenizer, max_len: int):
        ensure_dataset_exists(path)

        self.examples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")
        pad_id = tokenizer.token_to_id("<pad>")

        print(f"Cargando dataset desde {path}...")
        with open(path, encoding='utf-8') as f:
            lineas = f.readlines()

        print(f"Tokenizando {len(lineas):,} fragmentos...")
        saltados = 0

        for linea in lineas:
            texto = json.loads(linea.strip())["text"]
            ids = tokenizer.encode(texto).ids

            # Incluir delimitadores de secuencia
            ids = [bos_id] + ids + [eos_id]

            # Descartar secuencias con contexto insuficiente
            if len(ids) < 8:
                saltados += 1
                continue

            # Truncar secuencias que exceden la ventana de contexto
            if len(ids) > max_len + 1:
                ids = ids[:max_len + 1]

            # Aplicar padding para estandarizar longitud
            ids = ids + [pad_id] * (max_len + 1 - len(ids))

            # Separar secuencias de entrada (t) y objetivo (t+1)
            self.examples.append((
                torch.tensor(ids[:-1], dtype=torch.long),
                torch.tensor(ids[1:],  dtype=torch.long),
            ))

        print(f"Ejemplos válidos: {len(self.examples):,} (Descartados: {saltados})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ----------------------------------------------------------------------
# Función de entrenamiento por época
# ----------------------------------------------------------------------

def entrenar_epoca(model, loader, optimizer, scheduler, config, epoca, total_epocas):
    model.train()
    pad_id = tokenizer.token_to_id("<pad>")

    perdida_total = 0.0
    batches_vistos = 0
    tokens_procesados = 0
    t_inicio = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)

        # Cálculo de pérdida excluyendo tokens de padding
        loss = nn.functional.cross_entropy(
            logits.view(-1, config["vocab_size"]),
            targets.view(-1),
            ignore_index=pad_id,
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        # Optimización
        optimizer.step()
        scheduler.step()

        perdida_total += loss.item()
        batches_vistos += 1
        tokens_procesados += inputs.numel()

        # Registro periódico
        if (batch_idx + 1) % 50 == 0:
            perdida_promedio = perdida_total / batches_vistos
            perplexity = math.exp(min(perdida_promedio, 20))
            elapsed = time.time() - t_inicio
            tok_per_sec = tokens_procesados / elapsed

            print(f"  Época {epoca}/{total_epocas} | "
                  f"Batch {batch_idx+1:>4}/{len(loader)} | "
                  f"Loss: {perdida_promedio:.4f} | "
                  f"Perplexity: {perplexity:.1f} | "
                  f"{tok_per_sec:.0f} tok/s")

    return perdida_total / batches_vistos


# ----------------------------------------------------------------------
# Evaluación de modelo
# Computa inferencia para validar la métrica de loss y perplexity.
# ----------------------------------------------------------------------

def evaluar(model, loader, config):
    model.eval()
    perdida_total = 0.0
    batches = 0
    pad_id = tokenizer.token_to_id("<pad>")

    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.view(-1, config["vocab_size"]),
                targets.view(-1),
                ignore_index=pad_id,
            )
            perdida_total += loss.item()
            batches += 1

    return perdida_total / batches


# ----------------------------------------------------------------------
# Ejecución principal
# ----------------------------------------------------------------------

if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    print("Iniciando entrenamiento de MiniGPT\n")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # --- Tokenizer ---
    tokenizer = Tokenizer.from_file(CONFIG["tokenizer_path"])
    vocab_size = tokenizer.get_vocab_size()
    CONFIG["vocab_size"] = vocab_size
    print(f"Tokenizer cargado | Vocabulario: {vocab_size} tokens\n")

    # --- Dataset y DataLoader ---
    dataset = TextDataset(CONFIG["dataset_path"], tokenizer, CONFIG["max_len"])

    # Split 90/10 (Train/Val)
    val_size   = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, drop_last=True)

    print(f"\nDataset dividido:")
    print(f"Entrenamiento: {len(train_ds):,} ejemplos ({len(train_loader)} batches)")
    print(f"Validación:    {len(val_ds):,} ejemplos\n")

    # --- Modelo ---
    model = MiniGPT(**{k: CONFIG[k] for k in
                       ["vocab_size","d_model","n_heads","n_layers",
                        "d_ff","max_len","dropout"]})

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo inicializado | Parámetros: {total_params:,}\n")

    # --- Optimización ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=0.01,  # Regularización L2
    )

    # Decaimiento del factor de aprendizaje (Cosine Annealing)
    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5
    )

    # --- Bucle de entrenamiento ---
    print("="*60)
    print("Iniciando entrenamiento...")
    print(f"Épocas: {CONFIG['epochs']}  |  "
          f"Batch size: {CONFIG['batch_size']}  |  "
          f"LR: {CONFIG['lr']}")
    print("="*60)

    mejor_val_loss = float('inf')
    historial = []
    t_total = time.time()

    for epoca in range(1, CONFIG["epochs"] + 1):
        t_ep = time.time()

        train_loss = entrenar_epoca(
            model, train_loader, optimizer, scheduler,
            CONFIG, epoca, CONFIG["epochs"]
        )
        val_loss = evaluar(model, val_loader, CONFIG)
        val_ppl  = math.exp(min(val_loss, 20))

        duracion = time.time() - t_ep
        historial.append({"epoca": epoca,
                          "train_loss": train_loss,
                          "val_loss": val_loss})

        print(f"\n{'─'*60}")
        print(f"  Época {epoca} completa en {duracion:.0f}s")
        print(f"  Train Loss: {train_loss:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  |  "
              f"Val Perplexity: {val_ppl:.1f}")

        # Checkpointing
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            ruta = os.path.join(CONFIG["checkpoint_dir"], "best_model.pt")
            torch.save({
                "epoca"     : epoca,
                "config"    : CONFIG,
                "model_state": model.state_dict(),
                "val_loss"  : val_loss,
            }, ruta)
            print(f"  Checkpoint guardado (val_loss={val_loss:.4f})")

        print(f"{'─'*60}\n")

    # --- Finalización ---
    tiempo_total = time.time() - t_total
    print(f"{'='*60}")
    print(f"Entrenamiento completado en {tiempo_total/60:.1f} minutos")
    print(f"Mejor val_loss: {mejor_val_loss:.4f} (perplexity: {math.exp(min(mejor_val_loss,20)):.1f})")
    print(f"{'='*60}")

    # Persistencia de métricas
    with open("models/historial.json", "w") as f:
        json.dump(historial, f, indent=2)
    print("Historial exportado a models/historial.json")