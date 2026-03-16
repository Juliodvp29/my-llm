from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import os
import json

# ----------------------------------------------------------------------
# 0. Verificación del dataset
# ----------------------------------------------------------------------

def ensure_dataset_exists(path: str = "data/dataset.jsonl"):
    """Crea un registro mínimo si el dataset no existe, para evitar fallos en frío."""
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

    print(f"Archivo {path} inicializado. Ejecuta 'python data/prepare.py' primero.")


# ----------------------------------------------------------------------
# 1. Escritura del corpus de entrenamiento — sin cargar todo en RAM
# ----------------------------------------------------------------------

def escribir_corpus(dataset_path: str, corpus_path: str) -> int:
    """
    Convierte dataset.jsonl → texto_entrenamiento.txt línea a línea.
    Evita construir un string gigante en memoria con join().
    Devuelve el número de fragmentos escritos.
    """
    total = 0
    with open(dataset_path, encoding="utf-8") as fin, \
         open(corpus_path, "w", encoding="utf-8") as fout:
        for linea in fin:
            try:
                texto = json.loads(linea)["text"]
                fout.write(texto + "\n")
                total += 1
            except (json.JSONDecodeError, KeyError):
                continue
    return total


# ----------------------------------------------------------------------
# 2. Construcción y entrenamiento del tokenizer
# ----------------------------------------------------------------------

# Rutas
DATASET_PATH = "data/dataset.jsonl"
CORPUS_PATH  = "data/texto_entrenamiento.txt"
OUTPUT_PATH  = "models/tokenizer.json"

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

ensure_dataset_exists(DATASET_PATH)

print("Escribiendo corpus de entrenamiento...")
n_fragmentos = escribir_corpus(DATASET_PATH, CORPUS_PATH)
print(f"   {n_fragmentos:,} fragmentos listos para tokenizar")

# Tokens especiales con propósito documentado:
#   <pad>  — relleno para igualar longitud en batches
#   <unk>  — token desconocido (fallback de BPE)
#   <bos>  — inicio de secuencia (usado en train.py y generate.py)
#   <eos>  — fin de secuencia (usado en train.py y generate.py)
#   <sep>  — separador entre documentos concatenados (reservado)
#   <code> — marca inicio de bloque de código (reservado para fine-tuning)
TOKENS_ESPECIALES = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<code>"]

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer    = NFKC()
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder       = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size     = 32000,
    min_frequency  = 10,
    special_tokens = TOKENS_ESPECIALES,
    show_progress  = True,
)

print("Entrenando tokenizer BPE...")
tokenizer.train(files=[CORPUS_PATH], trainer=trainer)
vocab_size_real = tokenizer.get_vocab_size()
print(f"Tokenizer entrenado | Vocabulario final: {vocab_size_real:,} tokens")

# Nota: el vocabulario real puede ser menor que vocab_size=32000 si el corpus
# no tiene suficientes pares con min_frequency >= 10. Es normal.

# ----------------------------------------------------------------------
# 3. Guardado
# ----------------------------------------------------------------------

tokenizer.save(OUTPUT_PATH)
print(f"Tokenizer exportado a {OUTPUT_PATH}")

# ----------------------------------------------------------------------
# 4. Verificación
# ----------------------------------------------------------------------

print("\n" + "-" * 50)
print("Verificación de segmentación")
print("-" * 50)

pruebas = [
    # Código
    "function login(user, password)",
    "def calcular(a, b):",
    "import torch",
    "const usuario = await db.findOne({ id })",
    # Español
    "El aprendizaje automático",
    "La inteligencia artificial es fascinante",
    # Caracteres especiales y tildes
    "implementación, función, árbol, índice",
]

for frase in pruebas:
    encoded = tokenizer.encode(frase)
    print(f"\nTexto:    '{frase}'")
    print(f"Tokens:   {encoded.tokens}")
    print(f"Total:    {len(encoded.tokens)} tokens")
    # Verificar que el decoder reconstruye el texto original
    reconstruido = tokenizer.decode(encoded.ids)
    ok = "✓" if reconstruido.strip() == frase.strip() else "✗ DIFERENCIA"
    print(f"Decode:   '{reconstruido.strip()}' {ok}")

print("\n" + "-" * 50)
vocab = tokenizer.get_vocab()
print("IDs de tokens especiales:")
for t in TOKENS_ESPECIALES:
    print(f"   {t:<8} → ID {vocab[t]}")

# Verificación de consistencia con train.py
print("\nVerificación de orden de tokens especiales:")
esperados = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
todo_ok = True
for token, id_esperado in esperados.items():
    id_real = vocab.get(token, -1)
    estado  = "✓" if id_real == id_esperado else f"✗ (esperado {id_esperado}, obtenido {id_real})"
    print(f"   {token:<8} → {estado}")
    if id_real != id_esperado:
        todo_ok = False

if not todo_ok:
    print("\n⚠ Los IDs no coinciden con lo esperado por train.py.")
    print("  Revisa el orden de TOKENS_ESPECIALES o ajusta train.py.")
else:
    print("\n✓ IDs de tokens especiales consistentes con train.py.")