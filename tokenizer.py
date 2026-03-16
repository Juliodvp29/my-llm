from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
import json
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# --- 0. Inicialización de Dataset ---
# Crea un registro de ejemplo si el archivo no existe para evitar fallos iniciales.

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

    print(f"Archivo {path} inicializado. Ejecutar 'python data/prepare.py' para generar dataset completo.")


# --- 1. Lectura de Dataset ---
ensure_dataset_exists()

os.makedirs("data", exist_ok=True)

print("Leyendo dataset...")
textos = []
with open("data/dataset.jsonl", encoding="utf-8") as f:
    for linea in f:
        textos.append(json.loads(linea)["text"])

with open("data/texto_entrenamiento.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(textos))

print(f"   {len(textos):,} fragmentos listos para tokenizar")

# --- 2. Construcción y Entrenamiento ---
tokens_especiales = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<sep>",
    "<code>"
]

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=tokens_especiales,
    show_progress=True
)

print("Entrenando tokenizer BPE...")
tokenizer.train(files=["data/texto_entrenamiento.txt"], trainer=trainer)
print(f"Tokenizer entrenado | Vocabulario: {tokenizer.get_vocab_size()} tokens")

# --- 3. Guardado ---
os.makedirs("models", exist_ok=True)
tokenizer.save("models/tokenizer.json")
print("Tokenizer exportado a models/tokenizer.json")

# --- 4. Verificación de Tokenización ---
print("\n" + "-"*50)
print("Verificación de segmentación")
print("-"*50)

pruebas = [
    "function login(user, password)",
    "El aprendizaje automático",
    "import torch",
    "def calcular(a, b):",
]

for frase in pruebas:
    encoded = tokenizer.encode(frase)
    print(f"\nTexto:  '{frase}'")
    print(f"Tokens: {encoded.tokens}")
    print(f"Total:  {len(encoded.tokens)} tokens")

vocab = tokenizer.get_vocab()
print("\nIDs de tokens especiales:")
for t in tokens_especiales:
    print(f"   {t:6s} → ID {vocab[t]}")