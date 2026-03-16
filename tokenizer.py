from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os

# ── 1. DATOS DE ENTRENAMIENTO ──────────────────────────────────────────────────

texto_entrenamiento = """
El lenguaje es la capacidad más fascinante del ser humano.
Las redes neuronales aprenden patrones en los datos.
Un modelo de lenguaje predice la siguiente palabra dado un contexto.
La inteligencia artificial transforma la manera en que procesamos información.
El aprendizaje profundo usa múltiples capas para extraer características.
Los transformers revolucionaron el procesamiento del lenguaje natural.
Cada palabra tiene un significado que depende de su contexto.
El modelo aprende representaciones vectoriales de las palabras.
La atención es el mecanismo central de los transformers modernos.
Python es el lenguaje más usado en ciencia de datos e inteligencia artificial.
"""

# Guardamos el texto en un archivo (el trainer de BPE necesita archivos)
os.makedirs("data", exist_ok=True)
with open("data/texto_entrenamiento.txt", "w", encoding="utf-8") as f:
    f.write(texto_entrenamiento)

print("Archivo de entrenamiento creado")

# ── 2. CONSTRUIR Y ENTRENAR EL TOKENIZER ──────────────────────────────────────

# Tokens especiales
# <pad> → relleno para que todas las secuencias tengan el mismo largo
# <unk> → token desconocido (palabras que no están en el vocabulario)
# <bos> → beginning of sequence (inicio de texto)
# <eos> → end of sequence (fin de texto)
tokens_especiales = ["<pad>", "<unk>", "<bos>", "<eos>"]

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()  # divide por espacios primero

trainer = BpeTrainer(
    vocab_size=500,          # vocabulario pequeño para este ejemplo
    min_frequency=1,         # incluir pares que aparezcan al menos 1 vez
    special_tokens=tokens_especiales,
    show_progress=True
)

print("Entrenando tokenizer BPE...")
tokenizer.train(files=["data/texto_entrenamiento.txt"], trainer=trainer)
print(f"Tokenizer entrenado — vocabulario: {tokenizer.get_vocab_size()} tokens")

# ── 3. GUARDAR EL TOKENIZER ────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
tokenizer.save("models/tokenizer.json")
print("Tokenizer guardado en models/tokenizer.json")

# ── 4. FUNCIONAMIEMTO ─────────────────────────────────────────────────
print("\n" + "="*55)
print("EXPLORANDO EL TOKENIZER")
print("="*55)

frases = [
    "El modelo aprende del contexto",
    "inteligencia artificial",
    "transformers",
    "xkqz",   # palabra inventada
]

for frase in frases:
    encoded = tokenizer.encode(frase)
    decoded = tokenizer.decode(encoded.ids)

    print(f"\nTexto:   '{frase}'")
    print(f"Tokens:  {encoded.tokens}")
    print(f"IDs:     {encoded.ids}")
    print(f"Decoded: '{decoded}'")

# ── 5. TOKENS ESPECIALES
vocab = tokenizer.get_vocab()
print("\n" + "="*55)
print("IDs de tokens especiales:")
for t in tokens_especiales:
    print(f"   {t:6s} → ID {vocab[t]}")