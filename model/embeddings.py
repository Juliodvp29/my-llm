import torch
import torch.nn as nn
import math


# ══════════════════════════════════════════════════════════════════════
#  EMBEDDINGS
#
#  Asocian cada token ID con un vector de dimensión d_model.
#  La tabla de embeddings (nn.Embedding) es un conjunto de parámetros
#  entrenables que el modelo ajusta para representar similitudes semánticas.
# ══════════════════════════════════════════════════════════════════════

class TokenEmbedding(nn.Module):
    """
    Convierte una secuencia de IDs de tokens en vectores densos.

    Parámetros:
        vocab_size : cuántos tokens distintos existen (tamaño del vocabulario)
        d_model    : dimensión de cada vector (cuántos números por token)
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        # nn.Embedding es básicamente una tabla de búsqueda:
        # fila i = vector del token con ID i
        # Tiene vocab_size filas y d_model columnas
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor de IDs  → shape (batch_size, seq_len)
        salida            → shape (batch_size, seq_len, d_model)
        """
        # Multiplicamos por sqrt(d_model) — truco estándar de los Transformers
        # que estabiliza los valores al inicio del entrenamiento.
        return self.embedding(x) * math.sqrt(self.d_model)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
#
#  Agrega una señal de posición (sin/cos) a los embeddings para que el
#  modelo pueda distinguir el orden de los tokens sin aprender parámetros.
#  Esta señal es determinista y se puede calcular para cualquier longitud.
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Suma una señal de posición a los embeddings para que el modelo
    sepa el orden de los tokens en la secuencia.

    Parámetros:
        d_model  : debe coincidir con TokenEmbedding
        max_len  : longitud máxima de secuencia que soporta el modelo
        dropout  : regularización (apaga neuronas al azar durante entrenamiento)
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Construimos la tabla de encodings de una sola vez
        # shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Vector de posiciones: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Factor de escala para las frecuencias
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # Dimensiones pares → seno, dimensiones impares → coseno
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Agregamos dimensión de batch: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer: guarda pe como parte del modelo pero
        # NO como parámetro entrenable (no cambia con el gradiente)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: salida de TokenEmbedding → shape (batch_size, seq_len, d_model)
        """
        # Sumamos solo las posiciones que necesitamos (hasta seq_len)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  MÓDULO COMBINADO
#  En la práctica siempre usamos ambos juntos, así que los unimos.
# ══════════════════════════════════════════════════════════════════════

class TransformerEmbedding(nn.Module):
    """
    Embedding completo = TokenEmbedding + PositionalEncoding
    Es la primera capa del modelo.
    """

    def __init__(self, vocab_size: int, d_model: int,
                 max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Entrada:  IDs de tokens → (batch_size, seq_len)
        Salida:   vectores ricos → (batch_size, seq_len, d_model)
        """
        return self.pos_enc(self.token_emb(x))


# ══════════════════════════════════════════════════════════════════════
#  TEST
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Probando embeddings...\n")

    # Configuración de ejemplo
    VOCAB_SIZE = 264    # el vocabulario que construimos con el tokenizer
    D_MODEL    = 64     # cada token se representa con 64 números
    MAX_LEN    = 128
    BATCH_SIZE = 2      # procesamos 2 oraciones a la vez
    SEQ_LEN    = 10     # cada oración tiene 10 tokens

    # Creamos el módulo
    embedding = TransformerEmbedding(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_len=MAX_LEN,
        dropout=0.0     # sin dropout en evaluación
    )

    # Simulamos un batch de tokens (números aleatorios entre 0 y vocab_size)
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    print(f"Tokens de entrada:  shape {tokens.shape}")
    print(f"  → {BATCH_SIZE} oraciones, {SEQ_LEN} tokens cada una")
    print(f"  → valores entre 0 y {VOCAB_SIZE-1}")
    print(f"\nEjemplo (primera oración): {tokens[0].tolist()}")

    # Pasamos por el embedding
    output = embedding(tokens)
    print(f"\nSalida del embedding: shape {output.shape}")
    print(f"  → {BATCH_SIZE} oraciones, {SEQ_LEN} tokens, {D_MODEL} dimensiones cada uno")

    # Verificamos que dos tokens distintos tienen vectores distintos
    vec_token_0 = output[0, 0, :]   # primer token, primera oración
    vec_token_1 = output[0, 1, :]   # segundo token, primera oración
    similitud = torch.nn.functional.cosine_similarity(
        vec_token_0.unsqueeze(0),
        vec_token_1.unsqueeze(0)
    ).item()
    print(f"\nSimilitud coseno entre token[0] y token[1]: {similitud:.4f}")
    print("  (cercano a 1 = similares, cercano a 0 = diferentes)")

    # Contamos parámetros entrenables
    params = sum(p.numel() for p in embedding.parameters())
    print(f"\nParámetros entrenables en esta capa: {params:,}")
    print(f"  → la tabla de embeddings: {VOCAB_SIZE} × {D_MODEL} = {VOCAB_SIZE*D_MODEL:,}")

    print("\nembeddings.py funciona correctamente")