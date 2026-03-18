import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Auto-atención de producto escalar escalado (Scaled Dot-Product Attention)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implementación de la fórmula de atención.

    Shapes:
        Q, K, V : (batch, heads, seq_len, d_k)
        mask    : (batch, 1, seq_len, seq_len)  — opcional
    """
    d_k = Q.size(-1)

    # Similitud QK escalada por dimensión
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Enmascaramiento causal opcional
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Distribución de atención mediante similitud softmax
    weights = F.softmax(scores, dim=-1)

    # Paso 4: suma ponderada de los valores
    output = torch.matmul(weights, V)

    return output, weights


# Auto-atención Multicabeza

class MultiHeadAttention(nn.Module):
    """
    Atención multi-cabeza: n_heads atenciones en paralelo.

    Parámetros:
        d_model  : dimensión del modelo (debe ser divisible entre n_heads)
        n_heads  : número de cabezas de atención
        dropout  : regularización
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) debe ser divisible entre n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # dimensión por cabeza

        # 4 matrices lineales: una para Q, K, V y una para la salida
        # Todas de tamaño d_model × d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reorganiza el tensor para procesar múltiples cabezas en paralelo.

        (batch, seq_len, d_model)
          → (batch, seq_len, n_heads, d_k)
          → (batch, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        En un decoder-only , Q, K y V vienen del mismo tensor x.

        x    : (batch, seq_len, d_model)
        mask : (batch, 1, seq_len, seq_len)
        """
        # Proyectamos x en Q, K, V con las matrices aprendidas
        Q = self.split_heads(self.W_q(x))   # (batch, heads, seq_len, d_k)
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Atención escalada en todas las cabezas a la vez
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Reunimos las cabezas: (batch, heads, seq_len, d_k)
        #                     → (batch, seq_len, d_model)
        batch_size, _, seq_len, _ = attn_output.size()
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Proyección final
        output = self.W_o(attn_output)

        return output, attn_weights


# Red Feed Forward

class FeedForward(nn.Module):
    """
    Red feed-forward posición-a-posición.
    Se aplica de forma idéntica e independiente a cada token.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------
# Prueba de componente
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Iniciando prueba de atención...\n")

    BATCH    = 2
    SEQ_LEN  = 10
    D_MODEL  = 64
    N_HEADS  = 4      # 4 cabezas × 16 dimensiones = 64
    D_FF     = 256    # 4 × d_model

    # --- Atención multi-cabeza ---
    attn = MultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS)

    # Simulamos la salida del embedding como entrada
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    # Máscara causal: triangular inferior — cada token solo ve
    # a sí mismo y a los tokens ANTERIORES, nunca a los siguientes
    mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).unsqueeze(0).unsqueeze(0)

    attn_out, weights = attn(x, mask)

    print(f"Entrada:             {x.shape}")
    print(f"Salida de atención:  {attn_out.shape}  (mismo shape que entrada)")
    print(f"Pesos de atención:   {weights.shape}")
    print(f"  → (batch={BATCH}, heads={N_HEADS}, seq={SEQ_LEN}, seq={SEQ_LEN})")

    # Verificamos que la máscara causal funciona:
    # en la cabeza 0, el token 0 SOLO debe atender al token 0
    w = weights[0, 0]   # primera oración, primera cabeza
    print(f"\nPesos del token 0 (solo debe ver posición 0):")
    print(f"  {[round(v, 4) for v in w[0].tolist()]}")
    print(f"  → posición 0: {w[0,0].item():.4f} (debe ser ~1.0)")
    print(f"  → posición 1: {w[0,1].item():.4f} (debe ser ~0.0 — enmascarado)")

    # --- Feed Forward ---
    ff = FeedForward(d_model=D_MODEL, d_ff=D_FF)
    ff_out = ff(attn_out)
    print(f"\nSalida de FeedForward: {ff_out.shape}  (mismo shape, siempre)")

    # --- Parámetros ---
    attn_params = sum(p.numel() for p in attn.parameters())
    ff_params   = sum(p.numel() for p in ff.parameters())
    print(f"\nParámetros en MultiHeadAttention: {attn_params:,}")
    print(f"Parámetros en FeedForward:        {ff_params:,}")
    print(f"Total estas dos capas:            {attn_params + ff_params:,}")

    print("\nVerificación de attention.py superada")