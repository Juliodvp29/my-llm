import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from model.embeddings import TransformerEmbedding
from model.attention import MultiHeadAttention, FeedForward


#  Transformer

class TransformerBlock(nn.Module):
    """
    Componente fundamental del decoder.
    d_model: Dimensión intrínseca.
    n_heads: Número de cabezas de atención.
    d_ff: Dimensión oculta del módulo feed-forward.
    dropout: Probabilidad de regularización.
    """

    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention  = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Una LayerNorm antes de cada sub-capa (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        # Atención y conexión residual con Pre-LN
        attn_out, _ = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # Feed-forward y conexión residual con Pre-LN
        x = x + self.dropout(self.feed_forward(self.norm2(x)))

        return x


# Modelo de Lenguaje Base, Arquitectura decoder-only generativa de tipo escalar causal.

class MiniGPT(nn.Module):
    """
    Implementación del modelo de lenguaje auto-regresivo (Decoder-only Transformer).
    """

    def __init__(
        self,
        vocab_size : int,
        d_model    : int = 128,
        n_heads    : int = 4,
        n_layers   : int = 4,
        d_ff       : int = 512,
        max_len    : int = 256,
        dropout    : float = 0.1,
    ):
        super().__init__()

        # 1. Crear todas las capas primero
        self.embedding = TransformerEmbedding(
            vocab_size, d_model, max_len, dropout
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # 2. Inicializar pesos
        self._init_weights()

        # 3. Weight tying AL FINAL
        self.head.weight = self.embedding.token_emb.embedding.weight

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int,
                          device: torch.device) -> torch.Tensor:
        """Genera máscara causal (triangular inferior) para preservar contexto condicional autoregresivo."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagación forward para el cómputo de logits secuenciales."""
        seq_len = x.size(1)
        mask = self._make_causal_mask(seq_len, x.device)

        # Embedding + posición
        out = self.embedding(x)

        # Pasar por cada bloque Transformer
        for block in self.blocks:
            if getattr(self, "gradient_checkpointing", False) and self.training:
                # El checkpointing re-ejecuta el forward del bloque durante el backward
                # para ahorrar guardar las activaciones en memoria.
                out = checkpoint(block, out, mask, use_reentrant=False)
            else:
                out = block(out, mask)

        # Normalización final
        out = self.norm(out)

        # Proyección a vocabulario → logits
        logits = self.head(out)

        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 40, eos_id: int = None) -> torch.Tensor:
        """Rutina heurística generativa condicionada a un prompt inicial (input_ids)."""
        self.eval()
        # Usamos dinámicamente el tamaño de memoria máximo que tenga instanciado el modelo
        max_context = self.embedding.pos_enc.pe.size(1)
        
        for _ in range(max_new_tokens):
            # Recortar SOLAMENTE si excede el abismal límite del modelo (¡no un 256 harcodeado!)
            ctx = input_ids[:, -max_context:]

            # Forward pass
            logits = self(ctx)

            # Tomamos solo los logits del ÚLTIMO token
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Top-k: ponemos -inf a todo lo que no sea top-k
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = float('-inf')

            # Softmax → probabilidades → muestreamos el siguiente token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Concatenamos el nuevo token a la secuencia
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # ¡Si el modelo nos regala el <eos>, rompemos el bucle infinito y ahorramos a la GPU!
            if eos_id is not None and next_token.item() == eos_id:
                break

        return input_ids


# Ejecución de Pruebas Integradas

if __name__ == "__main__":
    print("Prueba de integración de MiniGPT...\n")

    # Configuración pequeña — corre bien en CPU
    config = dict(
        vocab_size = 264,
        d_model    = 128,
        n_heads    = 4,
        n_layers   = 4,
        d_ff       = 512,
        max_len    = 256,
        dropout    = 0.0,
    )

    model = MiniGPT(**config)

    # ── Conteo de parámetros ───────────────────────────────────────────
    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales:      {total:,}")
    print(f"Parámetros entrenables:  {train:,}")

    # ── Forward pass ──────────────────────────────────────────────────
    BATCH   = 2
    SEQ_LEN = 16
    tokens = torch.randint(0, 264, (BATCH, SEQ_LEN))
    logits = model(tokens)

    print(f"\nEntrada:  {tokens.shape}   (batch=2, seq=16)")
    print(f"Logits:   {logits.shape}  (batch=2, seq=16, vocab=264)")
    print(f"\nPara cada token, el modelo produce 264 logits.")
    print(f"El token con el logit más alto es la predicción.")

    # ── Desglose por capa ─────────────────────────────────────────────
    print(f"\nDesglose de parámetros:")
    emb_p   = sum(p.numel() for p in model.embedding.parameters())
    block_p = sum(p.numel() for p in model.blocks.parameters())
    head_p  = sum(p.numel() for p in model.head.parameters())
    print(f"  Embedding:  {emb_p:>8,}")
    print(f"  Bloques×4:  {block_p:>8,}  ({block_p//4:,} por bloque)")
    print(f"  Head:       {head_p:>8,}  (compartido con embedding)")

    # --- Generación de Prueba ---
    print(f"\n{'-'*45}")
    print("Prueba de generación condicional (inferencia base):")
    prompt = torch.tensor([[2, 67, 124]])
    output = model.generate(prompt, max_new_tokens=8, temperature=1.0)
    print(f"IDs resultantes: {output[0].tolist()}")
    
    print("\nVerificación de módulo transformer.py completada")