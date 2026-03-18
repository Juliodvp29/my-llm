import torch
from tokenizers import Tokenizer
from model.transformer import MiniGPT

# Carga de Modelo Pre-entrenado

def cargar_modelo(checkpoint_path: str, tokenizer_path: str):
    tokenizer = Tokenizer.from_file(tokenizer_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    vocab_tokenizer = tokenizer.get_vocab_size()
    vocab_modelo    = config["vocab_size"]
    if vocab_tokenizer != vocab_modelo:
        raise ValueError(
            f"Incompatibilidad: tokenizer tiene {vocab_tokenizer} tokens "
            f"pero el modelo fue entrenado con {vocab_modelo}. "
            f"Usa el tokenizer y checkpoint del mismo entrenamiento."
        )

    model = MiniGPT(
        vocab_size = config["vocab_size"],
        d_model    = config["d_model"],
        n_heads    = config["n_heads"],
        n_layers   = config["n_layers"],
        d_ff       = config["d_ff"],
        max_len    = config["max_len"],
        dropout    = 0.0,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Modelo cargado (Épocas: {config['epochs']}, val_loss: {checkpoint['val_loss']:.4f})\n")
    return model, tokenizer


def generar(model, tokenizer, prompt: str,
            max_new_tokens: int = 60,
            temperature: float = 0.8,
            top_k: int = 40) -> str:
    """
    Genera secuencia de tokens de forma autoregresiva.
    temperature: Controla la aleatoriedad (menor = determinista, mayor = creativo).
    top_k: Restringe el muestreo a los k tokens más probables.
    """
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    # Tokenizamos el prompt
    prompt_ids = tokenizer.encode(prompt).ids
    input_ids  = torch.tensor([[bos_id] + prompt_ids], dtype=torch.long)

    # Generamos
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    # Decodificamos solo los tokens nuevos (sin el prompt)
    nuevos_ids = output_ids[0, len(input_ids[0]):].tolist()

    # Cortamos en <eos> si aparece
    if eos_id in nuevos_ids:
        nuevos_ids = nuevos_ids[:nuevos_ids.index(eos_id)]

    return tokenizer.decode(nuevos_ids)


if __name__ == "__main__":
    model, tokenizer = cargar_modelo(
        checkpoint_path="models/checkpoints/best_model.pt",
        tokenizer_path ="models/tokenizer.json",
    )

    # --- Prueba 1: Generación de código ---
    prompts_codigo = [
        "function login(",
        "def calcular(",
        "import torch",
        "const usuario =",
    ]

    print("Generación de código\n" + "-"*50)
    for prompt in prompts_codigo:
        resultado = generar(model, tokenizer, prompt,
                           max_new_tokens=40, temperature=0.7)
        print(f"Prompt:    '{prompt}'")
        print(f"Generado:  {resultado}")
        print()

    # --- Prueba 2: Generación en texto natural ---
    prompts_texto = [
        "El aprendizaje automático",
        "Python es un lenguaje",
        "La inteligencia artificial",
    ]

    print("Generación de texto\n" + "-"*50)
    for prompt in prompts_texto:
        resultado = generar(model, tokenizer, prompt,
                           max_new_tokens=50, temperature=0.8)
        print(f"Prompt:    '{prompt}'")
        print(f"Generado:  {resultado}")
        print()

    # --- Prueba 3: Variación de temperatura ---
    prompt_test = "def entrenar("
    print("Efecto de temperatura\n" + "-"*50)
    for temp in [0.5, 0.8, 1.2]:
        resultado = generar(model, tokenizer, prompt_test,
                           max_new_tokens=30, temperature=temp)
        print(f"Temp {temp}: {resultado}")
    print()

    # --- Modo interactivo ---
    print("-"*50)
    print("Modo interactivo (ingresar 'salir' para terminar)\n")

    while True:
        prompt = input("Prompt: ").strip()
        if prompt.lower() in ("salir", "exit", "q"):
            break
        if not prompt:
            continue

        temp  = 0.8
        tokens = 60

        resultado = generar(model, tokenizer, prompt,
                           max_new_tokens=tokens, temperature=temp)
        print(f"Modelo: {resultado}\n")