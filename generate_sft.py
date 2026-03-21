import torch
from tokenizers import Tokenizer
from model.transformer import MiniGPT

def cargar_modelo(checkpoint_path, tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model = MiniGPT(
        vocab_size=config["vocab_size"], d_model=config["d_model"],
        n_heads=config["n_heads"], n_layers=config["n_layers"],
        d_ff=config["d_ff"], max_len=config["max_len"], dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Modelo SFT cargado (época {config['epochs']}, val_loss {checkpoint['val_loss']:.4f})\n")
    return model, tokenizer


def chat(model, tokenizer, pregunta, max_new_tokens=150, temperature=0.5, top_k=20):
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    # El formato que aprendió durante el SFT
    prompt = f"<human>: {pregunta}\n<assistant>:"
    ids = [bos_id] + tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    nuevos = output_ids[0, len(ids):].tolist()
    if eos_id in nuevos:
        nuevos = nuevos[:nuevos.index(eos_id)]

    return tokenizer.decode(nuevos).strip()


if __name__ == "__main__":
    model, tokenizer = cargar_modelo(
        checkpoint_path="models/checkpoints_sft/sft_best_model.pt",
        tokenizer_path="models/tokenizer.json",
    )

    # Pruebas automáticas
    preguntas = [
        "¿Qué es Python?",
        "¿Qué es el overfitting?",
        "¿Cuál es la capital de Colombia?",
        "Hola, ¿cómo estás?",
        "¿Qué es un transformer en IA?",
        "def fibonacci(n):",
    ]

    print("=" * 60)
    print("TEST DEL MODELO SFT")
    print("=" * 60)

    for pregunta in preguntas:
        respuesta = chat(model, tokenizer, pregunta)
        print(f"Pregunta: {pregunta}")
        print(f"Respuesta: {respuesta}")
        print("-" * 40)

    # Modo interactivo
    print("\nModo chat interactivo (escribe 'salir' para terminar)\n")
    while True:
        pregunta = input("Tú: ").strip()
        if pregunta.lower() in ("salir", "exit", "q"):
            break
        if not pregunta:
            continue
        respuesta = chat(model, tokenizer, pregunta)
        print(f"IA: {respuesta}\n")