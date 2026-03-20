lineas_unicas = []
vistas = set()

with open("data/sft_dataset.jsonl", "r", encoding="utf-8") as f:
    for linea in f:
        linea = linea.strip()
        if linea and linea not in vistas:
            vistas.add(linea)
            lineas_unicas.append(linea)

with open("data/sft_dataset.jsonl", "w", encoding="utf-8") as f:
    for linea in lineas_unicas:
        f.write(linea + "\n")

print(f"Total ejemplos únicos: {len(lineas_unicas)}")