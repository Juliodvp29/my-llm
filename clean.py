import re

lineas_limpias = []
problemas = 0
correctos = 0

with open("data/sft_dataset.jsonl", "r", encoding="utf-8") as f:
    for i, linea in enumerate(f, 1):
        linea = linea.strip()
        if not linea:
            continue

        # Primero limpiamos los escapes inválidos
        # \< y \> no son válidos en JSON, quitamos la barra
        linea_limpia = re.sub(r'\\([<>!])', r'\1', linea)

        try:
            import json

            obj = json.loads(linea_limpia)
            texto = obj["text"]

            if "<human>:" in texto and "<assistant>:" in texto:
                correctos += 1
            else:
                problemas += 1
                print(f"Línea {i} sin formato correcto:")
                print(texto[:200])
                print()

            lineas_limpias.append(json.dumps(obj, ensure_ascii=False))

        except Exception as e:
            print(f"Línea {i} con error irreparable: {e}")
            print(linea[:200])
            print()

# Guardamos el archivo limpio
with open("data/sft_dataset.jsonl", "w", encoding="utf-8") as f:
    for linea in lineas_limpias:
        f.write(linea + "\n")

print(f"\nTotal procesados: {len(lineas_limpias)}")
print(f"Con formato correcto: {correctos}")
print(f"Sin formato correcto: {problemas}")