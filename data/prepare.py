import os
import json
import re
import urllib.request
from pathlib import Path
import urllib.parse
import unicodedata
import time
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

if os.name == 'nt':
    import subprocess
    try:
        subprocess.run('chcp 65001', shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        pass

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def configurar_logger():
    logger = logging.getLogger("dataset-builder")
    logger.setLevel(logging.DEBUG)
    fmt_console = logging.Formatter('[%(asctime)s] %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
    fmt_file    = logging.Formatter('[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1))
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)
    logger.addHandler(ch)
    fh = logging.FileHandler('data/dataset-builder.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)
    return logger

logger = configurar_logger()

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
OUTPUT_FILE   = "data/dataset.jsonl"
CACHE_DIR     = "data/raw"               # caché por fuente — sobrevive interrupciones
os.makedirs("data", exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

from data.sources import IDS_LIBROS_GUTENBERG, ARTICULOS_WIKIPEDIA

RUTAS_REPOSITORIOS = [
    r"C:\Users\julio\PycharmProjects\my-llm",
    r"C:\Users\julio\Documents\Dev\typescript\fastapi",
    r"C:\Users\julio\Documents\Dev\typescript\angular-docs-es",
    r"C:\Users\julio\Documents\Dev\typescript\axolotl",
    r"C:\Users\julio\Documents\Dev\typescript\bulletproof-react",
    r"C:\Users\julio\Documents\Dev\typescript\DeepSpeed",
    r"C:\Users\julio\Documents\Dev\typescript\md-preview-extension",
    r"C:\Users\julio\Documents\Dev\typescript\pytorch-lightning",
    r"C:\Users\julio\Documents\Dev\typescript\requests",
    r"C:\Users\julio\Documents\Dev\typescript\RustPython",
    r"C:\Users\julio\Documents\Dev\typescript\transformers",
    r"C:\Users\julio\Documents\Dev\Angular\venti-multi-tenant",
    r"C:\Users\julio\Documents\Dev\Angular\lily-estetic-web",
    r"C:\Users\julio\Documents\Dev\Angular\srest-system",
    r"C:\Users\julio\Documents\Dev\Angular\ecotrace",
    r"C:\Users\julio\Documents\Dev\astro\my-portfolio",
    r"C:\Users\julio\Documents\Dev\Bun\ecotrace-api",
    r"C:\Users\julio\Documents\Dev\ionic\lingo",
    r"C:\Users\julio\Documents\Dev\monorepo\snapbuy",
    r"C:\Users\julio\Documents\Dev\monorepo\search-engine",
    r"C:\Users\julio\Documents\Dev\Python\MusicPlayer",
    r"C:\Users\julio\Documents\Dev\rust\rust-raytracer",
    r"C:\Users\julio\Documents\Dev\typescript\sqlmodel",
    r"C:\Users\julio\Documents\Dev\typescript\pydantic",
    r"C:\Users\julio\Documents\Dev\typescript\prisma",
    r"C:\Users\julio\Documents\Dev\typescript\book",
    r"C:\Users\julio\Documents\Dev\typescript\vite"
]

MAX_CODIGO_FRAGMENTS = 250000
# Tamaño máximo de fragmento en caracteres — alineado con max_len=512 tokens
# ~4 chars/token en español → 512 * 4 = ~2048 chars de techo seguro
MAX_FRAGMENT_CHARS = 1800


# ----------------------------------------------------------------------
# Utilidades de limpieza — DOS versiones: texto natural vs código
# ----------------------------------------------------------------------

def limpiar_texto_natural(text: str) -> str:
    """Para Wikipedia y libros: normaliza espacios, elimina control chars."""
    text = unicodedata.normalize("NFC", text)
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n ')
    text = re.sub(r'\n{3,}', '\n\n', text)      # máximo doble salto
    text = re.sub(r'[ \t]+', ' ', text)          # colapsa espacios/tabs pero NO saltos
    text = re.sub(r' *\n *', '\n', text)         # limpia espacios alrededor de saltos
    return text.strip()

def limpiar_codigo(text: str) -> str:
    """Para código fuente: preserva indentación y saltos de línea reales."""
    text = unicodedata.normalize("NFC", text)
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\t ')
    text = re.sub(r'\n{5,}', '\n\n\n', text)    # máximo triple salto vacío
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)  # trailing whitespace por línea
    return text.strip()


# ----------------------------------------------------------------------
# Fragmentación inteligente por párrafos
# ----------------------------------------------------------------------

def fragmentar_por_parrafos(texto: str, max_chars: int = MAX_FRAGMENT_CHARS, min_chars: int = 150) -> list[str]:
    """
    Divide texto respetando párrafos (doble salto de línea).
    No corta oraciones a la mitad. Si un párrafo individual es más largo
    que max_chars, lo subdivide por oraciones.
    """
    parrafos = re.split(r'\n{2,}', texto)
    fragmentos = []
    buffer = ""

    for parrafo in parrafos:
        parrafo = parrafo.strip()
        if not parrafo:
            continue

        # Si el párrafo solo ya es demasiado largo, subdividir por oraciones
        if len(parrafo) > max_chars:
            oraciones = re.split(r'(?<=[.!?])\s+', parrafo)
            sub_buffer = ""
            for oracion in oraciones:
                if len(sub_buffer) + len(oracion) + 1 > max_chars:
                    if len(sub_buffer) >= min_chars:
                        fragmentos.append(sub_buffer.strip())
                    sub_buffer = oracion
                else:
                    sub_buffer = (sub_buffer + " " + oracion).strip()
            if len(sub_buffer) >= min_chars:
                fragmentos.append(sub_buffer.strip())
            continue

        # Si agregar este párrafo supera el límite, vuelca el buffer
        if len(buffer) + len(parrafo) + 2 > max_chars:
            if len(buffer) >= min_chars:
                fragmentos.append(buffer.strip())
            buffer = parrafo
        else:
            buffer = (buffer + "\n\n" + parrafo).strip() if buffer else parrafo

    if len(buffer) >= min_chars:
        fragmentos.append(buffer.strip())

    return fragmentos


def fragmentar_codigo(texto: str, max_chars: int = MAX_FRAGMENT_CHARS, min_chars: int = 100) -> list[str]:
    """
    Divide código respetando bloques de funciones/clases (líneas en blanco).
    Preserva saltos de línea reales.
    """
    lineas = texto.split('\n')
    fragmentos = []
    buffer_lineas = []

    for linea in lineas:
        buffer_lineas.append(linea)
        contenido_buffer = '\n'.join(buffer_lineas)

        if len(contenido_buffer) >= max_chars:
            if len(contenido_buffer) >= min_chars:
                fragmentos.append(contenido_buffer)
            buffer_lineas = []

    resto = '\n'.join(buffer_lineas).strip()
    if len(resto) >= min_chars:
        fragmentos.append(resto)

    return fragmentos


# ----------------------------------------------------------------------
# Escritura con caché — cada fuente tiene su propio archivo .jsonl en raw/
# ----------------------------------------------------------------------

def guardar_en_cache(fragmentos: list[str], nombre_cache: str):
    """Guarda fragmentos en data/raw/<nombre>.jsonl (modo append)."""
    if not fragmentos:
        return
    ruta = os.path.join(CACHE_DIR, f"{nombre_cache}.jsonl")
    with open(ruta, 'a', encoding='utf-8') as f:
        for texto in fragmentos:
            if len(texto.strip()) > 50:
                f.write(json.dumps({"text": texto}, ensure_ascii=False) + '\n')


def cache_existe(nombre_cache: str) -> bool:
    ruta = os.path.join(CACHE_DIR, f"{nombre_cache}.jsonl")
    return os.path.exists(ruta) and os.path.getsize(ruta) > 0


def consolidar_cache_en_output():
    """Une todos los .jsonl de data/raw/ en data/dataset.jsonl final."""
    logger.info("Consolidando caché en dataset final...")
    total = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        for archivo in sorted(Path(CACHE_DIR).glob("*.jsonl")):
            with open(archivo, 'r', encoding='utf-8') as f:
                for linea in f:
                    out.write(linea)
                    total += 1
    logger.info(f"Dataset consolidado: {total:,} fragmentos → {OUTPUT_FILE}")
    return total


# ----------------------------------------------------------------------
# 1. Wikipedia con threads — con rate limiting real en los workers
# ----------------------------------------------------------------------

def fetch_wikipedia_article(title: str) -> list[str] | None:
    titulo_decodificado = urllib.parse.unquote(title)
    url_title = urllib.parse.quote(titulo_decodificado)
    url = (
        f"https://es.wikipedia.org/w/api.php"
        f"?action=query&prop=extracts&explaintext=1&format=json&titles={url_title}"
    )
    headers = {'User-Agent': 'MiLLMDataBuilder/2.0 (aprendizaje, sin fines comerciales)'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            pages = data['query']['pages']
            for page_id, info in pages.items():
                if page_id == '-1' or 'extract' not in info:
                    return None
                texto = limpiar_texto_natural(info['extract'])
                return fragmentar_por_parrafos(texto)
    except Exception as e:
        logger.debug(f"Error Wiki ({title}): {e}")
        return None


def procesar_wikipedia_con_hilos():
    NOMBRE_CACHE = "wikipedia"

    if cache_existe(NOMBRE_CACHE):
        # Contar cuántos ya hay para no rehacer el trabajo
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        n = sum(1 for _ in open(ruta, encoding='utf-8'))
        logger.info(f"Wikipedia: caché encontrada ({n:,} fragmentos). Saltando descarga.")
        return n

    logger.info("Iniciando extracción de Wikipedia (multihilo)...")
    total_frag = 0
    buffer = []
    exitos = 0

    # Delay entre requests dentro del worker — respeta la API de Wikipedia
    def fetch_con_delay(title):
        time.sleep(0.2)   # 200ms entre requests por hilo → ~5 req/s con 5 hilos
        return fetch_wikipedia_article(title)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_con_delay, t): t for t in ARTICULOS_WIKIPEDIA}
        for idx, future in enumerate(as_completed(futures), 1):
            try:
                frags = future.result()
                if frags:
                    buffer.extend(frags)
                    total_frag += len(frags)
                    exitos += 1
            except Exception:
                pass

            if len(buffer) >= 100 or idx == len(ARTICULOS_WIKIPEDIA):
                guardar_en_cache(buffer, NOMBRE_CACHE)
                buffer = []

            if idx % 25 == 0:
                logger.info(f"  [Wiki] {idx}/{len(ARTICULOS_WIKIPEDIA)} artículos. Fragmentos: {total_frag:,}")

    logger.info(f"Wikipedia: {exitos} artículos exitosos, {total_frag:,} fragmentos.")
    return total_frag


# ----------------------------------------------------------------------
# 2. Repositorios locales — fragmentos alineados con max_len del modelo
# ----------------------------------------------------------------------

def directorio_valido(filepath: str) -> bool:
    ruta = filepath.lower().replace('\\', '/')
    excluir = [
        'node_modules', '.git', 'venv', 'env', '__pycache__',
        '.idea', '.vscode', 'dist', 'build', 'out', 'target',
        'coverage', '.next', '.nuxt', '.angular', 'tmp', 'cache',
        '.pytest_cache', 'migrations', 'static/vendor',
    ]
    return not any(exc in ruta for exc in excluir)


def procesar_repositorios_locales():
    NOMBRE_CACHE = "codigo_local"

    if cache_existe(NOMBRE_CACHE):
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        n = sum(1 for _ in open(ruta, encoding='utf-8'))
        logger.info(f"Código local: caché encontrada ({n:,} fragmentos). Saltando escaneo.")
        return n

    logger.info(f"Escaneando repositorios locales (límite: {MAX_CODIGO_FRAGMENTS:,} fragmentos)...")
    extensiones_validas = {'.py', '.js', '.ts', '.html', '.css', '.java', '.cpp', '.c', '.go', '.rs', '.md', '.json', '.rs', '.r', 'tsx', 'jsx', '.kt'}
    total_frag = 0
    buffer = []
    archivos_procesados = 0

    for ruta_base in RUTAS_REPOSITORIOS:
        if not os.path.exists(ruta_base):
            logger.warning(f"  Ruta no encontrada: {ruta_base}")
            continue

        logger.info(f"  Escaneando: {ruta_base}")

        for root, dirs, files in os.walk(ruta_base):
            # Modificar dirs in-place para que os.walk no descienda en carpetas excluidas
            dirs[:] = [d for d in dirs if directorio_valido(os.path.join(root, d))]

            if total_frag >= MAX_CODIGO_FRAGMENTS:
                break

            for file in files:
                if total_frag >= MAX_CODIGO_FRAGMENTS:
                    break

                ext = os.path.splitext(file)[1].lower()
                if ext not in extensiones_validas:
                    continue
                if '.min.' in file or 'bundle' in file or 'lock' in file:
                    continue

                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        contenido = f.read()

                    # Descartar archivos triviales o gigantes
                    if not (50 < len(contenido) < 80_000):
                        continue
                    if contenido.count('\n') < 5:
                        continue

                    contenido_limpio = limpiar_codigo(contenido)

                    # Prefijo descriptivo — le dice al modelo el contexto
                    prefijo = f"# Archivo: {file} (extensión: {ext})\n\n"
                    texto_con_prefijo = prefijo + contenido_limpio

                    frags = fragmentar_codigo(texto_con_prefijo)
                    buffer.extend(frags)
                    total_frag += len(frags)
                    archivos_procesados += 1

                except Exception:
                    pass

            if len(buffer) >= 200:
                guardar_en_cache(buffer, NOMBRE_CACHE)
                buffer = []

        if total_frag >= MAX_CODIGO_FRAGMENTS:
            logger.info(f"  ⚠️ Límite de {MAX_CODIGO_FRAGMENTS:,} fragmentos alcanzado.")
            break

    guardar_en_cache(buffer, NOMBRE_CACHE)
    logger.info(f"Código local: {archivos_procesados:,} archivos, {total_frag:,} fragmentos.")
    return total_frag


# ----------------------------------------------------------------------
# 3. Gutenberg — con caché por libro y marcadores correctos
# ----------------------------------------------------------------------

def descargar_parsear_gutenberg(book_id: int) -> list[str] | None:
    # Gutenberg tiene mirrors — intentamos el principal y uno de respaldo
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://gutenberg.org/files/{book_id}/{book_id}-0.txt",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                texto = response.read().decode('utf-8', errors='ignore')

            # Eliminar encabezado legal de Gutenberg
            # El marcador real termina con *** seguido de doble salto de línea
            for patron_inicio in [
                "*** START OF THE PROJECT GUTENBERG EBOOK",
                "*** START OF THIS PROJECT GUTENBERG EBOOK",
            ]:
                idx = texto.find(patron_inicio)
                if idx != -1:
                    # Buscar el fin de esa línea (el *** de cierre) y el primer párrafo
                    resto = texto[idx:]
                    fin_marcador = resto.find('\n\n')
                    if fin_marcador != -1:
                        texto = resto[fin_marcador:].strip()
                    break

            for patron_fin in [
                "*** END OF THE PROJECT GUTENBERG EBOOK",
                "*** END OF THIS PROJECT GUTENBERG EBOOK",
            ]:
                idx = texto.find(patron_fin)
                if idx != -1:
                    texto = texto[:idx].strip()
                    break

            texto_limpio = limpiar_texto_natural(texto)

            # Fragmentar respetando párrafos
            fragmentos = fragmentar_por_parrafos(texto_limpio)
            if fragmentos:
                return fragmentos

        except Exception as e:
            logger.debug(f"  Error libro {book_id} desde {url}: {e}")
            continue

    return None


def procesar_gutenberg():
    NOMBRE_CACHE = "gutenberg"

    # Caché granular por libro — permite reanudar si se interrumpe
    cache_libros_path = os.path.join(CACHE_DIR, "gutenberg_descargados.json")
    if os.path.exists(cache_libros_path):
        with open(cache_libros_path, 'r') as f:
            libros_descargados: set = set(json.load(f))
    else:
        libros_descargados: set = set()

    pendientes = [bid for bid in IDS_LIBROS_GUTENBERG if bid not in libros_descargados]

    if not pendientes:
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        n = sum(1 for _ in open(ruta, encoding='utf-8')) if os.path.exists(ruta) else 0
        logger.info(f"Gutenberg: todos los libros ya en caché ({n:,} fragmentos). Saltando descarga.")
        return n

    logger.info(f"Descargando libros de Gutenberg ({len(pendientes)} pendientes de {len(IDS_LIBROS_GUTENBERG)})...")
    total_frag = 0
    buffer = []

    for i, book_id in enumerate(pendientes, 1):
        fragmentos = descargar_parsear_gutenberg(book_id)
        if fragmentos:
            buffer.extend(fragmentos)
            total_frag += len(fragmentos)
            libros_descargados.add(book_id)

            # Guardamos el progreso de qué libros están listos
            with open(cache_libros_path, 'w') as f:
                json.dump(list(libros_descargados), f)

        if len(buffer) >= 50 or i == len(pendientes):
            guardar_en_cache(buffer, NOMBRE_CACHE)
            buffer = []
            logger.info(f"  [Libros] {i}/{len(pendientes)} procesados. Fragmentos: {total_frag:,}")

        time.sleep(1.0)  # Rate limit con Gutenberg — no cambiar

    logger.info(f"Gutenberg: {total_frag:,} fragmentos nuevos.")
    return total_frag


# ----------------------------------------------------------------------
# Ejecución principal
# ----------------------------------------------------------------------

def generar_dataset_completo():
    inicio = time.time()
    logger.info("=" * 60)
    logger.info("INICIANDO GENERADOR DE DATASET LLM")
    logger.info("=" * 60)
    logger.info(f"Caché en: {os.path.abspath(CACHE_DIR)}")
    logger.info(f"Salida en: {os.path.abspath(OUTPUT_FILE)}")
    logger.info("Si interrumpes y reinicias, el progreso se conserva.")
    logger.info("=" * 60)

    frag_wiki = frag_codigo = frag_libros = 0

    try:
        frag_wiki   = procesar_wikipedia_con_hilos()
        logger.info("-" * 60)
        frag_codigo = procesar_repositorios_locales()
        logger.info("-" * 60)
        frag_libros = procesar_gutenberg()
        logger.info("-" * 60)

    except KeyboardInterrupt:
        logger.warning("Proceso interrumpido. El progreso hasta aquí está guardado en data/raw/")
        logger.warning("Puedes reanudar ejecutando el script de nuevo.")

    finally:
        total = consolidar_cache_en_output()
        tiempo = time.time() - inicio
        logger.info("=" * 60)
        logger.info("RESUMEN FINAL")
        logger.info(f"  Wikipedia:     {frag_wiki:,} fragmentos")
        logger.info(f"  Código local:  {frag_codigo:,} fragmentos")
        logger.info(f"  Libros:        {frag_libros:,} fragmentos")
        logger.info(f"  TOTAL:         {total:,} fragmentos en dataset.jsonl")
        logger.info(f"  Tiempo total:  {tiempo:.1f}s ({tiempo/60:.1f} min)")
        logger.info("=" * 60)


if __name__ == "__main__":
    generar_dataset_completo()