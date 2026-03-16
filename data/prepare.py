import os
import json
import re
import urllib.request
import urllib.error
import urllib.parse
import unicodedata
import time
import logging
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
    fh = logging.FileHandler('data/dataset-builder.log', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)
    return logger

logger = configurar_logger()

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
OUTPUT_FILE   = "data/dataset.jsonl"
CACHE_DIR     = "data/raw"
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

MAX_CODIGO_FRAGMENTS = 250_000
MAX_FRAGMENT_CHARS   = 1800


# ----------------------------------------------------------------------
# Utilidades de limpieza
# ----------------------------------------------------------------------

def limpiar_texto_natural(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n ')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    return text.strip()

def limpiar_codigo(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\t ')
    text = re.sub(r'\n{5,}', '\n\n\n', text)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    return text.strip()


# ----------------------------------------------------------------------
# Fragmentación
# ----------------------------------------------------------------------

def fragmentar_por_parrafos(texto: str, max_chars: int = MAX_FRAGMENT_CHARS, min_chars: int = 150) -> list[str]:
    parrafos = re.split(r'\n{2,}', texto)
    fragmentos = []
    buffer = ""

    for parrafo in parrafos:
        parrafo = parrafo.strip()
        if not parrafo:
            continue

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
    Divide código respetando bloques lógicos (funciones/clases delimitadas
    por líneas en blanco). Produce múltiples fragmentos pequeños en vez
    de acumular hasta el tope y volcar todo de golpe.
    """
    lineas = texto.split('\n')
    fragmentos = []
    buffer_lineas: list[str] = []

    for linea in lineas:
        buffer_lineas.append(linea)
        contenido = '\n'.join(buffer_lineas)

        # Umbral de volcado: cuando el buffer supera max_chars
        # Y estamos en una línea vacía (límite natural de bloque)
        if len(contenido) >= max_chars and linea.strip() == '':
            if len(contenido.strip()) >= min_chars:
                fragmentos.append(contenido.strip())
            buffer_lineas = []
            continue

        # Forzar corte si el buffer se desborda mucho (sin línea vacía cercana)
        if len(contenido) >= max_chars * 1.5:
            if len(contenido.strip()) >= min_chars:
                fragmentos.append(contenido.strip())
            buffer_lineas = []

    resto = '\n'.join(buffer_lineas).strip()
    if len(resto) >= min_chars:
        fragmentos.append(resto)

    return fragmentos


# ----------------------------------------------------------------------
# Caché
# ----------------------------------------------------------------------

def guardar_en_cache(fragmentos: list[str], nombre_cache: str):
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
    logger.info("Consolidando cache en dataset final...")
    total = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        for archivo in sorted(Path(CACHE_DIR).glob("*.jsonl")):
            with open(archivo, 'r', encoding='utf-8') as f:
                for linea in f:
                    out.write(linea)
                    total += 1
    logger.info(f"Dataset consolidado: {total:,} fragmentos -> {OUTPUT_FILE}")
    return total


# ----------------------------------------------------------------------
# Wikipedia — reintentos con backoff exponencial + URL encoding correcto
# ----------------------------------------------------------------------

def _encode_wiki_title(title: str) -> str:
    """
    Codifica correctamente el título para la API de Wikipedia.
    Preserva guiones bajos (espacios en wiki) pero encodea el resto.
    """
    # Primero decodificamos por si ya viene parcialmente encoded
    decoded = urllib.parse.unquote(title)
    # Re-encodeamos correctamente: espacios→%20, preservamos _
    # Wikipedia acepta _ como espacio, así que los preservamos
    encoded = urllib.parse.quote(decoded, safe='_:()/')
    return encoded


def fetch_wikipedia_article(title: str, max_retries: int = 3) -> list[str] | None:
    encoded_title = _encode_wiki_title(title)
    url = (
        f"https://es.wikipedia.org/w/api.php"
        f"?action=query&prop=extracts&explaintext=1"
        f"&titles={encoded_title}&format=json&redirects=1"
    )
    headers = {'User-Agent': 'MiLLMDataBuilder/2.0 (aprendizaje, sin fines comerciales)'}

    for intento in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                # Respetar Retry-After si lo devuelve el servidor
                data = json.loads(response.read().decode())
                pages = data['query']['pages']
                for page_id, info in pages.items():
                    if page_id == '-1' or 'extract' not in info:
                        return None
                    if not info['extract'].strip():
                        return None
                    texto = limpiar_texto_natural(info['extract'])
                    return fragmentar_por_parrafos(texto)

        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limit explícito — esperar más
                wait = (2 ** intento) * 5 + random.uniform(0, 2)
                logger.debug(f"  Rate limit 429 en '{title}'. Esperando {wait:.1f}s...")
                time.sleep(wait)
            elif e.code in (500, 502, 503):
                wait = (2 ** intento) * 2
                logger.debug(f"  Error {e.code} en '{title}'. Reintento {intento+1}/{max_retries}...")
                time.sleep(wait)
            else:
                logger.debug(f"  HTTP {e.code} en '{title}': {e}")
                return None

        except (urllib.error.URLError, TimeoutError, OSError) as e:
            wait = (2 ** intento) * 1.5 + random.uniform(0, 1)
            logger.debug(f"  Error de red en '{title}' (intento {intento+1}): {e}. Reintento en {wait:.1f}s...")
            time.sleep(wait)

        except Exception as e:
            logger.debug(f"  Error inesperado en '{title}': {e}")
            return None

    logger.debug(f"  Agotados {max_retries} intentos para '{title}'")
    return None


def procesar_wikipedia_con_hilos():
    NOMBRE_CACHE = "wikipedia"

    # --- Caché granular por artículo para poder reanudar ---
    cache_articulos_path = os.path.join(CACHE_DIR, "wikipedia_descargados.json")
    articulos_descargados: set = set()

    if os.path.exists(cache_articulos_path):
        try:
            with open(cache_articulos_path, 'r', encoding='utf-8') as f:
                contenido = f.read().strip()
                if contenido:
                    articulos_descargados = set(json.loads(contenido))
        except (json.JSONDecodeError, ValueError):
            logger.warning("wikipedia_descargados.json corrupto, empezando desde cero.")

    pendientes = [t for t in ARTICULOS_WIKIPEDIA if t not in articulos_descargados]

    if not pendientes:
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        n = sum(1 for _ in open(ruta, encoding='utf-8')) if os.path.exists(ruta) else 0
        logger.info(f"Wikipedia: todos los articulos en cache ({n:,} fragmentos). Saltando.")
        return n

    logger.info(f"Iniciando extraccion de Wikipedia ({len(pendientes)} pendientes de {len(ARTICULOS_WIKIPEDIA)})...")

    total_frag = 0
    buffer: list[str] = []
    exitos = 0
    fallos = 0

    # Configuración conservadora: 3 hilos con delay mayor evita rate limiting
    # ~3 req/s total es seguro para la API pública de Wikipedia
    def fetch_con_delay(title: str):
        time.sleep(random.uniform(0.3, 0.7))  # jitter para evitar sincronización
        return title, fetch_wikipedia_article(title)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch_con_delay, t): t for t in pendientes}

        for idx, future in enumerate(as_completed(futures), 1):
            try:
                titulo, frags = future.result()
                if frags:
                    buffer.extend(frags)
                    total_frag += len(frags)
                    exitos += 1
                    articulos_descargados.add(titulo)
                else:
                    fallos += 1
                    logger.debug(f"  Sin contenido: '{titulo}'")

            except Exception as e:
                fallos += 1
                logger.debug(f"  Excepcion en future: {e}")

            # Guardar progreso cada 50 artículos completados
            if idx % 50 == 0 or idx == len(pendientes):
                guardar_en_cache(buffer, NOMBRE_CACHE)
                buffer = []
                # Persistir qué artículos ya están descargados
                with open(cache_articulos_path, 'w', encoding='utf-8') as f:
                    json.dump(list(articulos_descargados), f, ensure_ascii=False)

            if idx % 25 == 0:
                logger.info(
                    f"  [Wiki] {idx}/{len(pendientes)} procesados | "
                    f"Exitos: {exitos} | Fallos: {fallos} | Fragmentos: {total_frag:,}"
                )

    logger.info(f"Wikipedia: {exitos} articulos exitosos, {fallos} fallidos, {total_frag:,} fragmentos.")
    return total_frag


# ----------------------------------------------------------------------
# Repositorios locales — extensiones corregidas + fragmentación mejorada
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


# FIX: todas las extensiones con punto, sin duplicados
EXTENSIONES_VALIDAS = {
    '.py', '.js', '.ts', '.tsx', '.jsx',
    '.html', '.css', '.scss', '.sass', '.less',
    '.java', '.cpp', '.c', '.h', '.hpp',
    '.go', '.rs', '.rb', '.php', '.swift', '.kt',
    '.md', '.mdx', '.json', '.yaml', '.yml', '.toml',
    '.sql', '.graphql', '.gql',
    '.sh', '.bash', '.zsh',
    '.r', '.m', '.cs', '.fs', '.ex', '.exs',
    '.vue', '.svelte',
}


def procesar_repositorios_locales():
    NOMBRE_CACHE = "codigo_local"

    if cache_existe(NOMBRE_CACHE):
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        n = sum(1 for _ in open(ruta, encoding='utf-8'))
        logger.info(f"Codigo local: cache encontrada ({n:,} fragmentos). Saltando escaneo.")
        return n

    logger.info(f"Escaneando repositorios locales (limite: {MAX_CODIGO_FRAGMENTS:,} fragmentos)...")
    total_frag = 0
    buffer: list[str] = []
    archivos_procesados = 0
    archivos_saltados = 0

    for ruta_base in RUTAS_REPOSITORIOS:
        if not os.path.exists(ruta_base):
            logger.warning(f"  Ruta no encontrada: {ruta_base}")
            continue

        logger.info(f"  Escaneando: {ruta_base}")

        for root, dirs, files in os.walk(ruta_base):
            dirs[:] = [d for d in dirs if directorio_valido(os.path.join(root, d))]

            if total_frag >= MAX_CODIGO_FRAGMENTS:
                break

            for file in files:
                if total_frag >= MAX_CODIGO_FRAGMENTS:
                    break

                ext = os.path.splitext(file)[1].lower()
                if ext not in EXTENSIONES_VALIDAS:
                    continue
                # Excluir archivos minificados, bundles y lockfiles
                if any(x in file for x in ('.min.', 'bundle', '.lock', '-lock', 'package-lock', 'yarn.lock')):
                    continue

                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        contenido = f.read()

                    if not (50 < len(contenido) < 100_000):
                        archivos_saltados += 1
                        continue
                    if contenido.count('\n') < 5:
                        archivos_saltados += 1
                        continue

                    contenido_limpio = limpiar_codigo(contenido)
                    prefijo = f"# Archivo: {file} (extension: {ext})\n\n"
                    texto_con_prefijo = prefijo + contenido_limpio

                    frags = fragmentar_codigo(texto_con_prefijo)
                    if frags:
                        buffer.extend(frags)
                        total_frag += len(frags)
                        archivos_procesados += 1

                except Exception:
                    archivos_saltados += 1

            if len(buffer) >= 500:
                guardar_en_cache(buffer, NOMBRE_CACHE)
                buffer = []

        if total_frag >= MAX_CODIGO_FRAGMENTS:
            logger.info(f"  Limite de {MAX_CODIGO_FRAGMENTS:,} fragmentos alcanzado.")
            break

    guardar_en_cache(buffer, NOMBRE_CACHE)
    logger.info(
        f"Codigo local: {archivos_procesados:,} archivos procesados, "
        f"{archivos_saltados:,} saltados, {total_frag:,} fragmentos."
    )
    return total_frag


# ----------------------------------------------------------------------
# Gutenberg
# ----------------------------------------------------------------------

def descargar_parsear_gutenberg(book_id: int) -> list[str] | None:
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://gutenberg.org/files/{book_id}/{book_id}-0.txt",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                texto = response.read().decode('utf-8', errors='ignore')

            for patron_inicio in [
                "*** START OF THE PROJECT GUTENBERG EBOOK",
                "*** START OF THIS PROJECT GUTENBERG EBOOK",
            ]:
                idx = texto.find(patron_inicio)
                if idx != -1:
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
            fragmentos = fragmentar_por_parrafos(texto_limpio)
            if fragmentos:
                return fragmentos

        except Exception as e:
            logger.debug(f"  Error libro {book_id} desde {url}: {e}")
            continue

    return None


def procesar_gutenberg():
    NOMBRE_CACHE = "gutenberg"

    cache_libros_path = os.path.join(CACHE_DIR, "gutenberg_descargados.json")
    libros_descargados: set = set()

    if os.path.exists(cache_libros_path):
        try:
            with open(cache_libros_path, 'r') as f:
                contenido = f.read().strip()
                if contenido:
                    libros_descargados = set(json.loads(contenido))
        except (json.JSONDecodeError, ValueError):
            logger.warning("gutenberg_descargados.json corrupto, empezando desde cero.")

    pendientes = [bid for bid in IDS_LIBROS_GUTENBERG if bid not in libros_descargados]

    if not pendientes:
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        n = sum(1 for _ in open(ruta, encoding='utf-8')) if os.path.exists(ruta) else 0
        logger.info(f"Gutenberg: todos los libros en cache ({n:,} fragmentos). Saltando.")
        return n

    logger.info(f"Descargando libros de Gutenberg ({len(pendientes)} pendientes de {len(IDS_LIBROS_GUTENBERG)})...")
    total_frag = 0
    buffer: list[str] = []

    for i, book_id in enumerate(pendientes, 1):
        fragmentos = descargar_parsear_gutenberg(book_id)
        if fragmentos:
            buffer.extend(fragmentos)
            total_frag += len(fragmentos)
            libros_descargados.add(book_id)

            with open(cache_libros_path, 'w') as f:
                json.dump(list(libros_descargados), f)

        if len(buffer) >= 50 or i == len(pendientes):
            guardar_en_cache(buffer, NOMBRE_CACHE)
            buffer = []
            logger.info(f"  [Libros] {i}/{len(pendientes)} procesados. Fragmentos: {total_frag:,}")

        time.sleep(1.0)

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
    logger.info(f"Cache en: {os.path.abspath(CACHE_DIR)}")
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
        logger.warning("Proceso interrumpido. El progreso esta guardado en data/raw/")
        logger.warning("Puedes reanudar ejecutando el script de nuevo.")

    finally:
        total = consolidar_cache_en_output()
        tiempo = time.time() - inicio
        logger.info("=" * 60)
        logger.info("RESUMEN FINAL")
        logger.info(f"  Wikipedia:     {frag_wiki:,} fragmentos")
        logger.info(f"  Codigo local:  {frag_codigo:,} fragmentos")
        logger.info(f"  Libros:        {frag_libros:,} fragmentos")
        logger.info(f"  TOTAL:         {total:,} fragmentos en dataset.jsonl")
        logger.info(f"  Tiempo total:  {tiempo:.1f}s ({tiempo/60:.1f} min)")
        logger.info("=" * 60)


if __name__ == "__main__":
    generar_dataset_completo()