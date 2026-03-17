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
    fmt_console = logging.Formatter('[%(asctime)s] %(levelname)-8s | %(message)s',
                                    datefmt='%H:%M:%S')
    fmt_file    = logging.Formatter('[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(
        open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    )
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
OUTPUT_FILE = "data/dataset.jsonl"
CACHE_DIR   = "data/raw"
os.makedirs("data",     exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

from data.sources import IDS_LIBROS_GUTENBERG, ARTICULOS_WIKIPEDIA, REPOS_GITHUB

# ── Límites de cada fuente ──────────────────────────────────────────────────
# Ajusta estos valores según el espacio disponible y el tiempo que tengas.
MAX_WIKIPEDIA_ARTICULOS = 0       # Límite de artículos (útil para pruebas rápidas)
MAX_GUTENBERG_LIBROS    = None       # Límite de libros de Gutenberg
MAX_OSCAR_FRAGMENTOS    = 25_000  # fragmentos de texto web en español (OSCAR)
MAX_DIALOGOS_FRAGMENTOS  = 120_000  # fragmentos de diálogos/conversaciones (HF)
MAX_GITHUB_FRAGMENTOS   = 70_000  # fragmentos de código de GitHub
MAX_FRAGMENT_CHARS      = 2_000    # longitud máxima de cada fragmento (chars)


# ======================================================================
# UTILIDADES DE LIMPIEZA
# ======================================================================

def limpiar_texto_natural(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = ''.join(c for c in text
                   if unicodedata.category(c)[0] != 'C' or c in '\n ')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    return text.strip()

def limpiar_codigo(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = ''.join(c for c in text
                   if unicodedata.category(c)[0] != 'C' or c in '\n\t ')
    text = re.sub(r'\n{5,}', '\n\n\n', text)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    return text.strip()


# ======================================================================
# FRAGMENTACIÓN
# ======================================================================

def fragmentar_por_parrafos(texto: str,
                             max_chars: int = MAX_FRAGMENT_CHARS,
                             min_chars: int = 150) -> list[str]:
    parrafos  = re.split(r'\n{2,}', texto)
    fragmentos = []
    buffer    = ""

    for parrafo in parrafos:
        parrafo = parrafo.strip()
        if not parrafo:
            continue

        if len(parrafo) > max_chars:
            oraciones  = re.split(r'(?<=[.!?])\s+', parrafo)
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


def fragmentar_codigo(texto: str,
                      max_chars: int = MAX_FRAGMENT_CHARS,
                      min_chars: int = 100) -> list[str]:
    lineas    = texto.split('\n')
    fragmentos = []
    buffer_lineas: list[str] = []

    for linea in lineas:
        buffer_lineas.append(linea)
        contenido = '\n'.join(buffer_lineas)

        if len(contenido) >= max_chars and linea.strip() == '':
            if len(contenido.strip()) >= min_chars:
                fragmentos.append(contenido.strip())
            buffer_lineas = []
            continue

        if len(contenido) >= max_chars * 1.5:
            if len(contenido.strip()) >= min_chars:
                fragmentos.append(contenido.strip())
            buffer_lineas = []

    resto = '\n'.join(buffer_lineas).strip()
    if len(resto) >= min_chars:
        fragmentos.append(resto)

    return fragmentos


# ======================================================================
# CACHÉ (igual que antes)
# ======================================================================

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


# ======================================================================
# FUENTE 1 — WIKIPEDIA (sin cambios respecto a tu versión original)
# ======================================================================

def _encode_wiki_title(title: str) -> str:
    decoded = urllib.parse.unquote(title)
    return urllib.parse.quote(decoded, safe='_:()/')

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
                data  = json.loads(response.read().decode())
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
                wait = (2 ** intento) * 5 + random.uniform(0, 2)
                time.sleep(wait)
            elif e.code in (500, 502, 503):
                time.sleep((2 ** intento) * 2)
            else:
                return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            wait = (2 ** intento) * 1.5 + random.uniform(0, 1)
            time.sleep(wait)
        except Exception:
            return None

    return None

def procesar_wikipedia_con_hilos():
    NOMBRE_CACHE = "wikipedia"

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

    limite_wiki = MAX_WIKIPEDIA_ARTICULOS if MAX_WIKIPEDIA_ARTICULOS is not None else len(ARTICULOS_WIKIPEDIA)
    articulos_objetivo = ARTICULOS_WIKIPEDIA[:limite_wiki]
    pendientes = [t for t in articulos_objetivo if t not in articulos_descargados]

    if not pendientes:
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        n = sum(1 for _ in open(ruta, encoding='utf-8')) if os.path.exists(ruta) else 0
        logger.info(f"Wikipedia: todos los articulos en cache ({n:,} fragmentos). Saltando.")
        return n

    logger.info(
        f"Iniciando extraccion de Wikipedia "
        f"({len(pendientes)} pendientes de un total de {limite_wiki})..."
    )

    total_frag = 0
    buffer: list[str] = []
    exitos = fallos = 0

    def fetch_con_delay(title: str):
        time.sleep(random.uniform(0.3, 0.7))
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
            except Exception:
                fallos += 1

            if idx % 50 == 0 or idx == len(pendientes):
                guardar_en_cache(buffer, NOMBRE_CACHE)
                buffer = []
                with open(cache_articulos_path, 'w', encoding='utf-8') as f:
                    json.dump(list(articulos_descargados), f, ensure_ascii=False)

            if idx % 25 == 0:
                logger.info(
                    f"  [Wiki] {idx}/{len(pendientes)} | "
                    f"Exitos: {exitos} | Fallos: {fallos} | Frags: {total_frag:,}"
                )

    logger.info(
        f"Wikipedia: {exitos} articulos exitosos, {fallos} fallidos, "
        f"{total_frag:,} fragmentos."
    )
    return total_frag


# ======================================================================
# FUENTE 2 — GUTENBERG (sin cambios)
# ======================================================================

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

            for patron in ["*** START OF THE PROJECT GUTENBERG EBOOK",
                           "*** START OF THIS PROJECT GUTENBERG EBOOK"]:
                idx = texto.find(patron)
                if idx != -1:
                    resto     = texto[idx:]
                    fin_marca = resto.find('\n\n')
                    if fin_marca != -1:
                        texto = resto[fin_marca:].strip()
                    break

            for patron in ["*** END OF THE PROJECT GUTENBERG EBOOK",
                           "*** END OF THIS PROJECT GUTENBERG EBOOK"]:
                idx = texto.find(patron)
                if idx != -1:
                    texto = texto[:idx].strip()
                    break

            texto_limpio = limpiar_texto_natural(texto)
            fragmentos   = fragmentar_por_parrafos(texto_limpio)
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
            pass

    limite_gut = MAX_GUTENBERG_LIBROS if MAX_GUTENBERG_LIBROS is not None else len(IDS_LIBROS_GUTENBERG)
    libros_objetivo = IDS_LIBROS_GUTENBERG[:limite_gut]
    pendientes = [bid for bid in libros_objetivo
                  if bid not in libros_descargados]

    if not pendientes:
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        n = sum(1 for _ in open(ruta, encoding='utf-8')) if os.path.exists(ruta) else 0
        logger.info(f"Gutenberg: todos los libros en cache ({n:,} fragmentos). Saltando.")
        return n

    logger.info(
        f"Descargando libros de Gutenberg "
        f"({len(pendientes)} pendientes de un total de {limite_gut})..."
    )
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
            logger.info(
                f"  [Libros] {i}/{len(pendientes)} procesados. "
                f"Fragmentos: {total_frag:,}"
            )

        time.sleep(1.0)

    logger.info(f"Gutenberg: {total_frag:,} fragmentos nuevos.")
    return total_frag


# ======================================================================
# FUENTE 3 — REPOSITORIOS LOCALES (solo se usa cuando corres en tu PC)
# En Colab esta función se salta automáticamente porque las rutas no existen.
# ======================================================================

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
    r"C:\Users\julio\Documents\Dev\typescript\vite",
]

EXTENSIONES_VALIDAS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css', '.scss',
    '.java', '.cpp', '.c', '.h', '.hpp', '.go', '.rs', '.rb', '.php',
    '.swift', '.kt', '.md', '.mdx', '.json', '.yaml', '.yml', '.toml',
    '.sql', '.graphql', '.sh', '.bash', '.r', '.cs', '.vue', '.svelte',
}

def directorio_valido(filepath: str) -> bool:
    ruta = filepath.lower().replace('\\', '/')
    excluir = [
        'node_modules', '.git', 'venv', 'env', '__pycache__',
        '.idea', '.vscode', 'dist', 'build', 'out', 'target',
        'coverage', '.next', '.nuxt', '.angular', 'tmp', 'cache',
    ]
    return not any(exc in ruta for exc in excluir)

def procesar_repositorios_locales():
    NOMBRE_CACHE = "codigo_local"

    # Detectar si alguna ruta local existe (estamos en la PC del usuario)
    rutas_existentes = [r for r in RUTAS_REPOSITORIOS if os.path.exists(r)]

    if not rutas_existentes:
        logger.info(
            "Repositorios locales: no se encontraron rutas locales "
            "(normal en Colab). Saltando."
        )
        return 0

    if cache_existe(NOMBRE_CACHE):
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        n    = sum(1 for _ in open(ruta, encoding='utf-8'))
        logger.info(f"Codigo local: cache encontrada ({n:,} fragmentos). Saltando.")
        return n

    logger.info(f"Escaneando {len(rutas_existentes)} repositorios locales...")
    total_frag = 0
    buffer: list[str] = []
    archivos_procesados = archivos_saltados = 0
    MAX_FRAGMENTOS_LOCAL = 10_000

    for ruta_base in rutas_existentes:
        for root, dirs, files in os.walk(ruta_base):
            dirs[:] = [d for d in dirs
                       if directorio_valido(os.path.join(root, d))]
            if total_frag >= MAX_FRAGMENTOS_LOCAL:
                break
            for file in files:
                if total_frag >= MAX_FRAGMENTOS_LOCAL:
                    break
                ext = os.path.splitext(file)[1].lower()
                if ext not in EXTENSIONES_VALIDAS:
                    continue
                if any(x in file for x in
                       ('.min.', 'bundle', '.lock', '-lock')):
                    continue
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r',
                              encoding='utf-8', errors='ignore') as f:
                        contenido = f.read()
                    if not (50 < len(contenido) < 100_000):
                        continue
                    if contenido.count('\n') < 5:
                        continue
                    limpio  = limpiar_codigo(contenido)
                    prefijo = f"# Archivo: {file} (extension: {ext})\n\n"
                    frags   = fragmentar_codigo(prefijo + limpio)
                    if frags:
                        buffer.extend(frags)
                        total_frag += len(frags)
                        archivos_procesados += 1
                except Exception:
                    archivos_saltados += 1

            if len(buffer) >= 500:
                guardar_en_cache(buffer, NOMBRE_CACHE)
                buffer = []

    guardar_en_cache(buffer, NOMBRE_CACHE)
    logger.info(
        f"Codigo local: {archivos_procesados:,} archivos, "
        f"{total_frag:,} fragmentos."
    )
    return total_frag


# ======================================================================
# FUENTE 4 — HUGGING FACE OSCAR (texto web en español)
# Usa streaming para NO descargar el dataset completo a disco.
# El dataset OSCAR pesa cientos de GB, pero con streaming tomamos solo
# lo que necesitamos y luego paramos.
# ======================================================================

def procesar_oscar_spanish():
    """
    Descarga fragmentos del dataset OSCAR (español) desde Hugging Face
    usando streaming — nunca descarga el dataset completo.

    Requiere: pip install datasets
    """
    NOMBRE_CACHE = "oscar_spanish"

    if cache_existe(NOMBRE_CACHE):
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        n    = sum(1 for _ in open(ruta, encoding='utf-8'))
        logger.info(f"OSCAR español: cache encontrada ({n:,} fragmentos). Saltando.")
        return n

    logger.info(
        f"Iniciando descarga de OSCAR español via streaming "
        f"(limite: {MAX_OSCAR_FRAGMENTOS:,} fragmentos)..."
    )

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "La librería 'datasets' no está instalada. "
            "Ejecuta: pip install datasets"
        )
        return 0

    total_frag = 0
    buffer: list[str] = []
    docs_procesados = docs_saltados = 0

    try:
        # Usamos un dataset público y accesible sin gating (Spanish Billion Words es genial)
        logger.info("  Cargando Spanish Billion Words (Muestra)...")
        dataset = load_dataset(
            "pablousieto/spanish_billion_words", 
            split="train",
            streaming=True
        )

        for doc in dataset:
            # Cada documento tiene un campo "text"
            texto = doc.get("text", "").strip()

            # Filtros de calidad básicos
            if len(texto) < 200:
                docs_saltados += 1
                continue
            # Descartamos documentos con demasiados caracteres no latinos
            # (spam, caracteres raros, etc.)
            chars_latinos = sum(
                1 for c in texto
                if unicodedata.category(c).startswith(('L', 'N', 'Z', 'P'))
            )
            if chars_latinos / max(len(texto), 1) < 0.85:
                docs_saltados += 1
                continue

            texto_limpio = limpiar_texto_natural(texto)
            frags = fragmentar_por_parrafos(texto_limpio)

            for frag in frags:
                if total_frag >= MAX_OSCAR_FRAGMENTOS:
                    break
                buffer.append(frag)
                total_frag += 1

            docs_procesados += 1

            # Guardamos en cache cada 1000 documentos
            if docs_procesados % 1000 == 0:
                guardar_en_cache(buffer, NOMBRE_CACHE)
                buffer = []
                logger.info(
                    f"  [OSCAR] {docs_procesados:,} docs procesados | "
                    f"Fragmentos: {total_frag:,}/{MAX_OSCAR_FRAGMENTOS:,}"
                )

            if total_frag >= MAX_OSCAR_FRAGMENTOS:
                logger.info(
                    f"  Limite de {MAX_OSCAR_FRAGMENTOS:,} fragmentos alcanzado."
                )
                break

    except Exception as e:
        logger.error(f"Error al procesar OSCAR: {e}")

    finally:
        # Guardamos lo que quede en el buffer
        guardar_en_cache(buffer, NOMBRE_CACHE)

    logger.info(
        f"OSCAR español: {docs_procesados:,} documentos procesados, "
        f"{docs_saltados:,} saltados, {total_frag:,} fragmentos guardados."
    )
    return total_frag


# ======================================================================
# FUENTE 4.1 — DIÁLOGOS Y LENGUAJE NATURAL (Hugging Face)
# ======================================================================

def procesar_dialogos_naturales():
    """
    Descarga fragmentos de datasets de diálogos y lenguaje hablado en español
    para mejorar la fluidez y naturalidad del modelo.
    """
    NOMBRE_CACHE = "dialogos_naturales"

    if cache_existe(NOMBRE_CACHE):
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        n    = sum(1 for _ in open(ruta, encoding='utf-8'))
        logger.info(f"Diálogos: cache encontrada ({n:,} fragmentos). Saltando.")
        return n

    logger.info(f"Iniciando descarga de diálogos en español (HF streaming)...")

    try:
        from datasets import load_dataset
    except ImportError:
        return 0

    total_frag = 0
    buffer: list[str] = []
    
    # Diálogos: Usamos opus-100 que es público y no-gated
    datasets_config = [
        {"path": "Helsinki-NLP/opus-100", "name": "en-es", "split": "train"},
    ]

    for conf in datasets_config:
        if total_frag >= MAX_DIALOGOS_FRAGMENTOS:
            break
            
        try:
            logger.info(f"  Cargando sub-dataset: {conf['path']} ({conf['name']})...")
            dataset = load_dataset(
                conf['path'], 
                conf['name'], 
                split=conf['split'], 
                streaming=True
            )

            for doc in dataset:
                # En opus_subtitles los datos vienen en un dict 'translation'
                # Ej: {'translation': {'en': 'Hello', 'es': 'Hola'}}
                if 'translation' in doc:
                    texto = doc['translation'].get('es', '')
                elif 'text' in doc:
                    texto = doc['text']
                else:
                    continue

                if len(texto) < 40: # Diálogos un poco más cortos son aceptables
                    continue

                texto_limpio = limpiar_texto_natural(texto)
                # No usamos fragmentar por párrafos aquí porque los diálogos suelen ser cortos
                buffer.append(texto_limpio)
                total_frag += 1
                
                if total_frag % 1000 == 0:
                    guardar_en_cache(buffer, NOMBRE_CACHE)
                    buffer = []
                    logger.info(f"    [Diálogos] {total_frag:,}/{MAX_DIALOGOS_FRAGMENTOS:,} fragmentos...")

                if total_frag >= MAX_DIALOGOS_FRAGMENTOS:
                    break

        except Exception as e:
            logger.warning(f"  Error cargando dataset de diálogos {conf['path']}: {e}")
            continue

    guardar_en_cache(buffer, NOMBRE_CACHE)
    logger.info(f"Diálogos: {total_frag:,} fragmentos guardados para fluidez natural.")
    return total_frag



# ======================================================================
# FUENTE 5 — GITHUB API (código sin clonar repos)
# Descarga archivos individuales via raw.githubusercontent.com
# usando la lista REPOS_GITHUB de sources.py
# ======================================================================

def _descargar_archivo_github(owner: str, repo: str,
                               branch: str, path: str,
                               max_retries: int = 3) -> str | None:
    """
    Descarga el contenido de un archivo de GitHub via raw URL.
    Devuelve el texto del archivo o None si falla.
    """
    # raw.githubusercontent.com no requiere autenticación para repos públicos
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    headers = {
        'User-Agent': 'MiLLMDataBuilder/2.0 (aprendizaje, sin fines comerciales)'
    }

    for intento in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as response:
                # Algunos archivos binarios (imágenes, etc.) hay que filtrarlos
                content_type = response.headers.get('Content-Type', '')
                if 'text' not in content_type and 'json' not in content_type:
                    return None
                return response.read().decode('utf-8', errors='ignore')

        except urllib.error.HTTPError as e:
            if e.code == 404:
                # El archivo no existe en esa rama, no reintentar
                return None
            if e.code == 429:
                # Rate limit de GitHub: esperamos más
                wait = (2 ** intento) * 10 + random.uniform(0, 3)
                logger.debug(f"  Rate limit GitHub ({owner}/{repo}). "
                             f"Esperando {wait:.1f}s...")
                time.sleep(wait)
            else:
                time.sleep((2 ** intento) * 2)

        except (urllib.error.URLError, TimeoutError, OSError) as e:
            wait = (2 ** intento) * 2 + random.uniform(0, 1)
            logger.debug(f"  Error red {owner}/{repo}/{path}: {e}. "
                         f"Reintento {intento+1}/{max_retries}...")
            time.sleep(wait)

        except Exception:
            return None

    return None


def procesar_github_api():
    """
    Descarga archivos de código de repos públicos de GitHub usando
    raw.githubusercontent.com (sin clonar, sin autenticación).

    La lista de repos y archivos viene de sources.REPOS_GITHUB.
    """
    NOMBRE_CACHE = "codigo_github"

    # Caché granular: guardamos qué archivos ya descargamos para reanudar
    cache_archivos_path = os.path.join(CACHE_DIR, "github_descargados.json")
    archivos_descargados: set = set()

    if os.path.exists(cache_archivos_path):
        try:
            with open(cache_archivos_path, 'r', encoding='utf-8') as f:
                contenido = f.read().strip()
                if contenido:
                    archivos_descargados = set(json.loads(contenido))
        except (json.JSONDecodeError, ValueError):
            logger.warning("github_descargados.json corrupto, empezando desde cero.")

    # Aplanamos la lista de repos en tareas individuales (owner, repo, branch, path)
    tareas_totales = []
    for owner, repo, branch, paths in REPOS_GITHUB:
        for path in paths:
            clave = f"{owner}/{repo}/{branch}/{path}"
            if clave not in archivos_descargados:
                tareas_totales.append((owner, repo, branch, path, clave))

    if not tareas_totales:
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        n    = sum(1 for _ in open(ruta, encoding='utf-8')) if os.path.exists(ruta) else 0
        logger.info(f"GitHub API: todos los archivos en cache ({n:,} fragmentos). Saltando.")
        return n

    logger.info(
        f"Descargando codigo de GitHub "
        f"({len(tareas_totales)} archivos pendientes, "
        f"limite {MAX_GITHUB_FRAGMENTOS:,} fragmentos)..."
    )

    total_frag = 0
    buffer: list[str] = []
    exitos = fallos = 0

    def descargar_tarea(tarea):
        owner, repo, branch, path, clave = tarea
        # Pequeño delay aleatorio para no saturar la API
        time.sleep(random.uniform(0.5, 1.2))
        contenido = _descargar_archivo_github(owner, repo, branch, path)
        return clave, path, contenido

    # 4 hilos: equilibrio entre velocidad y respeto a los límites de GitHub
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(descargar_tarea, t): t
            for t in tareas_totales
        }

        for idx, future in enumerate(as_completed(futures), 1):
            if total_frag >= MAX_GITHUB_FRAGMENTOS:
                # Cancelamos los futures pendientes si llegamos al límite
                for f in futures:
                    f.cancel()
                break

            try:
                clave, path, contenido = future.result()

                if contenido and len(contenido.strip()) > 100:
                    # Detectamos la extensión para el prefijo
                    ext = os.path.splitext(path)[1].lower()
                    nombre_archivo = os.path.basename(path)

                    limpio  = limpiar_codigo(contenido)
                    prefijo = f"# Archivo: {nombre_archivo} (extension: {ext})\n\n"
                    frags   = fragmentar_codigo(prefijo + limpio)

                    if frags:
                        buffer.extend(frags)
                        total_frag += len(frags)
                        exitos     += 1
                        archivos_descargados.add(clave)
                else:
                    fallos += 1

            except Exception as e:
                fallos += 1
                logger.debug(f"  Excepcion en future GitHub: {e}")

            # Guardamos caché cada 20 archivos
            if idx % 20 == 0 or idx == len(tareas_totales):
                guardar_en_cache(buffer, NOMBRE_CACHE)
                buffer = []
                with open(cache_archivos_path, 'w', encoding='utf-8') as f:
                    json.dump(list(archivos_descargados), f, ensure_ascii=False)

            if idx % 10 == 0:
                logger.info(
                    f"  [GitHub] {idx}/{len(tareas_totales)} archivos | "
                    f"Exitos: {exitos} | Fallos: {fallos} | "
                    f"Fragmentos: {total_frag:,}"
                )

    # Guardamos lo que quede
    guardar_en_cache(buffer, NOMBRE_CACHE)
    with open(cache_archivos_path, 'w', encoding='utf-8') as f:
        json.dump(list(archivos_descargados), f, ensure_ascii=False)

    logger.info(
        f"GitHub API: {exitos} archivos exitosos, {fallos} fallidos, "
        f"{total_frag:,} fragmentos."
    )
    return total_frag


# ======================================================================
# EJECUCIÓN PRINCIPAL
# ======================================================================

def generar_dataset_completo():
    inicio = time.time()
    logger.info("=" * 60)
    logger.info("INICIANDO GENERADOR DE DATASET LLM")
    logger.info("=" * 60)
    logger.info(f"Cache en:  {os.path.abspath(CACHE_DIR)}")
    logger.info(f"Salida en: {os.path.abspath(OUTPUT_FILE)}")
    logger.info("Si interrumpes y reinicias, el progreso se conserva.")
    logger.info("=" * 60)

    frag_wiki = frag_gutenberg = frag_local = frag_oscar = frag_github = 0

    try:
        # Fuente 1 — Wikipedia
        frag_wiki      = procesar_wikipedia_con_hilos()
        logger.info("-" * 60)

        # Fuente 2 — Gutenberg
        frag_gutenberg = procesar_gutenberg()
        logger.info("-" * 60)

        # Fuente 3 — Repos locales (solo en PC, se salta en Colab)
        frag_local     = procesar_repositorios_locales()
        logger.info("-" * 60)

        # Fuente 4 — OSCAR español (Hugging Face, streaming)
        frag_oscar     = procesar_oscar_spanish()
        logger.info("-" * 60)

        # Fuente 4.1 — Diálogos Naturales
        frag_dialogos  = procesar_dialogos_naturales()
        logger.info("-" * 60)

        # Fuente 5 — GitHub API (código sin clonar)
        frag_github    = procesar_github_api()
        logger.info("-" * 60)

    except KeyboardInterrupt:
        logger.warning("Proceso interrumpido. El progreso esta guardado en data/raw/")
        logger.warning("Puedes reanudar ejecutando el script de nuevo.")

    finally:
        total  = consolidar_cache_en_output()
        tiempo = time.time() - inicio

        logger.info("=" * 60)
        logger.info("RESUMEN FINAL")
        logger.info(f"  Wikipedia:    {frag_wiki:,} fragmentos")
        logger.info(f"  Gutenberg:    {frag_gutenberg:,} fragmentos")
        logger.info(f"  Repos local:  {frag_local:,} fragmentos")
        logger.info(f"  OSCAR español: {frag_oscar:,} fragmentos")
        logger.info(f"  Diálogos:     {frag_dialogos:,} fragmentos")
        logger.info(f"  GitHub API:   {frag_github:,} fragmentos")
        logger.info(f"  TOTAL:        {total:,} fragmentos en dataset.jsonl")
        logger.info(f"  Tiempo:       {tiempo:.1f}s ({tiempo/60:.1f} min)")
        logger.info("=" * 60)


if __name__ == "__main__":
    generar_dataset_completo()