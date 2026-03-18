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

# Logging
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

# Configuración
OUTPUT_FILE = "data/dataset.jsonl"
CACHE_DIR   = "data/raw"
os.makedirs("data",     exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

from data.sources import IDS_LIBROS_GUTENBERG, ARTICULOS_WIKIPEDIA, REPOS_GITHUB

# Límites de cada fuente
MAX_WIKIPEDIA_ARTICULOS = None
MAX_GUTENBERG_LIBROS    = None
MAX_OSCAR_FRAGMENTOS    = 130_000
MAX_DIALOGOS_FRAGMENTOS  = 300_000
MAX_GITHUB_FRAGMENTOS   = 200_000
MAX_FRAGMENT_CHARS      = 2_000    # longitud máxima de cada fragmento (chars)


# UTILIDADES DE LIMPIEZA

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


# FRAGMENTACIÓN

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


# CACHÉ

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


# FUENTE 1 — WIKIPEDIA

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


# FUENTE 2 — GUTENBERG

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


# FUENTE 3 — REPOSITORIOS LOCALES (solo en local)

RUTAS_REPOSITORIOS = []

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

    initial_count = 0
    if cache_existe(NOMBRE_CACHE):
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        initial_count = sum(1 for _ in open(ruta, encoding='utf-8'))
        
    MAX_FRAGMENTOS_LOCAL = 10_000
    if initial_count >= MAX_FRAGMENTOS_LOCAL:
        logger.info(f"Codigo local: cache completa ({initial_count:,} fragmentos). Saltando.")
        return initial_count
    
    if initial_count > 0:
        logger.info(f"Codigo local: cache parcial ({initial_count:,}/{MAX_FRAGMENTOS_LOCAL:,}). Continuando...")
    
    logger.info(f"Escaneando {len(rutas_existentes)} repositorios locales...")
    total_frag = initial_count
    encountered_frag = 0
    buffer: list[str] = []
    archivos_procesados = archivos_saltados = 0

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
                        for frag in frags:
                            encountered_frag += 1
                            if encountered_frag <= initial_count:
                                continue
                            
                            buffer.append(frag)
                            total_frag += 1
                        
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


# FUENTE 4 — HUGGING FACE OSCAR (texto web en español)

def procesar_oscar_spanish():
    """
    Descarga fragmentos del dataset OSCAR (español) desde Hugging Face
    usando streaming — nunca descarga el dataset completo.

    Requiere: pip install datasets
    """
    NOMBRE_CACHE = "oscar_spanish"

    initial_count = 0
    if cache_existe(NOMBRE_CACHE):
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        initial_count = sum(1 for _ in open(ruta, encoding='utf-8'))
        if initial_count >= MAX_OSCAR_FRAGMENTOS:
            logger.info(f"OSCAR español: cache completa ({initial_count:,} fragmentos). Saltando.")
            return initial_count
        else:
            logger.info(f"OSCAR español: cache parcial ({initial_count:,}/{MAX_OSCAR_FRAGMENTOS:,}). Continuando...")

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

    total_frag = initial_count
    encountered_frag = 0
    docs_procesados = docs_saltados = 0

    try:
        # Usamos un dataset público y accesible sin gating
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
                encountered_frag += 1
                if encountered_frag <= initial_count:
                    continue

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


# FUENTE 4.1 — DIÁLOGOS Y LENGUAJE NATURAL (Hugging Face)

def procesar_dialogos_naturales():
    """
    Descarga fragmentos de datasets de diálogos y lenguaje hablado en español
    para mejorar la fluidez y naturalidad del modelo.
    """
    NOMBRE_CACHE = "dialogos_naturales"

    initial_count = 0
    if cache_existe(NOMBRE_CACHE):
        ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
        initial_count = sum(1 for _ in open(ruta, encoding='utf-8'))
        if initial_count >= MAX_DIALOGOS_FRAGMENTOS:
            logger.info(f"Diálogos: cache completa ({initial_count:,} fragmentos). Saltando.")
            return initial_count
        else:
            logger.info(f"Diálogos: cache parcial ({initial_count:,}/{MAX_DIALOGOS_FRAGMENTOS:,}). Continuando...")

    logger.info(f"Iniciando descarga de diálogos en español (HF streaming)...")

    try:
        from datasets import load_dataset
    except ImportError:
        return 0

    total_frag = initial_count
    encountered_frag = 0
    buffer: list[str] = []
    # Diálogos: Usamos opus-100
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
                if 'translation' in doc:
                    texto = doc['translation'].get('es', '')
                elif 'text' in doc:
                    texto = doc['text']
                else:
                    continue

                if len(texto) < 40:
                    continue

                encountered_frag += 1
                if encountered_frag <= initial_count:
                    continue

                texto_limpio = limpiar_texto_natural(texto)
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




# FUENTE 5 — GITHUB API (Descargando ZIP del repositorio)

def procesar_github_api():
    """
    Descarga repos públicos de GitHub como ZIP desde github.com/.../archive/...
    Extrae al azar hasta 50 archivos válidos por repositorio.
    """
    NOMBRE_CACHE = "codigo_github"

    # Caché granular: guardamos qué repos ya descargamos para reanudar
    cache_repos_path = os.path.join(CACHE_DIR, "github_descargados.json")
    repos_descargados: set = set()

    if os.path.exists(cache_repos_path):
        try:
            with open(cache_repos_path, 'r', encoding='utf-8') as f:
                contenido = f.read().strip()
                if contenido:
                    repos_descargados = set(json.loads(contenido))
        except (json.JSONDecodeError, ValueError):
            logger.warning("github_descargados.json corrupto, empezando desde cero.")

    total_frag = 0
    ruta = os.path.join(CACHE_DIR, f"{NOMBRE_CACHE}.jsonl")
    if os.path.exists(ruta):
        total_frag = sum(1 for _ in open(ruta, encoding='utf-8'))

    if total_frag >= MAX_GITHUB_FRAGMENTOS:
        logger.info(f"GitHub: Límite alcanzado o superado ({total_frag:,} frags). Saltando.")
        return total_frag

    logger.info(
        f"Comenzando procesamiento de repos de GitHub "
        f"(límite {MAX_GITHUB_FRAGMENTOS:,} fragmentos)..."
    )

    buffer: list[str] = []
    exitos = fallos = 0

    import zipfile
    import io

    for owner, repo, branch, _ in REPOS_GITHUB:
        if total_frag >= MAX_GITHUB_FRAGMENTOS:
            logger.info("  Límite de fragmentos de GitHub alcanzado.")
            break

        clave = f"{owner}/{repo}"
        if clave in repos_descargados:
            continue

        logger.info(f"  [GitHub] Descargando ZIP de {owner}/{repo}...")

        # Intentamos con la rama especificada, o main/master si falla
        ramas_a_probar = [branch]
        if branch not in ["main", "master"]:
            ramas_a_probar.extend(["main", "master"])
        elif branch == "main":
            ramas_a_probar.append("master")
        else:
            ramas_a_probar.append("main")

        zip_data = None
        for b in ramas_a_probar:
            url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{b}.zip"
            req = urllib.request.Request(url, headers={'User-Agent': 'MiLLMDataBuilder/2.0'})
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    zip_data = response.read()
                    break
            except Exception:
                pass

        if not zip_data:
            logger.warning(f"    X Falló descarga de {owner}/{repo}")
            fallos += 1
            continue

        try:
            with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                # Filtrar solo archivos con extensiones válidas y no muy grandes
                archivos_validos = []
                for file_info in z.infolist():
                    if file_info.is_dir() or file_info.file_size > 200_000:
                        continue
                    ext = os.path.splitext(file_info.filename)[1].lower()
                    if ext in EXTENSIONES_VALIDAS:
                        # Excluir paths de test, build, minificados, etc.
                        fname = file_info.filename.lower()
                        ignorar = ['/test/', '/tests/', '.min.', '/node_modules/', '/dist/', '/build/', '/out/', 'vendor/']
                        if not any(ign in fname for ign in ignorar):
                            archivos_validos.append(file_info)

                # Tomamos algunos archivos al azar
                random.shuffle(archivos_validos)
                archivos_a_procesar = archivos_validos[:50]

                repo_frags = 0
                for file_info in archivos_a_procesar:
                    try:
                        contenido = z.read(file_info.filename).decode('utf-8', errors='ignore')
                        if len(contenido.strip()) < 100:
                            continue

                        ext = os.path.splitext(file_info.filename)[1].lower()
                        nombre_archivo = file_info.filename.split('/')[-1]

                        limpio = limpiar_codigo(contenido)
                        prefijo = f"# Repo: {owner}/{repo} | Archivo: {nombre_archivo}\n\n"
                        frags = fragmentar_codigo(prefijo + limpio)

                        if frags:
                            buffer.extend(frags)
                            total_frag += len(frags)
                            repo_frags += len(frags)
                    except Exception:
                        pass

                if repo_frags > 0:
                    exitos += 1
                    logger.info(f"    ✓ {owner}/{repo}: {repo_frags} fragmentos extraídos.")
                else:
                    fallos += 1
                    logger.info(f"    X {owner}/{repo}: Ningún fragmento extraído.")

        except Exception as e:
            logger.warning(f"    X Error descomprimiendo {owner}/{repo}: {e}")
            fallos += 1

        repos_descargados.add(clave)
        
        # Guardar en disco para que no se pierda el progreso de memoria
        guardar_en_cache(buffer, NOMBRE_CACHE)
        buffer = []
        with open(cache_repos_path, 'w', encoding='utf-8') as f:
            json.dump(list(repos_descargados), f, ensure_ascii=False)

        # Pausa amable
        time.sleep(2.0)

    # Si terminamos o se interrumpen, guardar buffer
    guardar_en_cache(buffer, NOMBRE_CACHE)
    with open(cache_repos_path, 'w', encoding='utf-8') as f:
        json.dump(list(repos_descargados), f, ensure_ascii=False)

    logger.info(
        f"GitHub ZIPs finalizado: {exitos} repos exitosos, {fallos} fallidos. "
        f"Total fragmentos GitHub: {total_frag:,}"
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

        # Fuente 5 — GitHub API
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