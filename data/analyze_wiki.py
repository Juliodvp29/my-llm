
import re
from collections import Counter

file_path = r'c:\Users\julio\PycharmProjects\my-llm\data\sources.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Extract ARTICULOS_WIKIPEDIA list using a regex to find the start and follow it to the bracket
# The Wikipedia articles list starts at line 43
# and goes until the REPOS_GITHUB or similar comment
# We'll use a regex search to extract it.

match = re.search(r'ARTICULOS_WIKIPEDIA = \[(.*?)\]\s*(?=\r?\n\s*#|\r?\n\s*[A-Z_]+ =)', content, re.DOTALL)
if match:
    list_content = match.group(1)
    # Filter only titles which are surrounded by quotes
    articles = re.findall(r'["\'](.*?)["\']', list_content)
else:
    # Fallback to search for anything before REPOS_GITHUB
    match = re.search(r'ARTICULOS_WIKIPEDIA = \[(.*?)\]', content, re.DOTALL)
    if match:
        list_content = match.group(1)
        articles = re.findall(r'["\'](.*?)["\']', list_content)
    else:
        print("Error: No se pudo extraer la lista")
        exit()

# Define categories
CATEGORIES = {
    'Ciencia y Tecnología': ['tecnología', 'física', 'química', 'biología', 'matemática', 'computación', 'red', 'sistema', 'procesamiento', 'digital', 'ia', 'inteligencia', 'algoritmo', 'protocolo', 'internet', 'web', 'código', 'software', 'hardware', 'espacio', 'cuántica', 'energía', 'ciencia', 'dato', 'información', 'ingeniería', 'mecánica', 'eléctrica', 'nanotecnología'],
    'Historia y Política': ['historia', 'imperio', 'guerra', 'revolución', 'dinastía', 'tratado', 'rey', 'reina', 'presidente', 'política', 'democracia', 'estado', 'socialismo', 'comunismo', 'fascismo', 'nacionalismo', 'independencia', 'batalla', 'siglo', 'edad', 'antigua', 'moderna', 'contemporánea', 'constitucional', 'parlamentario', 'derecho', 'paz', 'crisis', 'mundial'],
    'Geografía y Naturaleza': ['río', 'lago', 'mar', 'océano', 'desierto', 'montaña', 'cordillera', 'isla', 'país', 'ciudad', 'capital', 'continente', 'animal', 'planta', 'bosque', 'selva', 'clima', 'planeta', 'estrella', 'galaxia', 'antártida', 'amazonas', 'nilo', 'mediterráneo', 'península', 'caída', 'población', 'parque', 'natural'],
    'Cultura y Entretenimiento': ['música', 'cine', 'película', 'banda', 'arte', 'literatura', 'novela', 'pintura', 'teatro', 'escultura', 'arquitectura', 'videojuego', 'juego', 'youtuber', 'streamer', 'influencer', 'serie', 'actor', 'actriz', 'cantante', 'álbum', 'canción', 'festival', 'museo', 'cultura', 'moda', 'fotografía', 'cómic', 'animación', 'danza', 'poesía', 'estilo'],
    'Deportes': ['fútbol', 'baloncesto', 'tenis', 'estadio', 'copa', 'mundial', 'liga', 'campeón', 'atleta', 'jugador', 'deporte', 'olímpic', 'fórmula_1', 'gran_premio', 'tour', 'vuelta', 'derbi', 'club', 'entrenador', 'fichaje', 'maracaná', 'wembley', 'bernabéu', 'camp_nou'],
    'Economía y Empresa': ['mercado', 'empresa', 'economía', 'financiero', 'banco', 'comercio', 'inversión', 'negocio', 'marketing', 'pnb', 'pib', 'inflación', 'moneda', 'startup', 'amazon', 'google', 'facebook', 'microsoft', 'apple', 'tesla', 'netflix', 'bolsa', 'valor', 'precio', 'consumo'],
    'Sociedad y Humanidades': ['psicología', 'sociología', 'ética', 'filosofía', 'religión', 'educación', 'sociedad', 'derechos', 'civilización', 'persona', 'mente', 'conciencia', 'lenguaje', 'lingüística', 'comunicación', 'social', 'humano', 'conocimiento', 'identidad', 'feminismo', 'racismo', 'pobreza', 'bienestar']
}

cat_counts = Counter()
for art in articles:
    found = False
    art_lower = art.lower().replace("_", " ") # Wikipedia titles use _ for spaces
    for cat, keywords in CATEGORIES.items():
        if any(k in art_lower for k in keywords):
            cat_counts[cat] += 1
            found = True
            break
    if not found:
        cat_counts['Varios / Biografías'] += 1

print(f"Total Artículos encontrados: {len(articles)}")
print("\n--- TOP TEMÁTICAS (Wikipedia) ---")
for cat, count in cat_counts.most_common(10):
    print(f" {cat}: {count}")
