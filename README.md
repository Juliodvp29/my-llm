# Mini LLM

Un proyecto para construir, entrenar y evaluar un Modelo de Lenguaje Grande (LLM) tipo Transformer desde cero utilizando PyTorch. un tokenizer entrenado a medida y soporte nativo para entrenamiento distribuido en múltiples GPUs. (Proyecto desarrollado para estudio, investigación)

## Características Principales

- **Arquitectura desde Cero**: Implementación pura en PyTorch de las capas de un Transformer autoregresivo (Self-Attention, Embeddings, Multi-Head Attention).
- **Entrenamiento Distribuido (DDP)**: Entrenamiento optimizado empleando `DistributedDataParallel` en lugar del clásico `DataParallel` para sacar el máximo provecho al hardware multi-GPU eliminando cuellos de botella.
- **Eficiencia de Memoria**: Uso de _Mixed Precision_ (AMP) nativo de PyTorch y acumulación de gradientes para entrenar modelos robustos con memoria VRAM limitada.
- **Dataset Híbrido Multilingüe**: Pipelines automatizados para descargar y fusionar código fuente (repositorios de GitHub) y lenguaje natural (Wikipedia).
- **Generación Interactiva**: Script completo con control de `temperature`, autoregresión y `top_k` para probar qué tan bien el modelo autocompleta código o texto.

---

## 📁 Estructura del Proyecto

```text
my-llm/
├── data/
│   ├── prepare.py       # Descarga y procesa la data plana (Wikipedia/GitHub)
│   ├── sources.py       # Diccionarios y configuraciones de las fuentes de datos
│   └── dataset.jsonl    # Dataset final listo para ser ingerido por el modelo
├── model/
│   ├── transformer.py   # Ensamblaje del modelo principal (MiniGPT)
│   ├── attention.py     # Mecanismos de atención (Causal Self-Attention)
│   └── embeddings.py    # Codificación posicional y embeddings
├── models/
│   ├── checkpoints/     # Carpeta de guardado (best_model.pt, last_model.pt)
│   └── tokenizer.json   # El tokenizador BPE resultante
├── tokenizer.py         # Entrena un tokenizador personalizado basado en tus datos
├── train.py             # Bucle principal de entrenamiento (DDP, AMP, optimizadores)
└── generate.py          # Carga el modelo y genera inferencia interactiva
```

---

## Flujo de Trabajo

El proyecto está diseñado para ejecutarse en fases secuenciales.

### 1. Preparar el Dataset

Antes de entrenar, necesitas descargar y sanear los datos crudos definidos en `data/sources.py`.

```bash
python data/prepare.py
```

_Esto generará un archivo enorme llamado `dataset.jsonl` en tu carpeta `data` conteniendo todos tus fragmentos de texto y código normalizados._

### 2. Entrenar el Tokenizer

Este script lee tu datatset y crea un vocabulario de sub-palabras para enseñarle a fragmentar el texto eficientemente.

```bash
python tokenizer.py
```

### 3. Entrenar el Modelo

El script de entrenamiento está altamente parametrizado. Puedes lanzarlo de forma tradicional y él mismo se encargará de detectar tus GPUs; si estás en Kaggle, levantará de forma automática hilos múltiples para dividir la carga de trabajo.

```bash
python train.py
```

_Notas sobre el entrenamiento:_

- Utiliza **AdamW** con **Cosine Decay Warmup** como _learning rate scheduler_.
- El progreso (Loss, Perplexity, Tok/s) y los puntos de control (`.pt`) se auto-guardarán en la carpeta `models/checkpoints/` por cada época.

### 4. Generar Texto e Inferir

Una vez entrenado o mediante la carga de un _checkpoint_ previo, usa `generate.py` para probar tu modelo:

```bash
python generate.py
```

_Podrás ingresar "prompts" en la terminal y ver cómo tu IA completa automáticamente el resto del código o la oración basándose en la temperatura deseada._

---

## Detalles Técnicos del Modelo (Configuración Base)

La configuración `CONFIG` ubicada en `train.py` dicta la ambición del modelo.

- **Vocabulario**: 32,000 tokens.
- **Contexto (Max Length)**: 512 tokens.
- **Capas (n_layers)**: 24
- **Cabezales (n_heads)**: 16
- **Dimensión (d_model)**: 1024
- **Parámetros Estimados**: ~335 Millones

---

## Conceptos Matemáticos y Arquitectura implementada

Para desmitificar cómo "piensa" el modelo, aquí hay un resumen de la matemática clave y cómo fue programada en este repositorio:

### 1. Causal Self-Attention (Las "conexiones neuronales")

En el archivo `model/attention.py` radica el **Self-Attention**. El modelo transforma cada palabra (token) en tres vectores clave:

- **Query (Q)**: Lo que el token está buscando.
- **Key (K)**: Lo que el token contiene.
- **Value (V)**: La esencia del token si hay un _match_.

Matemáticamente, calcula el producto punto entre $Q$ y $K$ para determinar la "afinidad" (atención) entre dos palabras, y lo divide entre $\sqrt{d_k}$ para estabilizar el cálculo. Usamos una **Máscara Causal (Triangular)** llena de infinitos negativos (`-inf`) antes de aplicar _Softmax_. Esto tapa las palabras futuras (de la derecha) matemáticamente, obligando al modelo a adivinar el futuro solo viendo el pasado.

### 2. Positional Embeddings (El sentido del tiempo)

Los Transformers procesan todas las palabras de golpe pero tiene un defecto: pierden la noción del "orden" de las palabras.
En `model/embeddings.py`, solucionamos esto creando una matriz adicional. A cada _token embedding_ (el significado nativo de la palabra) le sumamos un **Embedding Posicional**, que es simplemente otro vector de la misma dimensión aprendido dinámicamente que rastrea la posición `[0, 1, 2, ..., 512]`. Así el modelo sabe qué palabra va de primera o última.

### 3. Cross-Entropy Loss (Castigando los errores)

Durante el ciclo en `train.py`, el modelo arroja _logits_ (números crudos) que indican qué es lo que piensa que va después. Esto pasa por una función **Softmax** que comprime todos estos números en probabilidades matemáticas del `0%` al `100%`.
El castigo a la equivocación que se ve como `Loss: 8.5` en la terminal, se calcula con **Cross-Entropy**. Esta función aplica un logaritmo negativo que castiga exponencialmente al modelo cuando le dio "muy poca probabilidad" a la palabra correcta matemática que estaba en nuestro archivo JSON.

### 4. Gradient Accumulation

El modelo requiere ver grandes lotes de datos para aprender bien pero la VRAM física no los soporta. La ecuación de actualización del optimizador (`optimizer.step()`) suma las derivadas parciales (los gradientes). En `train.py`, aprovechamos la propiedad de adición matemática: procesamos lotes chiquititos (ej: 4 ejemplos) y en lugar de aplicar el aprendizaje de golpe, sumamos (`acumulamos`) los gradientes obtenidos en diferentes rondas usando una variable retenida en memoria. Solo cuando reunimos 8 rondas (64 ejemplos "virtuales"), bajamos bandera matemática `optimizer.step()`, actualizando pesos

---

## Requisitos

El framework principal usado es PyTorch. Instala las dependencias típicas de Inteligencia Artificial:

```bash
pip install torch torchvision torchaudio
pip install tokenizers
```
