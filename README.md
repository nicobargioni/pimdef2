# Monitor de Atencion Visual en Tiempo Real

Sistema de monitoreo de atencion utilizando vision por computadora con MediaPipe, visualizacion con Streamlit y procesamiento con OpenCV.

---

## Objetivo

Detectar si un usuario esta prestando atencion frente a la camara, evaluando la posicion del rostro y la orientacion de la mirada, con validacion adicional mediante segmentacion semantica para asegurar que hay una persona real en escena (y no una foto).

---

## Estructura del Proyecto

```
procesamiento-de-imagenes/
├── main.py             # Interfaz principal con Streamlit (punto de entrada)
├── detector.py         # Logica de atencion y landmarks faciales
├── segmentacion.py     # Validacion de presencia humana por segmentacion
├── graficos.py         # Visualizacion del indice de atencion
├── requirements.txt    # Lista de dependencias
└── README.md           # Documentacion del proyecto
```

---

## Librerias Utilizadas

| Libreria | Version | Uso Principal |
|----------|---------|---------------|
| **streamlit** | 1.46.1 | Interfaz web interactiva para mostrar video y graficos en tiempo real |
| **opencv-python** | 4.12.0 | Captura de video, procesamiento de imagenes, dibujo de anotaciones |
| **mediapipe** | 0.10.14 | Deteccion facial (FaceMesh) y segmentacion de personas (SelfieSegmentation) |
| **numpy** | 2.2.6 | Operaciones numericas sobre arrays (mascaras, calculos) |
| **matplotlib** | 3.10.3 | Generacion de graficos del indice de atencion |

---

## Donde se Usa Cada Libreria

### main.py (Archivo Principal)

```python
import streamlit as st          # Linea 2: Interfaz web
import cv2                      # Linea 3: Procesamiento de imagenes
import numpy as np              # Linea 4: Calculos numericos
import time                     # Linea 5: Medicion de tiempo
from collections import deque   # Linea 6: Buffer circular para historial
import mediapipe as mp          # Linea 7: Vision por computadora
```

**Uso de cada libreria en main.py:**

- **streamlit**:
  - `st.title()`, `st.markdown()` - Crear la interfaz de usuario
  - `st.session_state` - Mantener estado entre recargas (camara, modelos, contadores)
  - `st.sidebar` - Panel lateral con configuraciones
  - `st.button()` - Botones de inicio/parada
  - `st.image()` - Mostrar el video en tiempo real
  - `st.pyplot()` - Mostrar graficos de matplotlib

- **opencv (cv2)**:
  - `cv2.VideoCapture(0)` - Captura de video desde la webcam
  - `cv2.flip(frame, 1)` - Efecto espejo horizontal
  - `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` - Conversion BGR a RGB para MediaPipe
  - `cv2.putText()` - Dibujar texto sobre el frame ("ATENTO"/"NO ATENTO")

- **mediapipe**:
  - `mp.solutions.face_mesh.FaceMesh()` - Modelo de deteccion de 468 puntos faciales
  - `mp.solutions.selfie_segmentation.SelfieSegmentation()` - Modelo de segmentacion persona/fondo

- **deque** (collections):
  - `deque(maxlen=100)` - Buffer circular que guarda los ultimos 100 valores de atencion

### detector.py (Deteccion Facial y Evaluacion de Atencion)

```python
import cv2                      # Linea 7
import mediapipe as mp          # Linea 8
```

**Funciones:**

1. **`dibujar_landmarks(frame, landmarks)`** - Dibuja la malla facial sobre el video
   - Usa `mp.solutions.drawing_utils.draw_landmarks()` para dibujar las conexiones FACEMESH_TESSELATION

2. **`evaluar_atencion(landmarks, w, h, umbrales...)`** - Calcula el score de atencion (0 a 1)

### segmentacion.py (Validacion de Presencia Humana)

```python
import numpy as np              # Linea 1
import cv2                      # Linea 2
```

**Funciones:**

1. **`detectar_presencia_persona(mask, umbral=0.1)`** - Detecta si hay persona real
2. **`aplicar_mascara_segmentacion(frame, mask)`** - Superpone mascara verde sobre la persona

### graficos.py (Visualizacion)

```python
import matplotlib.pyplot as plt  # Linea 5
```

**Funcion:**

1. **`graficar_atencion(ventana_atencion, x_vals)`** - Genera grafico de linea del indice de atencion

---

## Flujo de Procesamiento

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FLUJO DEL PROGRAMA                                │
└─────────────────────────────────────────────────────────────────────────────┘

1. INICIALIZACION (al presionar "Iniciar monitoreo")
   ├── cv2.VideoCapture(0) → Abre la webcam
   ├── FaceMesh() → Carga modelo de deteccion facial (468 landmarks)
   └── SelfieSegmentation() → Carga modelo de segmentacion

2. BUCLE PRINCIPAL (por cada frame)
   │
   ├── 2.1 CAPTURA
   │   └── cap.read() → Obtiene frame de la camara
   │
   ├── 2.2 PREPROCESAMIENTO
   │   ├── cv2.flip(frame, 1) → Espejo horizontal (efecto selfie)
   │   └── cv2.cvtColor(BGR → RGB) → Conversion para MediaPipe
   │
   ├── 2.3 DETECCION FACIAL (MediaPipe FaceMesh)
   │   └── face_mesh.process(rgb) → Retorna 468 puntos del rostro
   │
   ├── 2.4 SEGMENTACION (MediaPipe SelfieSegmentation)
   │   └── segmentador.process(rgb) → Retorna mascara binaria persona/fondo
   │
   ├── 2.5 VALIDACION DE PRESENCIA
   │   └── detectar_presencia_persona(mask) → True si hay persona real
   │
   ├── 2.6 EVALUACION DE ATENCION
   │   └── evaluar_atencion(landmarks) → Score 0.0 a 1.0
   │
   ├── 2.7 CLASIFICACION
   │   ├── Score >= 0.7 → "ATENTO" (verde)
   │   └── Score < 0.7  → "NO ATENTO" (rojo)
   │
   ├── 2.8 VISUALIZACION
   │   ├── Dibujar landmarks en el frame
   │   ├── Aplicar mascara de segmentacion (verde)
   │   ├── Mostrar texto y porcentaje de atencion
   │   └── Actualizar grafico en tiempo real
   │
   └── 2.9 ESPERA
       └── time.sleep(0.03) → ~30 FPS para no saturar CPU

3. FINALIZACION (al presionar "Detener")
   ├── cap.release() → Libera la camara
   └── Muestra resumen con promedio de atencion
```

---

## Algoritmo de Evaluacion de Atencion

El sistema evalua la atencion basandose en la posicion de puntos clave del rostro:

### Landmarks Utilizados (Indices de MediaPipe FaceMesh)

| Indice | Punto Anatomico | Uso |
|--------|-----------------|-----|
| 1 | Punta de la nariz | Detectar giro horizontal de cabeza |
| 33 | Ojo derecho (exterior) | Calcular centro de ojos |
| 263 | Ojo izquierdo (exterior) | Calcular centro de ojos |
| 10 | Frente (parte superior) | Referencia para altura de nariz |
| 152 | Menton | Referencia para altura de nariz |

### Calculo del Score

```
SCORE INICIAL = 0

1. Evaluacion horizontal de nariz:
   SI nariz.x esta entre [0.4, 0.6] del frame:
      score += 0.5
   SINO:
      Advertencia: "Nariz fuera de centro"

2. Evaluacion horizontal de ojos:
   centro_ojos_x = (ojo_derecho.x + ojo_izquierdo.x) / 2
   SI centro_ojos_x esta entre [0.4, 0.6]:
      score += 0.5
   SINO:
      Advertencia: "Ojos fuera de centro"

3. Penalizacion por cabeza baja:
   nose_rel_y = (nariz.y - frente.y) / (menton.y - frente.y)
   SI nose_rel_y > 0.7:
      score = 0  (penalizacion total)
   SINO SI centro_ojos.y > 0.25:
      score -= 0.3

SCORE FINAL: 0.0 a 1.0
UMBRAL ATENCION: score >= 0.7 → "ATENTO"
```

### Diagrama Visual de Zonas

```
        ┌─────────────────────────────────┐
        │           FRAME                 │
        │  ┌─────┬─────────────┬─────┐   │
        │  │     │   ZONA DE   │     │   │
        │  │ NO  │   ATENCION  │ NO  │   │
        │  │ATENTO│  (0.4-0.6) │ATENTO│   │
        │  │     │             │     │   │
        │  └─────┴─────────────┴─────┘   │
        │    0    0.4         0.6    1   │
        └─────────────────────────────────┘
              ←── EJE X (horizontal) ──→
```

---

## Segmentacion Semantica

### Proposito

Detectar si hay una **persona real** frente a la camara (no una foto o imagen estatica).

### Funcionamiento

1. **MediaPipe SelfieSegmentation** genera una mascara donde:
   - Pixeles de persona = valores altos (cercanos a 1)
   - Pixeles de fondo = valores bajos (cercanos a 0)

2. **Validacion**: Si mas del 10% del frame tiene pixeles de "persona", se considera valido.

```python
def detectar_presencia_persona(mask, umbral=0.1):
    porcentaje_visible = np.mean(mask > 0.6)  # % de pixeles "persona"
    return porcentaje_visible > umbral        # True si hay persona real
```

3. **Visualizacion**: Se aplica una mascara verde semitransparente sobre la persona detectada.

---

## Interfaz de Usuario (Streamlit)

### Panel Principal
- Video en tiempo real con anotaciones
- Grafico de atencion actualizado por frame
- Indicador de estado: "ATENTO" / "NO ATENTO"
- Porcentaje de atencion acumulado

### Panel Lateral (Sidebar)
- **Umbrales ajustables**:
  - Giro hacia izquierda (default: 0.4)
  - Giro hacia derecha (default: 0.6)
  - Cabeza baja (default: 0.25)
- **Opciones de visualizacion**:
  - Mostrar/ocultar landmarks faciales
  - Activar/desactivar segmentacion
  - Ver mascara de segmentacion
  - Mostrar resumen al finalizar
- **Cronometro** de tiempo activo

---

## Conceptos Teoricos de Procesamiento de Imagenes

### 1. Deteccion de Landmarks Faciales (Face Mesh)

MediaPipe Face Mesh utiliza una **red neuronal convolucional (CNN)** para detectar 468 puntos 3D del rostro en tiempo real.

**Proceso:**
1. La imagen RGB entra a la red
2. Se detecta la region del rostro (face detection)
3. Se refinan 468 puntos anatomicos (landmarks)
4. Cada punto tiene coordenadas (x, y, z) normalizadas [0, 1]

### 2. Segmentacion Semantica

La segmentacion semantica clasifica cada pixel de la imagen en categorias (persona/fondo).

**Proceso:**
1. La imagen entra a una red encoder-decoder
2. El encoder extrae caracteristicas
3. El decoder genera una mascara del mismo tamaño que la imagen
4. Cada pixel tiene una probabilidad de pertenecer a "persona"

### 3. Conversion de Espacio de Color

```python
cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

- **BGR**: Formato nativo de OpenCV (Blue-Green-Red)
- **RGB**: Formato requerido por MediaPipe (Red-Green-Blue)
- La conversion intercambia los canales R y B

### 4. Flip Horizontal (Efecto Espejo)

```python
cv2.flip(frame, 1)
```

- Invierte la imagen horizontalmente
- Crea efecto "selfie" natural (moverte a la derecha se ve a la derecha)

---

## Ejecucion

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Ejecutar el sistema

```bash
streamlit run main.py
```

O alternativamente:

```bash
python -m streamlit run main.py
```

---

## Dependencias Principales

```
streamlit==1.46.1
opencv-python==4.12.0.88
mediapipe==0.10.14
numpy==2.2.6
matplotlib==3.10.3
```

---

## Creditos

Desarrollado por: **Nicolas Bargioni**

Proyecto academico - Materia: Procesamiento de Imagenes

ISSD: Inteligencia Artificial y Ciencia de Datos - Año 2025
#   p i m - f i n a l - l o c a l h o s t  
 #   p i m - f i n a l - l o c a l h o s t  
 #   p i m - f i n a l - l o c a l h o s t  
 # pimdef
