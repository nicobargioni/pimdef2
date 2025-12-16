# === Importación de librerías ===
import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp
import threading
import av

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# === Importación de módulos personalizados ===
from detector import evaluar_atencion, dibujar_landmarks
from segmentacion import detectar_presencia_persona, aplicar_mascara_segmentacion
from graficos import graficar_atencion

# Configuración para servidores TURN/STUN (necesario para cloud)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------- Estado compartido (thread-safe) ----------
class AttentionState:
    """Clase para manejar estado compartido entre threads de forma segura."""
    def __init__(self):
        self.lock = threading.Lock()
        self.total_frames = 0
        self.atencion_frames = 0
        self.ventana_atencion = deque(maxlen=100)
        self.x_vals = deque(maxlen=100)
        self.attention_log = []
        self.last_attention_index = 0
        self.start_time = None

    def reset(self):
        with self.lock:
            self.total_frames = 0
            self.atencion_frames = 0
            self.ventana_atencion.clear()
            self.x_vals.clear()
            self.attention_log.clear()
            self.last_attention_index = 0
            self.start_time = time.time()

    def update(self, is_attentive):
        with self.lock:
            self.total_frames += 1
            if is_attentive:
                self.atencion_frames += 1

            if self.total_frames > 0:
                atencion_index = int((self.atencion_frames / self.total_frames) * 100)
                self.last_attention_index = atencion_index
                self.ventana_atencion.append(atencion_index)
                self.x_vals.append(self.total_frames)
                self.attention_log.append(atencion_index)

    def get_stats(self):
        with self.lock:
            return {
                'total_frames': self.total_frames,
                'atencion_frames': self.atencion_frames,
                'attention_index': self.last_attention_index,
                'ventana_atencion': list(self.ventana_atencion),
                'x_vals': list(self.x_vals),
                'attention_log': list(self.attention_log),
                'start_time': self.start_time
            }

# Inicializar estado global
if 'attention_state' not in st.session_state:
    st.session_state.attention_state = AttentionState()

# ---------- Procesador de Video ----------
class AttentionVideoProcessor:
    """Procesador de video que analiza la atención en cada frame."""

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.segmentador = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

        # Configuración (se actualiza desde la UI)
        self.umbral_giro_izquierda = 0.4
        self.umbral_giro_derecha = 0.6
        self.umbral_ojos_y_baja = 0.25
        self.mostrar_landmarks = True
        self.usar_segmentacion = True
        self.ver_mascara_segmentacion = True

        # Referencia al estado compartido
        self.state = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Espejo horizontal

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        segment = self.segmentador.process(rgb)
        h, w, _ = img.shape

        score = 0
        hay_persona = True
        texto = "Sin rostro"
        color = (150, 150, 255)
        is_attentive = False

        # Validación de persona real
        if self.usar_segmentacion:
            hay_persona = detectar_presencia_persona(segment.segmentation_mask)

        # Procesamiento facial
        if hay_persona and results.multi_face_landmarks:
            for rostro in results.multi_face_landmarks:
                if self.mostrar_landmarks:
                    img = dibujar_landmarks(img, rostro)

                score, _ = evaluar_atencion(
                    rostro, w, h,
                    self.umbral_giro_izquierda,
                    self.umbral_giro_derecha,
                    self.umbral_ojos_y_baja
                )

            if score >= 0.7:
                is_attentive = True
                texto = "ATENTO"
                color = (0, 255, 0)
            else:
                texto = "NO ATENTO"
                color = (0, 0, 255)
        elif not hay_persona:
            texto = "Sin persona detectada"
            color = (150, 150, 255)

        # Actualizar estado
        if self.state:
            self.state.update(is_attentive)
            stats = self.state.get_stats()
            atencion_index = stats['attention_index']
        else:
            atencion_index = 0

        # Anotaciones visuales
        cv2.putText(img, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(img, f"Atencion: {atencion_index}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Máscara de segmentación
        if self.ver_mascara_segmentacion and self.usar_segmentacion:
            img = aplicar_mascara_segmentacion(img, segment.segmentation_mask)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------- UI ----------
st.title("Monitor de Atención Visual en Tiempo Real")
st.markdown("""
Este programa utiliza visión por computadora para analizar tu nivel de atención durante una videollamada.
Evalúa si tu rostro está centrado y si tu mirada se mantiene hacia el frente.
Ideal para contextos educativos, de trabajo remoto o validación de presencia.

A través de la webcam, el sistema detecta si desviás la mirada, girás la cabeza o bajás la vista,
y muestra un indicador visual de atención junto a un gráfico en tiempo real.

### Instrucciones:
1. Permitir el acceso a la cámara cuando el navegador lo solicite.
2. Ajustar los umbrales de atención en la barra lateral según tu preferencia.
3. Presionar "START" para comenzar a evaluar tu atención.
4. Observar el indicador de atención y el gráfico en tiempo real.
5. Presionar "STOP" cuando quieras detener el monitoreo.
""")
st.subheader("La premisa es la siguiente")
st.markdown("Para demostrar tu atención, procurá estar justo en medio de donde te muestra la cámara")

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Umbrales de Atención")

    with st.expander("Ajustes de Umbrales", expanded=False):
        st.markdown("""
        Ajustá la sensibilidad del sistema de atención:
        - **Giro izquierda/derecha**: margen de movimiento horizontal permitido.
        - **Cabeza baja**: inclinación vertical antes de penalizar.
        """)
        umbral_giro_izquierda = st.slider("Giro hacia izquierda", 0.0, 1.0, 0.4, step=0.01)
        umbral_giro_derecha = st.slider("Giro hacia derecha", 0.0, 1.0, 0.6, step=0.01)
        umbral_ojos_y_baja = st.slider("Cabeza baja", 0.0, 1.0, 0.25, step=0.01)

    st.markdown("---")
    st.subheader("Configuración")

    mostrar_landmarks = st.checkbox("Mostrar landmarks faciales", value=True)
    st.caption("Visualiza los puntos y líneas sobre tu rostro (FaceMesh).")

    usar_segmentacion = st.checkbox("Activar segmentación semántica", value=True)
    st.caption("Valida que haya una persona real (no una imagen).")

    ver_mascara_segmentacion = st.checkbox("Ver máscara de segmentación", value=True)
    st.caption("Superpone una máscara verde sobre la persona detectada.")

    mostrar_grafico = st.checkbox("Mostrar gráfico en tiempo real", value=True)
    st.caption("Muestra el gráfico de atención mientras monitorea.")

    if st.button("Reiniciar estadísticas"):
        st.session_state.attention_state.reset()
        st.rerun()

# ---------- WebRTC Streamer ----------
def video_processor_factory():
    processor = AttentionVideoProcessor()
    processor.state = st.session_state.attention_state
    processor.umbral_giro_izquierda = umbral_giro_izquierda
    processor.umbral_giro_derecha = umbral_giro_derecha
    processor.umbral_ojos_y_baja = umbral_ojos_y_baja
    processor.mostrar_landmarks = mostrar_landmarks
    processor.usar_segmentacion = usar_segmentacion
    processor.ver_mascara_segmentacion = ver_mascara_segmentacion
    return processor

col1, col2 = st.columns(2)

with col1:
    ctx = webrtc_streamer(
        key="attention-monitor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=video_processor_factory,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ---------- Gráfico en tiempo real ----------
with col2:
    if mostrar_grafico:
        grafico_placeholder = st.empty()
        stats_placeholder = st.empty()

# Actualizar gráfico periódicamente
if ctx.state.playing and mostrar_grafico:
    stats = st.session_state.attention_state.get_stats()

    if stats['ventana_atencion'] and stats['x_vals']:
        fig = graficar_atencion(stats['ventana_atencion'], stats['x_vals'])
        with col2:
            grafico_placeholder.pyplot(fig)

        # Mostrar estadísticas
        with col2:
            if stats['start_time']:
                elapsed = int(time.time() - stats['start_time'])
                minutes, seconds = divmod(elapsed, 60)
                stats_placeholder.markdown(f"""
                **Estadísticas actuales:**
                - Tiempo: {minutes:02d}:{seconds:02d}
                - Frames procesados: {stats['total_frames']}
                - Atención promedio: {stats['attention_index']}%
                """)

# ---------- Resumen final ----------
if not ctx.state.playing:
    stats = st.session_state.attention_state.get_stats()
    if stats['attention_log']:
        st.subheader("Resumen de atención")
        promedio_total = sum(stats['attention_log']) / len(stats['attention_log'])
        st.markdown(f"Promedio total: **{promedio_total:.2f}%**")

        if stats['ventana_atencion'] and stats['x_vals']:
            fig_final = graficar_atencion(stats['ventana_atencion'], stats['x_vals'])
            st.pyplot(fig_final)

# ---------- Footer fijo ----------
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f9f9f9;
            color: #666;
            text-align: center;
            font-size: 0.85em;
            padding: 0.5em 0;
            border-top: 1px solid #ddd;
        }
    </style>
    <div class="footer">
        Desarrollado por <strong>Nicolás Bargioni</strong> | Año 2025 | ISSD: Inteligencia Artificial y Ciencia de Datos
    </div>
    """,
    unsafe_allow_html=True
)
