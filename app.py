import cv2
from ultralytics import YOLO
import os
import time

# --- CONFIGURACIÓN ---c
CONFIDENCE_THRESHOLD = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX

print("🚀 INICIANDO SISTEMA DE VISION AGROJAM...")

# 1. Cargar Modelo de INVENTARIO (Humanos/Gallinas)
print("🔹 Cargando modelo Inventario...")
model_inventario = YOLO('yolov8n.pt') 

# 2. Cargar Modelo de PRODUCCIÓN (Huevos)
# Buscamos si ya descargaste el archivo. Si no, usa el base para no romper el programa.
path_huevos = 'huevos.pt' 

if os.path.exists(path_huevos):
    print("✅ Modelo de HUEVOS encontrado. Cargando...")
    model_produccion = YOLO(path_huevos)
    estado_modelo_huevos = "REAL (IA Entrenada)"
else:
    print("⚠️ AUN NO TIENES 'huevos.pt'. Usando modelo base temporal (SOLO PRUEBA).")
    model_produccion = YOLO('yolov8n.pt') # Placeholder
    estado_modelo_huevos = "MODO PRUEBA (Esperando archivo)"

# Configurar Cámara
# cap = cv2.VideoCapture(0)
address = "http://192.168.1.12:8080/video" # <--- CAMBIA LOS NUMEROS POR LOS TUYOS
cap = cv2.VideoCapture(address)
cap.set(3, 1280) # Ancho HD
cap.set(4, 720)  # Alto HD

# Variables de Estado
modo_actual = 1 # 1=Inventario, 2=Producción
color_ui = (0, 255, 0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- LOGICA DE SELECCION DE MODELO ---
    if modo_actual == 1:
        # MODO INVENTARIO (Detectar Personas simulando Gallinas)
        active_model = model_inventario
        classes_to_detect = [0] # Clase 0 es Persona en COCO
        mode_text = "MODO: MONITORIZACION (GALLINAS/HUMANOS)"
        ui_color = (0, 255, 255) # Amarillo
        
    elif modo_actual == 2:
        # MODO PRODUCCION (Detectar Huevos)
        active_model = model_produccion
        if estado_modelo_huevos == "REAL (IA Entrenada)":
            classes_to_detect = None # Detectar todo lo que aprendió (huevos)
        else:
            classes_to_detect = [0] # Si es prueba, detectamos personas para que veas algo
            
        mode_text = "MODO: CONTROL DE CALIDAD (HUEVOS)"
        ui_color = (255, 0, 255) # Magenta

    # --- INFERENCIA ---
    # Hacemos la predicción
    results = active_model.predict(frame, classes=classes_to_detect, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    # Obtenemos imagen dibujada y conteo
    annotated_frame = results[0].plot()
    conteo = len(results[0].boxes)

    # --- INTERFAZ GRAFICA (DASHBOARD) ---
    # Barra superior negra
    cv2.rectangle(annotated_frame, (0,0), (1280, 100), (0,0,0), -1)
    
    # Textos
    cv2.putText(annotated_frame, "AGROJAM VISION SYSTEM v1.0", (20, 30), FONT, 0.6, (150,150,150), 1)
    cv2.putText(annotated_frame, mode_text, (20, 70), FONT, 1, (255,255,255), 2)
    
    # Contador gigante a la derecha
    cv2.putText(annotated_frame, f"CANTIDAD: {conteo}", (900, 70), FONT, 1.5, ui_color, 3)

    # Instrucciones pie de página
    cv2.putText(annotated_frame, "[TECLA 1] Inventario | [TECLA 2] Huevos | [Q] Salir", (20, 700), FONT, 0.6, (255,255,255), 2)

    # Si estamos en modo huevos pero aun no tienes el archivo
    if modo_actual == 2 and estado_modelo_huevos != "REAL (IA Entrenada)":
         cv2.putText(annotated_frame, "⚠️ ESPERANDO ARCHIVO 'huevos.pt'", (300, 360), FONT, 1.2, (0,0,255), 3)

    # Mostrar imagen
    cv2.imshow("Agrojam Demo", annotated_frame)

    # Control de Teclas
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('1'): modo_actual = 1
    if key == ord('2'): modo_actual = 2

cap.release()
cv2.destroyAllWindows()
