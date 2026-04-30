import cv2
import time
import math

# =======================
# CONFIGURACIÓN VISIÓN ARTIFICIAL
# =======================
# 1. ACTUALIZADO: Diccionario 36h11 (el que tienes en la pantalla de tu celular)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()

try:
    tracker = cv2.TrackerCSRT_create()
except AttributeError:
    tracker = cv2.legacy.TrackerCSRT_create()
    
tracking_active = False

TAMANO_REAL_APRILTAG_CM = 10.0  # <--- Si en tu celular mide distinto a 10cm, cámbialo aquí
factor_escala = 0.0             

last_centroid = None
last_time_vel = 0
velocidad_actual = 0.0
velocidad_promedio = 0.0
sum_velocidades = 0.0
count_velocidades = 0

# =======================
# INICIAR WEBCAM
# =======================
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Cámara iniciada correctamente.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =======================
    # DETECCIÓN APRILTAG
    # =======================
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        p1 = corners[0][0][0]
        p2 = corners[0][0][1]
        ancho_px = math.dist(p1, p2)
        
        if ancho_px > 0:
            factor_escala = TAMANO_REAL_APRILTAG_CM / ancho_px

    # === MODO DEPURACIÓN: SI NO HAY APRILTAG, USAR PIXELES ===
    escala_usada = factor_escala if factor_escala > 0 else 1.0
    unidad = "cm/s" if factor_escala > 0 else "px/s"

    # =======================
    # TRACKING DE OBJETO Y VELOCIDAD
    # =======================
    if tracking_active:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            cx = x + w // 2
            cy = y + h // 2
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            current_time = time.time()
            dt = current_time - last_time_vel
            
            if last_centroid is not None and dt > 0:
                dist_px = math.dist((cx, cy), last_centroid)
                dist_fisica = dist_px * escala_usada
                velocidad_actual = dist_fisica / dt
                
                sum_velocidades += velocidad_actual
                count_velocidades += 1
                velocidad_promedio = sum_velocidades / count_velocidades
            
            last_centroid = (cx, cy)
            last_time_vel = current_time
        else:
            cv2.putText(frame, "OBJETO PERDIDO", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            last_centroid = None 

    # =======================
    # OSD / INTERFAZ DE USUARIO
    # =======================
    cv2.putText(frame, f"Vel. Inst: {velocidad_actual:.2f} {unidad}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"Vel. Prom: {velocidad_promedio:.2f} {unidad}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    if factor_escala > 0:
        cv2.putText(frame, f"Escala OK: 1px = {factor_escala:.2f}cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "SIN APRILTAG (Midiendo en Pixeles)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Prueba Vision - Tarea 1", frame)

    key = cv2.waitKey(1) & 0xFF

    # =======================
    # SELECCIÓN Y ESCUDO DE SEGURIDAD
    # =======================
    if key == ord('t') and not tracking_active:
        bbox = cv2.selectROI("Prueba Vision - Tarea 1", frame, False)
        
        # Solo inicia si dibujaste un rectángulo válido (ancho y alto mayores a 0)
        if bbox[2] > 0 and bbox[3] > 0:
            tracker.init(frame, bbox)
            tracking_active = True
            last_time_vel = time.time()
        else:
            print("Selección vacía. Vuelve a presionar 't' y arrastra el mouse.")
    
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()