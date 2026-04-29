import cv2
import time
import math

# =======================
# CONFIGURACIÓN VISIÓN ARTIFICIAL
# =======================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
aruco_params = cv2.aruco.DetectorParameters()

try:
    tracker = cv2.TrackerCSRT_create()
except AttributeError:
    tracker = cv2.legacy.TrackerCSRT_create()
    
tracking_active = False

TAMANO_REAL_APRILTAG_CM = 10.0  # <--- AJUSTA ESTO AL TAMAÑO DE TU APRILTAG IMPRESO
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
            
            if last_centroid is not None and dt > 0 and factor_escala > 0:
                dist_px = math.dist((cx, cy), last_centroid)
                dist_cm = dist_px * factor_escala
                velocidad_actual = dist_cm / dt
                
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
    cv2.putText(frame, f"Vel. Inst: {velocidad_actual:.2f} cm/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"Vel. Prom: {velocidad_promedio:.2f} cm/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    if factor_escala > 0:
        cv2.putText(frame, f"Escala: 1px = {factor_escala:.2f}cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "BUSCANDO APRILTAG...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Prueba Vision - Tarea 1", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('t') and not tracking_active:
        bbox = cv2.selectROI("Prueba Vision - Tarea 1", frame, False)
        tracker.init(frame, bbox)
        tracking_active = True
        last_time_vel = time.time()
    
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
