import cv2
import time
import math
import numpy as np

# =======================
# 1. CONFIGURACION VISION ARTIFICIAL
# =======================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()

TAMANO_REAL_APRILTAG_CM = 10.0  
factor_escala = 0.0             

# Variables de seguimiento y calculo
last_centroid = None
last_time_vel = 0
velocidad_actual = 0.0
desplazamiento_total_cm = 0.0
tiempo_total_movimiento = 0.0
inicio_movimiento = None

# Filtro de color HSV para objeto AZUL
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

inicio_simulacion = time.time()
bateria_simulada = 85

# =======================
# 2. INICIAR WEBCAM
# =======================
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: No se pudo abrir la camara.")
    exit()

print("SISTEMA INICIADO. Calibrando con AprilTag...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =======================
    # 3. DETECCION Y ESCALAMIENTO (Req. 3 y 4)
    # =======================
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    
    escala_usada = 1.0
    unidad = "px"

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # --- CALCULO DE ESCALA ---
        esquinas_ref = corners[0][0]
        lado1 = math.dist(esquinas_ref[0], esquinas_ref[1])
        lado2 = math.dist(esquinas_ref[1], esquinas_ref[2])
        ancho_px_tag = (lado1 + lado2) / 2
        
        if ancho_px_tag > 0:
            factor_escala = TAMANO_REAL_APRILTAG_CM / ancho_px_tag
            escala_usada = factor_escala
            unidad = "cm"

        # --- TRAZADO DE DISTANCIA ENTRE TAGS (Si hay 2 o mas) ---
        if len(ids) >= 2:
            # Centro del Tag 1
            c1 = int(np.mean(corners[0][0][:, 0])), int(np.mean(corners[0][0][:, 1]))
            # Centro del Tag 2
            c2 = int(np.mean(corners[1][0][:, 0])), int(np.mean(corners[1][0][:, 1]))
            
            # Dibujar linea entre tags
            cv2.line(frame, c1, c2, (0, 255, 255), 2)
            
            # Calcular distancia fisica entre los dos tags
            dist_puntos_px = math.dist(c1, c2)
            dist_puntos_cm = dist_puntos_px * escala_usada
            
            # Mostrar texto de la distancia en el punto medio
            mid_x, mid_y = (c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2
            cv2.putText(frame, f"{dist_puntos_cm:.1f} cm", (mid_x - 30, mid_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # =======================
    # 4. RASTREO Y FILTRO DE PRECISION (Req. 5)
    # =======================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objeto_detectado = False

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 600:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            objeto_detectado = True
            
            current_time = time.time()
            if last_centroid is not None:
                dt = current_time - last_time_vel
                dist_px = math.dist((cx, cy), last_centroid)
                dist_real = dist_px * escala_usada
                
                # UMBRAL DE MOVIMIENTO: Filtro para evitar ruido
                if dist_real > 1.5: 
                    desplazamiento_total_cm += dist_real
                    velocidad_actual = dist_real / dt
                    
                    if inicio_movimiento is None:
                        inicio_movimiento = current_time
                    tiempo_total_movimiento = current_time - inicio_movimiento
                else:
                    velocidad_actual = 0.0

            last_centroid = (cx, cy)
            last_time_vel = current_time

    # =======================
    # 5. OSD Y VELOCIDAD PROMEDIO (Req. 6, 7 y 8)
    # =======================
    vel_promedio = desplazamiento_total_cm / tiempo_total_movimiento if tiempo_total_movimiento > 0 else 0.0

    if not objeto_detectado:
        cv2.putText(frame, "ALERTA: OBJETO PERDIDO", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        last_centroid = None 
        velocidad_actual = 0.0

    # Telemetria simulada
    t_vuelo = int(time.time() - inicio_simulacion)
    cv2.putText(frame, f"[SIM] Bat: {bateria_simulada}% | Alt: 120cm | Vuelo: {t_vuelo}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Resultados de medicion
    cv2.putText(frame, f"Vel. Inst: {velocidad_actual:.2f} {unidad}/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(frame, f"Vel. Prom: {vel_promedio:.2f} {unidad}/s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Desplazamiento: {desplazamiento_total_cm:.2f} {unidad}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Estado de calibracion
    msg = f"Escala: 1px = {factor_escala:.3f}cm" if factor_escala > 0 else "BUSCANDO APRILTAG..."
    cv2.putText(frame, msg, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Prueba de Camara - UOH", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()