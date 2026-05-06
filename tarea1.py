from djitellopy import Tello
import cv2
import time
import math
import numpy as np
import os
import platform

# =======================
# 1. CONFIGURACIÓN Y TELLO
# =======================
tello = Tello()
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()
print(f"Batería inicial: {tello.get_battery()}%")

# Configuración Visión Artificial
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()
TAMANO_REAL_APRILTAG_CM = 10.0  

# Filtro Azul
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# Variables de Medición
last_centroid = None
last_time_vel = time.time()
velocidad_actual = 0.0
desplazamiento_total_cm = 0.0
tiempo_total_movimiento = 0.0
inicio_movimiento = None
factor_escala = 0.0

# =======================
# 2. FUNCIONES DE APOYO
# =======================
def safe_land():
    print("Aterrizando de emergencia...")
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()

# =======================
# 3. LOOP PRINCIPAL
# =======================
speed = 40
tello.takeoff() # Despegue inicial automático

try:
    while True:
        # Captura de video del Tello
        img = frame_read.frame
        if img is None: continue
        
        # El Tello entrega RGB, OpenCV usa BGR para procesar/mostrar
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w, _ = frame.shape

        # --- DETECCIÓN APRILTAG (Calibración) ---
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
        escala_usada = 1.0
        unidad = "px"

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            esquinas_ref = corners[0][0]
            # Cálculo de escala promediando lados para precisión
            ancho_px = (math.dist(esquinas_ref[0], esquinas_ref[1]) + 
                        math.dist(esquinas_ref[1], esquinas_ref[2])) / 2
            if ancho_px > 0:
                factor_escala = TAMANO_REAL_APRILTAG_CM / ancho_px
                escala_usada = factor_escala
                unidad = "cm"

        # --- RASTREO OBJETO AZUL ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objeto_detectado = False

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 600:
                bx, by, bw, bh = cv2.boundingRect(c)
                cx, cy = bx + bw // 2, by + bh // 2
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                objeto_detectado = True
                
                now = time.time()
                if last_centroid is not None:
                    dt = now - last_time_vel
                    dist_real = math.dist((cx, cy), last_centroid) * escala_usada
                    
                    # Filtro de histéresis (mínimo 1.5 cm de movimiento)
                    if dist_real > 1.5:
                        desplazamiento_total_cm += dist_real
                        velocidad_actual = dist_real / dt
                        if inicio_movimiento is None: inicio_movimiento = now
                        tiempo_total_movimiento = now - inicio_movimiento
                    else:
                        velocidad_actual = 0.0
                
                last_centroid = (cx, cy)
                last_time_vel = now

        # --- INTERFAZ (OSD) ---
        vel_promedio = desplazamiento_total_cm / tiempo_total_movimiento if tiempo_total_movimiento > 0 else 0
        
        # Datos Dron Reales
        cv2.putText(frame, f"BAT: {tello.get_battery()}% | ALT: {tello.get_height()}cm", (10, 30), 2, 0.6, (0, 255, 0), 2)
        # Mediciones
        cv2.putText(frame, f"V. Inst: {velocidad_actual:.2f} {unidad}/s", (10, 60), 2, 0.6, (255, 0, 255), 2)
        cv2.putText(frame, f"V. Prom: {vel_promedio:.2f} {unidad}/s", (10, 90), 2, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Dist: {desplazamiento_total_cm:.2f} {unidad}", (10, 120), 2, 0.6, (255, 255, 255), 2)
        
        if not objeto_detectado:
            cv2.putText(frame, "ALERTA: OBJETO PERDIDO", (10, 180), 2, 0.7, (0, 0, 255), 2)
            last_centroid = None

        cv2.imshow("Tello Control & Vision - UOH", frame)

        # --- CONTROL DE TECLADO ---
        key = cv2.waitKey(1) & 0xFF
        lr, fb, ud, yaw = 0, 0, 0, 0
        
        if key == ord('w'): fb = speed
        elif key == ord('s'): fb = -speed
        if key == ord('a'): lr = -speed
        elif key == ord('d'): lr = speed
        if key == ord('r'): ud = speed
        elif key == ord('f'): ud = -speed
        if key == ord('q'): yaw = -speed
        elif key == ord('e'): yaw = speed
        
        tello.send_rc_control(lr, fb, ud, yaw)

        # Salida y Aterrizaje
        if key == ord('l') or key == 27: # 'l' o ESC
            break

finally:
    safe_land()