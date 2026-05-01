import cv2
import time
import math
import numpy as np

# =======================
# 1. CONFIGURACIÓN VISIÓN ARTIFICIAL
# =======================
# Diccionario 36h11 detectado en pruebas previas
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()

TAMANO_REAL_APRILTAG_CM = 10.0  
factor_escala = 0.0             

last_centroid = None
last_time_vel = 0
velocidad_actual = 0.0
velocidad_promedio = 0.0
sum_velocidades = 0.0
count_velocidades = 0

# Filtro de color HSV para objeto AZUL (100% Autónomo)
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# Variables para simular la telemetría del Dron
inicio_simulacion = time.time()
bateria_simulada = 85

# =======================
# 2. INICIAR WEBCAM (Simulador Tello)
# =======================
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara simulada.")
    exit()

print("SIMULADOR INICIADO. Buscando AprilTag y Objeto Azul...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =======================
    # 3. DETECCIÓN APRILTAG Y ESCALAMIENTO MEJORADO (Req. 3 y 4)
    # =======================
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # Extraer las 4 esquinas del AprilTag para reducir distorsión de perspectiva
        p_arriba_izq = corners[0][0][0]
        p_arriba_der = corners[0][0][1]
        p_abajo_der = corners[0][0][2]
        p_abajo_izq = corners[0][0][3]
        
        # Calcular los 4 lados para evitar errores si la cámara está chueca
        ancho_sup = math.dist(p_arriba_izq, p_arriba_der)
        ancho_inf = math.dist(p_abajo_izq, p_abajo_der)
        alto_izq = math.dist(p_arriba_izq, p_abajo_izq)
        alto_der = math.dist(p_arriba_der, p_abajo_der)
        
        # Sacar el promedio de los 4 lados
        ancho_px_promedio = (ancho_sup + ancho_inf + alto_izq + alto_der) / 4.0
        
        if ancho_px_promedio > 0:
            factor_escala = TAMANO_REAL_APRILTAG_CM / ancho_px_promedio

    # Memoria de escala: Permite mantener la calibración aunque el tag se pierda unos frames
    if factor_escala > 0:
        escala_usada = factor_escala
        unidad = "cm/s"
    else:
        escala_usada = 1.0
        unidad = "px/s"

    # =======================
    # 4. RASTREO AUTÓNOMO Y VELOCIDAD (Req. 4 y 5)
    # =======================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objeto_detectado = False

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        
        # Filtrar ruido de fondo, solo detecta objetos azules decentemente grandes
        if area > 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            cx = x + w // 2
            cy = y + h // 2
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            objeto_detectado = True
            
            current_time = time.time()
            dt = current_time - last_time_vel
            
            if dt > 0.15: 
                if last_centroid is not None:
                    dist_px = math.dist((cx, cy), last_centroid)
                    dist_fisica = dist_px * escala_usada
                    velocidad_actual = dist_fisica / dt
                    
                    sum_velocidades += velocidad_actual
                    count_velocidades += 1
                    velocidad_promedio = sum_velocidades / count_velocidades
                
                last_centroid = (cx, cy)
                last_time_vel = current_time
    # =======================
    # 5. MEDIDA DE SEGURIDAD (Req. 7)
    # =======================
    if not objeto_detectado:
        # Congelar mediciones ante pérdida temporal del objeto
        cv2.putText(frame, "ALERTA: OBJETO PERDIDO - MEDICION PAUSADA", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        last_centroid = None 
        velocidad_actual = 0.0 # Se congela la instantánea, el promedio se mantiene

    # =======================
    # 6. OSD / LECTURA DE ESTADOS INTERNOS (Req. 6 y 8)
    # =======================
    # Telemetría Simulada
    tiempo_vuelo_simulado = int(time.time() - inicio_simulacion)
    altitud_simulada = 120 # cm constantes para la simulación
    
    cv2.putText(frame, f"[SIM] Bat: {bateria_simulada}% | Alt: {altitud_simulada}cm | Vuelo: {tiempo_vuelo_simulado}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Datos de Velocidad
    cv2.putText(frame, f"Vel. Inst: {velocidad_actual:.2f} {unidad}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"Vel. Prom: {velocidad_promedio:.2f} {unidad}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Estado del AprilTag
    if factor_escala > 0:
        cv2.putText(frame, f"Escala OK: 1px = {factor_escala:.2f}cm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "SIN APRILTAG", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Simulador Camara Dron - Tarea 1", frame)

    # Botón de emergencia por teclado (Req. 7)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("Emergencia activada. Aterrizaje simulado...")
        break

cap.release()
cv2.destroyAllWindows()