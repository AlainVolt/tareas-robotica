from djitellopy import Tello
import cv2, time, os, sys, signal, platform, math
import numpy as np
from datetime import datetime

# =======================
# 1. CONFIGURACIÓN INICIAL Y OS
# =======================
OS = platform.system()
USE_PYNPUT = (OS == "Darwin")
if USE_PYNPUT:
    from pynput import keyboard
    keys = set()
    def on_press(key):
        try: keys.add(key.char)
        except: 
            if key == keyboard.Key.esc: keys.add('esc')
    def on_release(key):
        try: keys.discard(key.char)
        except:
            if key == keyboard.Key.esc: keys.discard('esc')
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

# =======================
# 2. CONFIGURACIÓN VISIÓN ARTIFICIAL
# =======================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()
TAMANO_REAL_APRILTAG_CM = 10.0  

# Filtro de color objeto AZUL
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# Variables de medición
last_centroid = None
last_time_vel = 0
velocidad_actual = 0.0
desplazamiento_total_cm = 0.0
tiempo_total_movimiento = 0.0
inicio_movimiento = None
factor_escala = 0.0

# =======================
# 3. CONEXIÓN TELLO
# =======================
tello = Tello()
tello.connect()
print(f"Batería Dron: {tello.get_battery()}%")
tello.streamon()
frame_read = tello.get_frame_read()
time.sleep(2)

# Carpeta de guardado
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("images", timestamp)
os.makedirs(save_dir, exist_ok=True)

# =======================
# 4. FUNCIONES DE SEGURIDAD
# =======================
def safe_land():
    print("INICIANDO ATERRIZAJE SEGURO...")
    for _ in range(5):
        tello.send_rc_control(0,0,0,0)
        time.sleep(0.05)
    try:
        tello.land()
        print("ATERRIZAJE EXITOSO")
    except:
        tello.emergency()

def handler(sig, frame):
    safe_land()
    tello.streamoff()
    tello.end()
    sys.exit(0)

signal.signal(signal.SIGINT, handler)

# =======================
# 5. LOOP DE OPERACIÓN Y VISIÓN
# =======================
tello.takeoff() # Despegue automático inicial
speed = 40
last_frame_time = time.time()
last_rc_time = 0
rc_interval = 0.05
frame_id = 0

while True:
    # Captura de frame real del Tello
    frame_raw = frame_read.frame
    if frame_raw is None: continue
    
    # Convertir RGB (Tello) a BGR (OpenCV)
    frame = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)

    # --- DETECCIÓN Y ESCALAMIENTO (Req. 3 y 4) ---
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    escala_usada = 1.0
    unidad = "px"

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # Escala basada en el primer tag detectado
        esquinas_ref = corners[0][0]
        ancho_px = (math.dist(esquinas_ref[0], esquinas_ref[1]) + math.dist(esquinas_ref[1], esquinas_ref[2])) / 2
        if ancho_px > 0:
            factor_escala = TAMANO_REAL_APRILTAG_CM / ancho_px
            escala_usada = factor_escala
            unidad = "cm"

        # Trazo de distancia entre tags (si hay 2 o más)
        if len(ids) >= 2:
            c1 = int(np.mean(corners[0][0][:, 0])), int(np.mean(corners[0][0][:, 1]))
            c2 = int(np.mean(corners[1][0][:, 0])), int(np.mean(corners[1][0][:, 1]))
            cv2.line(frame, c1, c2, (0, 255, 255), 2)
            dist_cm = math.dist(c1, c2) * escala_usada
            mid_p = ((c1[0]+c2[0])//2, (c1[1]+c2[1])//2)
            cv2.putText(frame, f"{dist_cm:.1f} cm", mid_p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- RASTREO OBJETO AZUL Y VELOCIDAD (Req. 4 y 5) ---
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
            
            now_t = time.time()
            if last_centroid is not None:
                dt = now_t - last_time_vel
                dist_real = math.dist((cx, cy), last_centroid) * escala_usada
                if dist_real > 1.5: # Filtro de ruido
                    desplazamiento_total_cm += dist_real
                    velocidad_actual = dist_real / dt
                    if inicio_movimiento is None: inicio_movimiento = now_t
                    tiempo_total_movimiento = now_t - inicio_movimiento
                else:
                    velocidad_actual = 0.0
            last_centroid = (cx, cy)
            last_time_vel = now_t

    # --- CÁLCULO PROMEDIO Y OSD (Req. 6, 7 y 8) ---
    vel_prom = desplazamiento_total_cm / tiempo_total_movimiento if tiempo_total_movimiento > 0 else 0.0
    
    # Telemetría Real del Dron
    cv2.putText(frame, f"BAT: {tello.get_battery()}% | ALT: {tello.get_height()}cm", (10, 30), 2, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"V. Inst: {velocidad_actual:.2f} {unidad}/s", (10, 60), 2, 0.6, (255, 0, 255), 2)
    cv2.putText(frame, f"V. Prom: {vel_prom:.2f} {unidad}/s", (10, 90), 2, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Distancia: {desplazamiento_total_cm:.2f} {unidad}", (10, 120), 2, 0.6, (255, 255, 255), 2)

    if not objeto_detectado:
        cv2.putText(frame, "OBJETO PERDIDO", (10, 160), 2, 0.6, (0, 0, 255), 2)
        last_centroid = None

    cv2.imshow("Control Tello & Vision Artificial - UOH", frame)

    # --- GESTIÓN DE ENTRADAS (Teclado) ---
    if USE_PYNPUT:
        cv2.pollKey()
        pressed = keys.copy()
    else:
        key = cv2.waitKey(1) & 0xFF
        pressed = set()
        if key != 255: pressed.add(chr(key))

    if 'l' in pressed or 'esc' in pressed:
        safe_land()
        break

    # Guardado de frames (Req. 2)
    curr_now = time.time()
    if curr_now - last_frame_time >= (1.0/5.0):
        cv2.imwrite(os.path.join(save_dir, f"{frame_id:06d}.png"), frame)
        frame_id += 1
        last_frame_time = curr_now

    # Control RC
    lr, fb, ud, yaw = 0, 0, 0, 0
    if 'w' in pressed: fb = speed
    if 's' in pressed: fb = -speed
    if 'a' in pressed: lr = -speed
    if 'd' in pressed: lr = speed
    if 'r' in pressed: ud = speed
    if 'f' in pressed: ud = -speed
    if 'q' in pressed: yaw = -speed
    if 'e' in pressed: yaw = speed

    if curr_now - last_rc_time > rc_interval:
        tello.send_rc_control(lr, fb, ud, yaw)
        last_rc_time = curr_now

# Limpieza final
tello.streamoff()
tello.end()
cv2.destroyAllWindows()