from djitellopy import Tello
import cv2, time, os, sys, signal, platform
from datetime import datetime
import math

# =======================
# DETECTAR OS Y TECLADO (Mantenido de tu código)
# =======================
OS = platform.system()
USE_PYNPUT = (OS == "Darwin")  
print(f"USE PYNPUT: {USE_PYNPUT}")
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
# CONFIGURACIÓN VISIÓN ARTIFICIAL (NUEVO)
# =======================
# 1. Configuración de AprilTag (Diccionario 16h5 recomendado para drones)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
aruco_params = cv2.aruco.DetectorParameters()

# 2. Configuración del Tracker (Seguimiento de objeto)
tracker = cv2.TrackerCSRT_create()
tracking_active = False

# 3. Variables Físicas y Cinemáticas
TAMANO_REAL_APRILTAG_CM = 10.0  # <--- CAMBIA ESTO AL TAMAÑO DE TU APRILTAG IMPRESO
factor_escala = 0.0             # cm / pixel

last_centroid = None
last_time_vel = 0
velocidad_actual = 0.0
velocidad_promedio = 0.0
sum_velocidades = 0.0
count_velocidades = 0

# =======================
# TELLO INICIALIZACIÓN
# =======================
tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

tello.streamoff()
tello.streamon()
frame_read = tello.get_frame_read()
time.sleep(2)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("images", timestamp)
os.makedirs(save_dir, exist_ok=True)

tello.takeoff()
time.sleep(2)

def safe_land():
    print("LANDING...")
    for _ in range(5):
        tello.send_rc_control(0,0,0,0)
        time.sleep(0.05)
    time.sleep(0.3)
    for _ in range(3):
        try:
            tello.land()
            print("LANDED OK")
            return
        except:
            time.sleep(0.5)
    print("FORCED EMERGENCY")
    tello.emergency()

def handler(sig, frame):
    safe_land()
    tello.streamoff()
    tello.end()
    sys.exit(0)

signal.signal(signal.SIGINT, handler)

# =======================
# LOOP PRINCIPAL
# =======================
fps = 5
interval = 1.0 / fps
last_frame_time = time.time()
frame_id = 0
speed = 40
last_rc_time = 0
rc_interval = 0.05

while True:
    frame = frame_read.frame
    # Tello entrega BGR por defecto. Si los colores se ven invertidos, descomenta la siguiente línea:
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    
    if frame is None:
        continue

    # =======================
    # TELEMETRÍA (Req. 6)
    # =======================
    altitud = tello.get_distance_tof()
    bateria = tello.get_battery()
    tiempo_vuelo = tello.get_flight_time()

    # =======================
    # DETECCIÓN APRILTAG (Req. 3 y 4)
    # =======================
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # Calcular el ancho en pixeles del marcador [0]
        p1 = corners[0][0][0] # Esquina superior izquierda
        p2 = corners[0][0][1] # Esquina superior derecha
        ancho_px = math.dist(p1, p2)
        
        if ancho_px > 0:
            # Calcular factor: cuántos centímetros reales representa 1 pixel
            factor_escala = TAMANO_REAL_APRILTAG_CM / ancho_px

    # =======================
    # TRACKING DE OBJETO Y VELOCIDAD (Req. 4 y 5)
    # =======================
    if tracking_active:
        success, bbox = tracker.update(frame)
        if success:
            # Objeto detectado: dibujar caja delimitadora
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Calcular el centro de la caja (centroide)
            cx = x + w // 2
            cy = y + h // 2
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Calcular la velocidad
            current_time = time.time()
            dt = current_time - last_time_vel
            
            if last_centroid is not None and dt > 0 and factor_escala > 0:
                dist_px = math.dist((cx, cy), last_centroid)
                dist_cm = dist_px * factor_escala
                velocidad_actual = dist_cm / dt
                
                sum_velocidades += velocidad_actual
                count_velocidades += 1
                velocidad_promedio = sum_velocidades / count_velocidades
            
            # Actualizar estado para el siguiente frame
            last_centroid = (cx, cy)
            last_time_vel = current_time
        else:
            # Medida de seguridad: Congelar mediciones ante pérdida temporal del objeto (Req. 7)
            cv2.putText(frame, "OBJETO PERDIDO", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            last_centroid = None 

    # =======================
    # OSD / INTERFAZ DE USUARIO (Req. 8)
    # =======================
    # Textos de Telemetría
    cv2.putText(frame, f"Bat: {bateria}% | Alt: {altitud}cm | Vuelo: {tiempo_vuelo}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Textos de Velocidad
    cv2.putText(frame, f"Vel. Inst: {velocidad_actual:.2f} cm/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Vel. Prom: {velocidad_promedio:.2f} cm/s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    if factor_escala > 0:
        cv2.putText(frame, f"Escala: 1px = {factor_escala:.2f}cm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "BUSCANDO APRILTAG...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, "Presiona 't' para seleccionar objeto a seguir", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # =======================
    # DISPLAY Y TECLADO
    # =======================
    cv2.imshow("Tello", frame)

    if USE_PYNPUT:
        cv2.pollKey()
        pressed = keys.copy()
    else:
        key = cv2.waitKey(1) & 0xFF
        pressed = set()
        if key != 255: pressed.add(chr(key))

    # Iniciar tracking manualmente presionando 't'
    if 't' in pressed and not tracking_active:
        # Pausa el video y permite dibujar un rectángulo con el mouse sobre el objeto
        bbox = cv2.selectROI("Tello", frame, False)
        tracker.init(frame, bbox)
        tracking_active = True
        last_time_vel = time.time()

    # =======================
    # SAVE, CONTROL Y LAND (Mantenido de tu código original)
    # =======================
    now = time.time()
    if now - last_frame_time >= interval:
        cv2.imwrite(os.path.join(save_dir, f"{frame_id:06d}.png"), frame)
        frame_id += 1
        last_frame_time = now

    lr, fb, ud, yaw = 0, 0, 0, 0
    if 'w' in pressed: fb = speed
    if 's' in pressed: fb = -speed
    if 'a' in pressed: lr = -speed
    if 'd' in pressed: lr = speed
    if 'r' in pressed: ud = speed
    if 'f' in pressed: ud = -speed
    if 'q' in pressed: yaw = -speed
    if 'e' in pressed: yaw = speed

    if now - last_rc_time > rc_interval:
        tello.send_rc_control(lr, fb, ud, yaw)
        last_rc_time = now

    if 'l' in pressed or 'esc' in pressed:
        safe_land()
        break

# =======================
# CLEANUP
# =======================
tello.streamoff()
tello.end()
cv2.destroyAllWindows()