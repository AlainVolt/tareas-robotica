from djitellopy import Tello
import cv2, time, os, sys, signal, platform
from datetime import datetime

# =======================
# DETECTAR OS
# =======================
OS = platform.system()
USE_PYNPUT = (OS == "Darwin")  # macOS
print(f"USE PYNPUT: {USE_PYNPUT}")
if USE_PYNPUT:
    from pynput import keyboard
    keys = set()

    def on_press(key):
        try:
            keys.add(key.char)
        except:
            if key == keyboard.Key.esc:
                keys.add('esc')

    def on_release(key):
        try:
            keys.discard(key.char)
        except:
            if key == keyboard.Key.esc:
                keys.discard('esc')

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

# =======================
# TELLO
# =======================
tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

tello.streamoff()
tello.streamon()
frame_read = tello.get_frame_read()
time.sleep(2)

# =======================
# SAVE DIR
# =======================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("images", timestamp)
os.makedirs(save_dir, exist_ok=True)

# =======================
# TAKEOFF
# =======================
tello.takeoff()
time.sleep(2)

# =======================
# SAFE LAND FUNCTION
# =======================
def safe_land():
    print("LANDING...")

    # detener rc
    for _ in range(5):
        tello.send_rc_control(0,0,0,0)
        time.sleep(0.05)

    time.sleep(0.3)

    # intentar land varias veces
    for _ in range(3):
        try:
            tello.land()
            print("LANDED OK")
            return
        except:
            time.sleep(0.5)

    print("FORCED EMERGENCY")
    tello.emergency()

# =======================
# CTRL+C
# =======================
def handler(sig, frame):
    safe_land()
    tello.streamoff()
    tello.end()
    sys.exit(0)

signal.signal(signal.SIGINT, handler)

# =======================
# LOOP
# =======================
fps = 5
interval = 1.0 / fps
last_frame_time = time.time()
frame_id = 0

speed = 40

# RC rate limit
last_rc_time = 0
rc_interval = 0.05

while True:
    frame = frame_read.frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame is None:
        continue

    # =======================
    # DISPLAY
    # =======================
    cv2.imshow("Tello", frame)

    if USE_PYNPUT:
        cv2.pollKey()
        pressed = keys.copy()
    else:
        key = cv2.waitKey(1) & 0xFF
        pressed = set()
        if key != 255:
            pressed.add(chr(key))

    # =======================
    # SAVE
    # =======================
    now = time.time()
    if now - last_frame_time >= interval:
        cv2.imwrite(os.path.join(save_dir, f"{frame_id:06d}.png"), frame)
        frame_id += 1
        last_frame_time = now

    # =======================
    # CONTROL
    # =======================
    lr, fb, ud, yaw = 0, 0, 0, 0

    if 'w' in pressed: fb = speed
    if 's' in pressed: fb = -speed
    if 'a' in pressed: lr = -speed
    if 'd' in pressed: lr = speed
    if 'r' in pressed: ud = speed
    if 'f' in pressed: ud = -speed
    if 'q' in pressed: yaw = -speed
    if 'e' in pressed: yaw = speed

    # =======================
    # SEND RC
    # =======================
    if now - last_rc_time > rc_interval:
        tello.send_rc_control(lr, fb, ud, yaw)
        last_rc_time = now

    # =======================
    # LAND
    # =======================
    if 'l' in pressed or 'esc' in pressed:
        safe_land()
        break



# =======================
# CLEANUP
# =======================
tello.streamoff()
tello.end()
cv2.destroyAllWindows()
