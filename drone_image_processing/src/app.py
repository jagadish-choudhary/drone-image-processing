import cv2
import numpy as np
import threading
from inference_sdk import InferenceHTTPClient

# 1. Initialize the Roboflow Cloud Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="api_key_here"  
)

MODEL_ID = "drone-water-quality-monitor/1" 
latest_predictions = []

# Enterprise UI BGR Color Palette
UI_COLORS = {
    "Clean": (255, 191, 0),      
    "Turbid": (50, 127, 255),    
    "Algae": (0, 215, 0),        
    "Polluted": (0, 0, 255)      
}

def fetch_inference(frame_to_infer):
    global latest_predictions
    # Compress the payload for the cloud
    small_frame = cv2.resize(frame_to_infer, (800, 600))
    cv2.imwrite("temp_frame.jpg", small_frame)
    try:
        result = CLIENT.infer("temp_frame.jpg", model_id=MODEL_ID)
        # TARGET MEMORY: Only update the UI if the AI actually found something.
        if "predictions" in result and len(result["predictions"]) > 0:
            latest_predictions = result["predictions"]
    except Exception as e:
        pass 

cap = cv2.VideoCapture("drone_footage_3.mp4")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Standardize frame to HUD size (1024x768)
    frame = cv2.resize(frame, (1024, 768))

    # Fire background API call every 15 frames (approx 2 times a second)
    if frame_count % 15 == 0:
        threading.Thread(target=fetch_inference, args=(frame.copy(),), daemon=True).start()

    # ---------------------------------------------------------
    # ENTERPRISE HUD VISUALIZATION 
    # ---------------------------------------------------------
    for pred in latest_predictions:
        class_name = pred["class"]
        box_color = UI_COLORS.get(class_name, (0, 255, 255))
        
        # Map 800x600 API response to 1024x768 screen
        scale_x = 1024 / 800.0
        scale_y = 768 / 600.0
        
        if "points" in pred:
            pts = np.array([[int(p['x']), int(p['y'])] for p in pred["points"]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            pts[:, 0, 0] = (pts[:, 0, 0] * scale_x).astype(np.int32)
            pts[:, 0, 1] = (pts[:, 0, 1] * scale_y).astype(np.int32)

            cv2.polylines(frame, [pts], isClosed=True, color=box_color, thickness=2, lineType=cv2.LINE_AA)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], box_color, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # DRAW DATA PLATES
        x = int(pred["x"] * scale_x)
        y = int(pred["y"] * scale_y)
        w = int(pred["width"] * scale_x)
        h = int(pred["height"] * scale_y)
        tl = (int(x - w/2), int(y - h/2)) 
        
        text = f" TARGET: {class_name.upper()} "
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, 0.5, 1)
        
        pad = 6
        plate_start = (tl[0], tl[1] - th - (pad * 2))
        plate_end = (tl[0] + tw, tl[1])
        
        ui_overlay = frame.copy()
        cv2.rectangle(ui_overlay, plate_start, plate_end, (20, 20, 20), -1)
        cv2.addWeighted(ui_overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.line(frame, plate_start, (plate_end[0], plate_start[1]), box_color, 2, cv2.LINE_AA)
        cv2.putText(frame, text, (tl[0], tl[1] - pad), font, 0.5, box_color, 1, cv2.LINE_AA)
        
        # CENTER RETICLE (Image Moments)
        if "points" in pred:
            M = cv2.moments(pts)
            if M['m00'] != 0:
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            else:
                cx, cy = x, y 
        else:
            cx, cy = x, y
            
        cv2.circle(frame, (cx, cy), 4, box_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 12, box_color, 1, cv2.LINE_AA)

        # FORCE TERMINAL OUTPUT (Massive Data Wall)
        print(f"\n[TELEMETRY] Target: {class_name.upper()} | Center -> X:{cx}, Y:{cy}", flush=True)
        if "points" in pred:
            print(f"[RAW POLYGON MATH] {pred['points']}", flush=True)
    cv2.imshow("Live Telemetry: Simulated UAV Feed", frame)
    frame_count += 1

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

