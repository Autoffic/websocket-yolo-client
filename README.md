# websocket-yolo-client

url: http://localhost:5000

Emits "YOLO_EVENT" and sends the [bounding box coordinates, confidence, class, {detection type}] which will be in the form of string.<br>
Here, detection type is either "Inference" or "Tracking"

# Example usage:
## Starting the detections 
python detectvehicles.py --source "~/Downloads/traffic.mp4" --web-socket

## For listening the event ( in server or client)

@socketio.on("YOLO_EVENT")
def handel_yolo_event(received_data):
  print(received_data)
