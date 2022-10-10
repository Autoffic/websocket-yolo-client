# websocket-yolo-client
Vehicle detection using yolov5 and transmitting data through socketio-client
<br>

url: http://localhost:5000
<br>

Emits "YOLO_EVENT" and sends the array: `[bounding box coordinates, confidence, class, {detection type}]` which will be in the form of string.<br>
Here, detection type is either "Inference" or "Tracking"

# Example usage:
## Starting the detections with websocket client enabled
`python detectvehicles.py --source "~/Downloads/traffic.mp4" --web-socket`
<br>

## For listening the event ( in server or client)
```
@socketio.on("YOLO_EVENT")
def handle_yolo_event(received_data):
  print(received_data)
```
