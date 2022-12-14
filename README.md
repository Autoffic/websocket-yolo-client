# Vehicle detection using yolov5 and transmitting data through socketio-client  

Emits "YOLO_EVENT" and sends the array: `[PC_ID, passed lane, vehicle class]`
which will be in the form of string.  

The number of Region Of Interests and number of lanes can be set as an argument.
Websocket is disabled if `--web-socket` argument isn't passed.  

To save ROI and/or lanes co-ordinates for a paticular video in csv for using it in the next run, there are scripts in tools directory. The csvs generated will be dependent on the name of the video or the url(if url is passed). 
This won't work for webcam.  

## Example usages:
- ### Starting the detections with websocket client enabled
  `python detectvehicles.py --source "~/Downloads/traffic.mp4" --web-socket`
- ### Passing the number of lanes
  `python detectvehicles.py --source "~/Downloads/traffic.mp4" --web-socket --number-of-lanes 3`
- ### Passing the number of ROIs
  `python detectvehicles.py --source "~/Downloads/traffic.mp4" --web-socket --number-of-lanes 3 --number-of-rois 2`
- ### Saving ROIs and lanes coordinates to csv
  `python tools/generateroicsv.py --generate-lanes-csv --source ~/Downloads/traffic.mp4 --number-of-rois 1 --number-of-lanes 3`

## For listening the event ( in server or client )
```
@socketio.on("YOLO_EVENT")
def handle_yolo_event(received_data):
  print(received_data)
```
