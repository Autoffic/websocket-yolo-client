from pathlib import Path
import sys

debug=True

# ssh url
url = "http://localhost:5000"

# to make source folder available

module_path = Path(__file__).resolve()
project_path = module_path.parents[1]

if str(project_path) not in sys.path:
    sys.path.append(str(project_path))  # adding project folder to path

from src.emittowebsocket import *

# starting websocket on default port 
connect(url)

# listening for yolo events
@socketio_client.on(yolo_event)
def get_next_light_configuration(received_data):

    if debug:
        print(f"\n Event: {yolo_event} \n \
                   Received_Data: {received_data}")
