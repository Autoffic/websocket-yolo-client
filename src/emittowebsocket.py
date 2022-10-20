from typing import Any
import socketio
import json

socketio_client = socketio.Client()
yolo_event = 'YOLO_EVENT'
default_url = 'http://127.0.0.1:5000'

# starting the connection, should be called only once
def connect(url: str = default_url) -> None:
    socketio_client.connect(url)

# terminating the connection, show be called only once and only after the socket has been connected
def disconnect() -> None:
    socketio_client.disconnect()

# emitting the message, the message will be serialized
def emit(message: Any) -> None:
    serialized_message = str(message)

    socketio_client.emit(yolo_event, serialized_message)