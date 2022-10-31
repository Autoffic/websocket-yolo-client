from typing import Any, List
import socketio

socketio_client = socketio.Client(logger=True, engineio_logger=True)
yolo_event = 'YOLO_EVENT'
default_url = 'http://localhost:5000'
PC_ID = "EAST"

# starting the connection, should be called only once
def connect(url: str = default_url) -> None:
    socketio_client.connect(url)

# terminating the connection, show be called only once and only after the socket has been connected
def disconnect() -> None:
    socketio_client.disconnect()

# emitting the message, the message will be serialized
def emit(message: List) -> None:
    '''
    The message should be in the form of a list
    '''

    # appending the pc id to the front of the list
    message.insert(0, PC_ID)
    serialized_message = str(message)

    # continue even if the message cannot be emitted
    try:
        socketio_client.emit(yolo_event, serialized_message)
    except socketio.server.exceptions.SocketIOError as err:
        print(err)
