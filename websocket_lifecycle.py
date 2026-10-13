"""Treat transport closure during an awaited inference as a disconnect."""
from starlette.websockets import WebSocketDisconnect, WebSocketState


def _closed_transport(websocket, error):
    if (websocket.client_state == WebSocketState.DISCONNECTED
            or websocket.application_state == WebSocketState.DISCONNECTED):
        return True
    # Uvicorn's SansIO transport can close before Starlette sees a disconnect
    # event. Its state then remains CONNECTED until the next receive call.
    return str(error) in {
        "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'.",
        "Unexpected ASGI message 'websocket.close', after sending 'websocket.close'.",
        'Cannot call "send" once a close message has been sent.',
    }


async def send_json_if_open(websocket, message):
    try:
        await websocket.send_json(message)
    except (WebSocketDisconnect, OSError):
        return False
    except RuntimeError as error:
        if not _closed_transport(websocket, error):
            raise
        return False
    return True


async def close_if_open(websocket):
    try:
        await websocket.close()
    except (WebSocketDisconnect, OSError):
        pass
    except RuntimeError as error:
        if not _closed_transport(websocket, error):
            raise
