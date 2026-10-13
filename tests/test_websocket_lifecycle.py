"""CPU work can finish after the transport has closed; return quietly."""
import unittest
from unittest.mock import AsyncMock
from starlette.websockets import WebSocketDisconnect, WebSocketState
from websocket_lifecycle import close_if_open, send_json_if_open


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    def socket(self):
        socket = AsyncMock()
        socket.client_state = WebSocketState.CONNECTED
        socket.application_state = WebSocketState.CONNECTED
        return socket

    async def test_transport_close_before_starlette_disconnect_event(self):
        socket = self.socket()
        socket.send_json.side_effect = RuntimeError(
            "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'.")
        self.assertFalse(await send_json_if_open(socket, {'type':'partial'}))
        socket.send_json.assert_awaited_once()

    async def test_peer_disconnection_during_final_send(self):
        for error in (WebSocketDisconnect(1006), BrokenPipeError(), ConnectionResetError()):
            socket = self.socket()
            socket.send_json.side_effect = error
            self.assertFalse(await send_json_if_open(socket, {'type':'final'}))

    async def test_closed_application_state(self):
        socket = self.socket()
        socket.application_state = WebSocketState.DISCONNECTED
        socket.send_json.side_effect = RuntimeError('Cannot call "send" once a close message has been sent.')
        self.assertFalse(await send_json_if_open(socket, {}))

    async def test_peer_closes_between_final_send_and_server_close(self):
        socket = self.socket()
        socket.close.side_effect = RuntimeError(
            "Unexpected ASGI message 'websocket.close', after sending 'websocket.close'.")
        await close_if_open(socket)

    async def test_unrelated_errors_are_not_hidden(self):
        socket = self.socket()
        socket.send_json.side_effect = RuntimeError('Invalid inference result')
        with self.assertRaisesRegex(RuntimeError, 'Invalid inference result'):
            await send_json_if_open(socket, {})
        socket.close.side_effect = RuntimeError('Protocol implementation bug')
        with self.assertRaisesRegex(RuntimeError, 'Protocol implementation bug'):
            await close_if_open(socket)

    async def test_connected_send_preserves_message(self):
        socket = self.socket()
        message = {'type':'partial', 'text':'ਵਾਹਿਗੁਰੂ'}
        self.assertTrue(await send_json_if_open(socket, message))
        socket.send_json.assert_awaited_once_with(message)
