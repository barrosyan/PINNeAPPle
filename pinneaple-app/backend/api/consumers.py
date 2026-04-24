"""
Django Channels WebSocket consumer for live training progress.
"""
import json
from channels.generic.websocket import AsyncWebsocketConsumer


class TrainingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.run_id    = self.scope["url_route"]["kwargs"]["run_id"]
        self.group_name = f"training_{self.run_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        # Send current status from registry (for late-connecting clients)
        from api.training_worker import get_status
        status = get_status(self.run_id)
        if status:
            await self.send(text_data=json.dumps({
                "type":    "init",
                "status":  status["status"],
                "history": status["history"],
                "result":  status["result"],
            }))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            data = json.loads(text_data)
            if data.get("type") == "stop":
                from api.training_worker import stop_training
                stop_training(self.run_id)

    async def training_update(self, event):
        await self.send(text_data=json.dumps(event["payload"]))
