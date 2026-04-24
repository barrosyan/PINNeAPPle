import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pinneaple_backend.settings")

django_asgi_app = get_asgi_application()

from api.routing import websocket_urlpatterns  # noqa: E402 (after django setup)

application = ProtocolTypeRouter({
    "http":      django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
})
