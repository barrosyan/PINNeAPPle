"""
Authentication views — register, me, logout, password change.
JWT login and refresh are handled by simplejwt built-in views.
"""
from django.contrib.auth import get_user_model
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import TokenError

User = get_user_model()


def _user_payload(user) -> dict:
    return {
        "id":          user.id,
        "username":    user.username,
        "email":       user.email,
        "first_name":  user.first_name,
        "last_name":   user.last_name,
        "date_joined": user.date_joined.isoformat(),
    }


def _tokens_for(user) -> tuple[str, str]:
    refresh = RefreshToken.for_user(user)
    return str(refresh.access_token), str(refresh)


class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        username   = request.data.get("username", "").strip()
        email      = request.data.get("email", "").strip().lower()
        password   = request.data.get("password", "")
        first_name = request.data.get("first_name", "").strip()
        last_name  = request.data.get("last_name", "").strip()

        if not username:
            return Response({"error": "Username is required"}, status=400)
        if len(username) < 3:
            return Response({"error": "Username must be at least 3 characters"}, status=400)
        if len(password) < 8:
            return Response({"error": "Password must be at least 8 characters"}, status=400)
        if User.objects.filter(username__iexact=username).exists():
            return Response({"error": "Username already taken"}, status=400)
        if email and User.objects.filter(email__iexact=email).exists():
            return Response({"error": "Email already in use"}, status=400)

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
        )
        access, refresh = _tokens_for(user)
        return Response(
            {"user": _user_payload(user), "access": access, "refresh": refresh},
            status=201,
        )


class MeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(_user_payload(request.user))

    def patch(self, request):
        user = request.user
        user.first_name = request.data.get("first_name", user.first_name).strip()
        user.last_name  = request.data.get("last_name",  user.last_name).strip()
        new_email = request.data.get("email", "").strip().lower()
        if new_email and new_email != user.email:
            if User.objects.filter(email__iexact=new_email).exclude(pk=user.pk).exists():
                return Response({"error": "Email already in use"}, status=400)
            user.email = new_email
        user.save()
        return Response(_user_payload(user))


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        refresh_token = request.data.get("refresh")
        if not refresh_token:
            return Response({"error": "refresh token required"}, status=400)
        try:
            RefreshToken(refresh_token).blacklist()
        except TokenError:
            pass  # already blacklisted or expired — treat as success
        return Response({"detail": "Logged out successfully"})


class ChangePasswordView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        current  = request.data.get("current_password", "")
        new_pass = request.data.get("new_password", "")

        if not request.user.check_password(current):
            return Response({"error": "Current password is incorrect"}, status=400)
        if len(new_pass) < 8:
            return Response({"error": "New password must be at least 8 characters"}, status=400)
        if current == new_pass:
            return Response({"error": "New password must differ from current"}, status=400)

        request.user.set_password(new_pass)
        request.user.save()
        # Re-issue tokens so the client stays logged in
        access, refresh = _tokens_for(request.user)
        return Response({"access": access, "refresh": refresh, "detail": "Password changed"})
