from rest_framework.views import APIView
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from api.problem_defs import (
    PROBLEMS, CATEGORIES, get_problem, list_problems,
    formulate_with_ai, suggest_models,
)


class ProblemListView(APIView):
    """Public — anyone can browse the problem library."""
    permission_classes = [AllowAny]

    def get(self, request):
        category = request.query_params.get("category")
        items    = list_problems(category)
        return Response([{"key": k, **v} for k, v in items])


class ProblemDetailView(APIView):
    """Public."""
    permission_classes = [AllowAny]

    def get(self, request, key):
        prob = get_problem(key)
        if prob:
            return Response({"key": key, **prob})
        return Response({"error": "not found"}, status=404)


class ProblemCategoriesView(APIView):
    """Public."""
    permission_classes = [AllowAny]

    def get(self, request):
        return Response(CATEGORIES)


class FormulateView(APIView):
    """Private — requires auth (uses AI quota)."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        description = request.data.get("description", "").strip()
        if not description:
            return Response({"error": "description required"}, status=400)
        spec = formulate_with_ai(description)
        return Response(spec)


class SuggestModelsView(APIView):
    """Private."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        problem = request.data
        return Response(suggest_models(problem))
