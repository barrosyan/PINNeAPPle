from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from api.views.auth      import RegisterView, MeView, LogoutView, ChangePasswordView
from api.views.projects  import ProjectViewSet
from api.views.problems  import (ProblemListView, ProblemDetailView,
                                  ProblemCategoriesView, FormulateView,
                                  SuggestModelsView)
from api.views.training  import (StartTrainingView, TrainingRunDetailView,
                                  TrainingRunListView, StopTrainingView,
                                  TrainingStatusView)
from api.views.inference  import InferenceView
from api.views.benchmarks import (BenchmarkListView, BenchmarkRunView,
                                   BenchmarkResultsView)
from api.views.files      import (FileUploadView, FileListView,
                                   FileDetailView, FileDataView)

router = DefaultRouter()
router.register(r"projects", ProjectViewSet, basename="project")

urlpatterns = [
    path("", include(router.urls)),

    # ── Auth ──────────────────────────────────────────────────────────────────
    path("auth/register/",         RegisterView.as_view()),
    path("auth/login/",            TokenObtainPairView.as_view()),
    path("auth/token/refresh/",    TokenRefreshView.as_view()),
    path("auth/logout/",           LogoutView.as_view()),
    path("auth/me/",               MeView.as_view()),
    path("auth/change-password/",  ChangePasswordView.as_view()),

    # ── Problems (partially public) ───────────────────────────────────────────
    path("problems/",                   ProblemListView.as_view()),
    path("problems/categories/",        ProblemCategoriesView.as_view()),
    path("problems/formulate/",         FormulateView.as_view()),
    path("problems/suggest/",           SuggestModelsView.as_view()),
    path("problems/<str:key>/",         ProblemDetailView.as_view()),

    # ── Training ──────────────────────────────────────────────────────────────
    path("projects/<uuid:project_id>/train/",       StartTrainingView.as_view()),
    path("training/",                               TrainingRunListView.as_view()),
    path("training/<uuid:run_id>/",                 TrainingRunDetailView.as_view()),
    path("training/ws/<str:ws_run_id>/stop/",       StopTrainingView.as_view()),
    path("training/ws/<str:ws_run_id>/status/",     TrainingStatusView.as_view()),

    # ── Inference ─────────────────────────────────────────────────────────────
    path("inference/<uuid:run_id>/",    InferenceView.as_view()),

    # ── Benchmarks ────────────────────────────────────────────────────────────
    path("benchmarks/",                 BenchmarkListView.as_view()),
    path("benchmarks/run/",             BenchmarkRunView.as_view()),
    path("benchmarks/results/",         BenchmarkResultsView.as_view()),

    # ── Files ─────────────────────────────────────────────────────────────────
    path("files/",                      FileListView.as_view()),
    path("files/upload/",               FileUploadView.as_view()),
    path("files/<uuid:file_id>/",       FileDetailView.as_view()),
    path("files/<uuid:file_id>/data/",  FileDataView.as_view()),
]
