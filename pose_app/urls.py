from django.urls import path
from .views import ComparePoseView, FeedbackView, CurrentPoseNameView

urlpatterns = [
    path("compare-pose/", ComparePoseView.as_view()),
    path("feedback/", FeedbackView.as_view()),
    path("current-pose/", CurrentPoseNameView.as_view()),
]
