from django.urls import path
from .views import PredictCareerView

urlpatterns = [
    path('predecir/', PredictCareerView.as_view(), name='predict-career'),
]
