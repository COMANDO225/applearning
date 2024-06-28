from django.urls import path
from .views import NumberListView, PredictCareerView

urlpatterns = [
    path('numbers/', NumberListView.as_view(), name='number-list'),
    path('predecir/', PredictCareerView.as_view(), name='predict-career'),
]
