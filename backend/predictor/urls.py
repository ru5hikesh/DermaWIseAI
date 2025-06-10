from django.urls import path
from .views import PredictionView, BedrockChatView

urlpatterns = [
    path('predict/', PredictionView.as_view(), name='predict'),
    path('bedrock/chat/', BedrockChatView.as_view(), name='bedrock-chat'),
]
