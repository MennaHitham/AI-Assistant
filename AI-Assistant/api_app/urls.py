from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health_check, name='health_check'),
    path('initialize/', views.initialize_pipeline, name='initialize'),
    path('chat/', views.chat, name='chat'),
    path('youtube/process/', views.process_youtube, name='process_youtube'),
    path('recommendations/', views.get_recommendations, name='recommendations'),
    path('presentation/create/', views.create_presentation, name='create_presentation'),
    path('presentation/download/<str:filename>/', views.download_presentation, name='download_presentation'),
    path('documents/upload/', views.upload_document, name='upload_document'),
    path('documents/ask/', views.upload_and_ask, name='upload_and_ask'),
    path('images/upload/', views.upload_image, name='upload_image'),
]
