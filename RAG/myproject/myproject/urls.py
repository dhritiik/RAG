from django.urls import path
from chat import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process/', views.process_pdfs, name='process_pdfs'),
    path('ask/', views.ask_question, name='ask_question'),  # Define the path and name

]
