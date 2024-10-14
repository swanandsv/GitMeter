from . import views
from django.urls import path 

urlpatterns = [
    path("", views.home, name="home"),
    path("no_github/", views.no_github, name="no_github"),
    path("results/<str:username>/", views.results, name="results"),
    path('check_code_quality_status/', views.check_code_quality_status, name='check_code_quality_status'),
    path('code_quality_results/', views.code_quality_results, name='code_quality_results'),
    path('non_code_quality_info/<str:username>/', views.non_code_quality_info, name='non_code_quality_info'),
    path('check_non_code_quality_status/', views.check_non_code_quality_status, name='check_non_code_quality_status'),  # Added line
]