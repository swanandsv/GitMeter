from django.db import models

# Create your models here.


class Resume(models.Model):
    file = models.FileField(upload_to='resumes/')