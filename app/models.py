from django.db import models
from django.contrib.auth.models import User


class Todo(models.Model):
    name = models.CharField(max_length=30)
    description = models.TextField(max_length=255)
    done = models.BooleanField(default=False)
    user = models.ForeignKey(User)


class RSSFeed(models.Model):
    name = models.TextField()
    url = models.URLField()
    enabled = models.BooleanField(default=True)
    user = models.ForeignKey(User)
