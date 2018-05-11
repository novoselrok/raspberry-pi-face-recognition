import feedparser
import requests
from django import http
from django.conf import settings
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from sklearn.externals import joblib

from app import utils
from app.models import Todo, RSSFeed

import numpy as np

LJUBLJANA_WEATHER = "http://api.openweathermap.org/data/2.5/weather?q=Ljubljana&appid=52a006a41f97d8f748d4097a03e77456&units=metric"


@csrf_exempt
def auth(request):
    if request.method == 'POST':
        clf = joblib.load(settings.CLASSIFIER_PATH)
        hist = utils.get_hist(request.FILES['image'].read())

        # label = clf.predict([hist])
        label = clf.classes_[np.argmax(clf.predict_proba([hist]))]
        print(clf.predict_proba([hist]))
        user_pk = int(label[0])
        user = User.objects.get(pk=user_pk)
        # WEATHER
        resp = requests.get(LJUBLJANA_WEATHER)
        weather = resp.json()
        weather_info = "The temperature outside is {} degrees Celsius. I would describe the weather as {}.".format(
            weather['main']['temp'], weather['weather'][0]['description'])

        # TODOS
        todos = Todo.objects.filter(user=user, done=False)
        todos_text = "You have the following things to do: "
        todos_text += u" ".join([u"#{}: {}.".format(idx + 1, todo.name) for idx, todo in enumerate(todos)])

        # FEEDS
        feeds = RSSFeed.objects.filter(user=user, enabled=True)
        feeds_text = "Here are the latest news from your feeds: "
        for feed in feeds:
            parsed = feedparser.parse(feed.url)
            feeds_text += ". The latest titles on {} are: ".format(feed.name)
            feeds_text += u" ".join([u"#{}: {}.".format(idx + 1, e.title) for idx, e in enumerate(parsed.entries[:3])])

        # Get a joke
        joke_resp = requests.get("https://icanhazdadjoke.com/", headers={'Accept': 'application/json'}).json()

        return http.JsonResponse(
            {"person": user.first_name, "weather": weather_info, "feeds": feeds_text, "todos": todos_text,
             "joke": joke_resp["joke"]})

    return http.JsonResponse({"error": "Non-supported HTTP method."})


def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'app/signup.html', {'form': form})


def home(request):
    return HttpResponse()


def upload(request):
    if request.method == 'POST':
        files = request.FILES.getlist('img')
        images = [f.read() for f in files]

        utils.build_classifier(request.user, images)

        return HttpResponse('All ok.')
    else:
        return render(request, 'app/upload.html')
