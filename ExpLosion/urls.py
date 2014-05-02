from django.conf.urls import patterns, url

from django.contrib import admin
from gui import views
from gui.user_code import BaseExplosionAnalysis


analyser = BaseExplosionAnalysis()
analyser.populate_experiments_db()

admin.autodiscover()
urlpatterns = patterns('',  # do not remove the first parameter
                       url(r'^analyse$', views.analyse, kwargs={'analyser': analyser}, name='analyse'),
                       url(r'^add_group$', views.add_group, name='add_group'),
                       url(r'^index$', views.index, name='index'),
                       url(r'^$', views.index, name='index'),
)
