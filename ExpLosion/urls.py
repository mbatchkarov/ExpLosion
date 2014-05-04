from django.conf.urls import patterns, url

from django.contrib import admin
from gui import views
from gui.user_code import BaseExplosionAnalysis


analyser = BaseExplosionAnalysis()
analyser.populate_experiments_db()

admin.autodiscover()
urlpatterns = patterns('',  # do not remove the first parameter
                       url(r'^analyse$', views.analyse, kwargs={'analyser': analyser}),
                       url(r'^add_group$', views.add_group),
                       url(r'^clear_groups$', views.clear_groups),
                       url(r'^toggle_duplicates', views.show_current_selection, kwargs={'allow_pruning': True}),
                       url(r'^index$', views.index),
                       url(r'^$', views.index),
)
