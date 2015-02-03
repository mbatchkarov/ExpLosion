# coding: utf-8
import os
import django
# the name of the python file that has all settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'ExpLosion.settings'
django.setup() # new in django 1.7

# some code that uses the django ORM
# this import needs to happen AFTER django has been configured
# from gui.user_code import *
# exp_ids = [37, 43, 49]
# get_demsar_diagram(*get_demsar_params(exp_ids))
