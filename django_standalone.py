# coding: utf-8
import os
# the name of the python file that has all settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'ExpLosion.settings'

# this import needs to happen AFTER django has been configured
from gui.user_code import *

# some code that uses the django ORM
exp_ids = [137, 143, 149]
sign_table, names, scores = get_demsar_params(exp_ids)
get_demsar_diagram(sign_table, names, scores)
