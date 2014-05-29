from collections import OrderedDict
import os
from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.forms.models import model_to_dict
from django.template.loader import render_to_string
from django.views.decorators.cache import never_cache

from gui.models import Experiment, Table


def index(request):
    # get all fields of the Experiment object
    data = model_to_dict(Experiment.objects.get(id=1), exclude=['id'])
    valid_fields = {}
    # field -> all values it has in the database
    for column_name in data.keys():
        valid_fields[column_name] = sorted(set(Experiment.objects.values_list(column_name, flat=True)))
    return render_to_response('index.html', {'data': OrderedDict(sorted(valid_fields.items()))})


def demo(request):
    return render_to_response('demo.html')


def analyse(request, analyser=None):
    response = HttpResponse()

    exp_ids = request.session.get('groups', [])
    for table in analyser.get_tables(exp_ids):
        content = render_to_string('table.html', table.__dict__)
        response.write(content)

    for img in analyser.get_generated_figures(exp_ids):
        content = render_to_string('image.html', {'image': img})
        response.write(content)

    for path in analyser.get_static_figures(exp_ids):
        response.write('<br> %s <br> <img src="%s"><br>' % (os.path.basename(path), path))

    return response


@never_cache
def show_current_selection(request, allow_pruning=False):
    # render all currently requested experiments
    existing_experiments = request.session.get('groups', [])
    experiments = Experiment.objects.all().filter(id__in=existing_experiments)
    if len(experiments) < 1:
        return HttpResponse('No experiments match your current selection')
    desc = 'Settings for selected experiments:'
    header = sorted(x for x in experiments[0].__dict__ if not x.startswith('_'))
    rows = []
    for exp in experiments:
        rows.append([getattr(exp, field) for field in header])
    table = Table(header, rows, desc)

    if allow_pruning:
        prune = not request.session.get('prune_duplicates', False)  # initially false
        request.session['prune_duplicates'] = prune
        if prune:
            table.prune()
    return render_to_response('table.html', table.__dict__)


def add_group(request):
    # store the newly requested experiments in the session
    params = {k[:-2]: request.GET.getlist(k) for k in request.GET.keys()}
    query_dict = {'%s__in' % k: v for k, v in params.items()}
    new_experiments = Experiment.objects.values_list('id', flat=True).filter(**query_dict)
    existing_experiments = request.session.get('groups', [])
    existing_experiments.extend([x for x in new_experiments if x not in existing_experiments])
    request.session['groups'] = existing_experiments

    return show_current_selection(request)


def clear_groups(request):
    request.session.flush()
    return HttpResponse()