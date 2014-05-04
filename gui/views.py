from collections import OrderedDict
from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.forms.models import model_to_dict
from django.template.loader import render_to_string

from gui.models import Experiment
from gui.user_code import Table


def index(request):
    # get all fields of the Experiment object
    data = model_to_dict(Experiment.objects.get(number=1), exclude=['number'])
    valid_fields = {}
    # field -> all values it has in the database
    for column_name in data.keys():
        valid_fields[column_name] = sorted(set(Experiment.objects.values_list(column_name, flat=True)))
    return render_to_response('index.html', {'data': OrderedDict(sorted(valid_fields.items()))})


def analyse(request, analyser=None):
    params = dict(request.GET.items())
    params['analyser'] = analyser
    response = HttpResponse()

    for table in analyser.get_tables():
        content = render_to_string('table.html', table.__dict__)
        response.write(content)

    for img in analyser.get_figures():
        content = render_to_string('image.html', {'image': img})
        response.write(content)

    return response


def add_group(request):
    # store the newly requested experiments in the session
    params = {k[:-2]: request.GET.getlist(k) for k in request.GET.keys()}
    query_dict = {'%s__in' % k: v for k, v in params.items()}
    new_experiments = Experiment.objects.values_list('number', flat=True).filter(**query_dict)
    existing_experiments = request.session.get('groups', [])
    existing_experiments.extend([x for x in new_experiments if x not in existing_experiments])
    request.session['groups'] = existing_experiments

    # render all currently requested experiments
    experiments = Experiment.objects.all().filter(number__in=existing_experiments)
    if len(experiments) < 1:
        return HttpResponse('No experiments match your current selection')

    data = {}
    desc = 'Settings for selected experiments:'
    header = sorted(x for x in experiments[0].__dict__ if not x.startswith('_'))
    rows = []
    for exp in experiments:
        rows.append([getattr(exp, field) for field in header])

    table = Table(header, rows, desc)
    return render_to_response('table.html', table.__dict__)


def clear_groups(request):
    print 'clearing groups'
    request.session['groups'] = []
    return HttpResponse()