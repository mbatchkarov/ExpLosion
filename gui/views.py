from collections import OrderedDict
from itertools import groupby
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
        if column_name == 'labelled':
            all_values = Experiment.objects.values_list(column_name, flat=True)
            acceptable_values = set(x if len(x) < 5 else 'TechTC' for x in all_values)  # what a hack!
            valid_fields[column_name] = acceptable_values
        else:
            valid_fields[column_name] = sorted(set(Experiment.objects.values_list(column_name, flat=True)))
    print(valid_fields)
    return render_to_response('index.html', {'data': OrderedDict(sorted(valid_fields.items()))})


def demo(request):
    return render_to_response('demo.html')


def analyse(request, get_tables=lambda foo: [], get_generated_figures=lambda foo: [],
            get_static_figures=lambda foo: []):
    response = HttpResponse()

    exp_ids = request.session.get('groups', [])
    for table in get_tables(exp_ids):
        content = render_to_string('table.html', table.__dict__)
        response.write(content)

    for img in get_generated_figures(exp_ids):
        content = render_to_string('image.html', {'image': img})
        response.write(content)

    for path in get_static_figures(exp_ids):
        response.write('<br> %s <br> <img src="%s"><br>' % (os.path.basename(path), path))

    return response


@never_cache
def show_current_selection(request, allow_pruning=False):
    # render all currently requested experiments
    existing_experiment_groups = [foo for foo in request.session.get('groups', [])]
    representative_experiment_ids = [foo[0] for foo in existing_experiment_groups]
    representative_experiments = Experiment.objects.all().filter(id__in=representative_experiment_ids)
    if len(representative_experiments) < 1:
        return HttpResponse('No experiments match your current selection')
    desc = 'Settings for selected experiments:'
    header = sorted(x for x in representative_experiments[0].__dict__ if not x.startswith('_'))
    rows = []
    for i, exp in enumerate(representative_experiments):
        rows.append(['%d(+%d)' % (exp.id, len(existing_experiment_groups[i]) - 1) if field == 'id' \
                         else getattr(exp, field) for field in header])
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
    if query_dict['labelled__in'] == ['TechTC']:
        del query_dict['labelled__in']
        query_dict['labelled__contains'] = 'techtc'
    new_experiments = Experiment.objects.values_list('id', flat=True).filter(**query_dict)
    existing_experiments = request.session.get('groups', [])
    existing_experiments = [exp for group in existing_experiments for exp in group]
    existing_experiments.extend([x for x in new_experiments if x not in existing_experiments])

    all_selected = Experiment.objects.filter(id__in=existing_experiments)

    def get_coarse_attr(obj, attr):
        val = getattr(obj, attr)
        if attr == 'labelled' and 'techtc' in val.lower():
            val = 'TechTC'
        return val

    def get_coarse_experiment_settings(x, excluded=['id', '_state']):
        fields = set(foo for foo in x.__dict__.keys())
        for foo in excluded:
            fields.remove(foo)
        return tuple([(f, get_coarse_attr(x, f)) for f in sorted(fields)])

    grouped_experiments = []
    s = sorted(all_selected, key=get_coarse_experiment_settings)
    for key, group in groupby(s, get_coarse_experiment_settings):
        grouped_experiments.append([foo.id for foo in group])

    request.session['groups'] = grouped_experiments

    return show_current_selection(request)


def clear_groups(request):
    request.session.flush()
    return HttpResponse()