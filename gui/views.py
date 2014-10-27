from collections import OrderedDict
import os
from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.forms.models import model_to_dict
from django.template.loader import render_to_string
from django.views.decorators.cache import never_cache

from gui.models import Experiment, Vectors, Table

excluded_cl_columns = ['id', 'date_ran', 'git_hash', 'vectors']  # todo id needed in both these?
excluded_vector_columns = ['id', 'can_build', 'path', 'unlabelled_percentage', 'modified', 'size']
columns_to_show = {}


def init_columns_to_show():
    # get all fields of the Experiment object
    data = model_to_dict(Experiment.objects.get(id=1), exclude=excluded_cl_columns)
    # field -> all values it has in the database
    for column_name in data.keys():
        columns_to_show[column_name] = set(Experiment.objects.values_list(column_name, flat=True))
    data = model_to_dict(Vectors.objects.get(id=1), exclude=excluded_vector_columns)
    for column_name in data.keys():
        columns_to_show['vectors__%s' % column_name] = set(Vectors.objects.values_list(column_name, flat=True))


def index(request):
    init_columns_to_show()
    return render_to_response('index.html', {'data': OrderedDict(sorted(columns_to_show.items()))})


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
    existing_experiments = [foo for foo in request.session.get('groups', [])]
    # representative_experiment_ids = [foo[0] for foo in existing_experiments]
    existing_experiments = Experiment.objects.all().filter(id__in=existing_experiments)
    if len(existing_experiments) < 1:
        return HttpResponse('No experiments match your current selection')
    desc = 'Settings for selected experiments:'
    if not columns_to_show:
        init_columns_to_show()
    header = ['id', 'vectors__id'] + list(columns_to_show.keys())
    print()
    rows = []
    for i, exp in enumerate(existing_experiments):
        row = []
        for field in header:
            if 'vectors__' in field:
                # need to follow foreign key
                if exp.vectors:
                    row.append(getattr(exp.vectors, field.split('__')[1]))
                else:
                    row.append(None)
            else:
                row.append(getattr(exp, field))
        rows.append(row)
    table = Table(header, rows, desc)
    print(rows)
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
    existing_experiments = [exp for exp in existing_experiments]
    existing_experiments.extend([x for x in new_experiments if x not in existing_experiments])

    all_selected = Experiment.objects.filter(id__in=existing_experiments)
    request.session['groups'] = [foo.id for foo in all_selected]

    return show_current_selection(request)


def clear_groups(request):
    request.session.flush()
    return HttpResponse()