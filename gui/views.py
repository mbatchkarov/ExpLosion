from collections import OrderedDict
import os
from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.forms.models import model_to_dict
from django.template.loader import render_to_string
from django.views.decorators.cache import never_cache
from pandas import DataFrame

from gui.models import Experiment, Vectors
from gui.user_code import get_tables, get_generated_figures, get_static_figures

excluded_cl_columns = ['id', 'date_ran', 'git_hash', 'vectors']  # todo id needed in both these?
excluded_vector_columns = ['id', 'can_build', 'path', 'modified', 'size']
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


def analyse(request):
    response = HttpResponse()

    exp_ids = request.session.get('groups', [])
    for table in get_tables(exp_ids):
        response.write(to_html(table))

    for img in get_generated_figures(exp_ids):
        content = render_to_string('image.html', {'image': img})
        response.write(content)

    for path in get_static_figures(exp_ids):
        response.write('<br> %s <br> <img src="%s"><br>' % (os.path.basename(path), path))

    return response


@never_cache
def show_current_selection(request, allow_pruning=True):
    # render all currently requested experiments
    existing_experiments = [foo for foo in request.session.get('groups', [])]
    # representative_experiment_ids = [foo[0] for foo in existing_experiments]
    existing_experiments = Experiment.objects.all().filter(id__in=existing_experiments)
    if len(existing_experiments) < 1:
        return HttpResponse('No experiments match your current selection')
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
    table = DataFrame(rows, columns=header)
    print(rows)
    if allow_pruning:
        prune = not request.session.get('prune_duplicates', False)  # initially false
        request.session['prune_duplicates'] = prune
        if prune:
            table = prune_table(table)
    response = HttpResponse()
    response.write(to_html(table))
    return response


def add_group(request):
    # store the newly requested experiments in the session
    params = {k[:-2]: request.GET.getlist(k) for k in request.GET.keys()}
    query_dict = {'%s__in' % k: v for k, v in params.items()}
    print(query_dict)
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


def prune_table(df):
    """
    Removes columns where all values are duplicates
    """
    dupl_names = []
    for column_name in df.columns:
        if len(set(df[column_name])) == 1:
            dupl_names.append(column_name)

    if dupl_names and len(df) > 1:
        df = df.drop(dupl_names, axis=1)
    return df

def to_html(table):
    return table.to_html(classes="table table-nonfluid table-hover table-bordered table-condensed tablesorter")
