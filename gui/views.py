from collections import OrderedDict
import os
from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.template.loader import render_to_string
from django.views.decorators.cache import never_cache
from pandas import DataFrame
from gui.models import Experiment
from gui.user_code import get_tables, get_generated_figures, get_static_figures
from notebooks.common_imports import settings_of


EXCLUDED_COLUMNS = ['expansions__vectors_id']


def convert_type(col_name:str, col_value:list):
    col_type = column_types.get(col_name, None)
    if col_type is None:
        return col_value
    return [None if x == 'None' else col_type(x) for x in col_value]


def init_columns_to_show():
    # get all fields of the Experiment object
    data = {}
    for e in Experiment.objects.all():
        data.update(settings_of(e.id, exclude=EXCLUDED_COLUMNS))

    # field -> all values it has in the database
    names, types = dict(), dict()
    for column_name in data.keys():
        col_values = set(Experiment.objects.values_list(column_name, flat=True))
        names[column_name] = col_values
        col_type = set(type(x) for x in col_values if x is not None)
        types[column_name] = list(col_type)[0]
    return names, types


columns_to_show, column_types = init_columns_to_show()


def index(request):
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
    existing_experiments = Experiment.objects.filter(id__in=existing_experiments)
    if len(existing_experiments) < 1:
        return HttpResponse('No experiments match your current selection')
    if not columns_to_show:
        init_columns_to_show()
    header = ['id', 'expansions__vectors__id'] + list(columns_to_show.keys())
    rows = []
    for i, exp in enumerate(existing_experiments):
        row = Experiment.objects.filter(id=exp.id).values_list(*header)[0]
        rows.append(row)
    table = DataFrame(rows, columns=header).set_index(['id'])[sorted(header[1:])]
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
    query_dict = {'%s__in' % k: convert_type(k, v) for k, v in params.items()}
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

if __name__ == '__main__':
    import django_standalone
    init_columns_to_show()