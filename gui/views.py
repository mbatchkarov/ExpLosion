from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.forms.models import model_to_dict
from django.template.loader import render_to_string

from gui.models import Experiment


def index(request):
    # get all fields of the Experiment object
    data = model_to_dict(Experiment.objects.get(number=1), exclude=['number'])
    valid_fields = {}
    # field -> all values it has in the database
    for column_name in data.keys():
        valid_fields[column_name] = set(Experiment.objects.values_list(column_name, flat=True))
    return render_to_response('index.html', {'data': valid_fields})


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
