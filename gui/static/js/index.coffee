add_group = ->
    checked_boxes = new Object()
    for paramdiv in document.getElementById('form').children
        checked_boxes[paramdiv.id] = []
        for boxdiv in paramdiv.children
            if boxdiv.getElementsByTagName('input')[0].checked
                checked_boxes[paramdiv.id].push(boxdiv.getElementsByTagName('input')[0].name)

    suffix = jQuery.param checked_boxes
    suffix = "?".concat(suffix)
    myurl = "/add_group".concat(suffix)
    $.ajax(
        url: myurl
        success: (data) ->
            d = $('#groups-div')
            d[0].className = "visible_div"
            d.html data
            $(".table").tablesorter()
            return
    )
    return

clear_groups = ->
    $.ajax(
        url: '/clear_groups'
        success: (data) ->
            $('#groups-div').html ""
            $('#results-div').html ""
            $('#groups-div')[0].className = "invisible_div"
            $(".table").tablesorter()
            return
    )

toggle_duplicates = ->
    $.ajax(
        url: '/toggle_duplicates'
        success: (data) ->
            $('#groups-div').html data
            $('#groups-div')[0].className = "visible_div"
            $(".table").tablesorter()
    )

analyze_selected_experiments = ->
    $.ajax
        url: "/analyse"
        success: (data) ->
            $("#results-div").html data
            $(".table").tablesorter()
            return
    return

$(document).ready(bla = ->
    $('input:checkbox').bootstrapSwitch('size', 'mini')
)
