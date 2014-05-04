process_form_element = (output_hash, element) ->
    name = element.name
    value = element.value
    type = element.type
    checked = element.checked
    tagName = element.tagName

    if tagName is "TEXTAREA"
        output_hash[name] = value
    else if tagName is "INPUT"
        switch type
            when "text", "hidden", "password" then output_hash[name] = value
            when "radio", "checkbox"
                if checked
                    if value
                        output_hash[name] = value
                    else
                        output_hash[name] = "on"

    return output_hash

parse_form_params_to_hash = (form) ->
    # autotranslated (and adapted) to coffeescript from
    # http://stackoverflow.com/questions/316781/how-to-build-query-string-with-javascript
    params = new Object()
    res = process_form_element(params, elem) for elem in form.elements
    return res

add_group = ->
    checked_boxes = new Object()
    for paramdiv in document.getElementById('form').children
        checked_boxes[paramdiv.id] = []
        for box in paramdiv.children
            if box.checked
                checked_boxes[paramdiv.id].push(box.name)
    suffix = jQuery.param checked_boxes

    suffix = "?".concat(suffix)
    myurl = "/add_group".concat(suffix)
    $.ajax(
        url: myurl
        success: (data) ->
            d = $('#groups-div')
            d[0].className = "visible_div"
            d.html data
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
            return
    )

toggle_duplicates = ->
    $.ajax(
        url: '/toggle_duplicates'
        success: (data) ->
            $('#groups-div').html data
            $('#groups-div')[0].className = "visible_div"
    )

analyze_selected_experiments = ->
    params = parse_form_params_to_hash($("#form")[0]) # make a query dict out the the form values
    query_string = "?".concat(jQuery.param(params)) # turn dict into query string

    $.ajax
        url: "/analyse".concat(query_string)
        success: (data) ->
            $("#results-div").html data
            return
    return
