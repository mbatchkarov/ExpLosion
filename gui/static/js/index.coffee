reload = (my_suffix) ->
    $("#results-div").html "" #give user feedback by flashing page
    $.ajax
        url: "/analyse".concat(my_suffix)
        success: (data) ->
            $("#results-div").html data
            return
    return


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

form_params = (form) ->
    # autotranslated (and adapted) to coffeescript from
    # http://stackoverflow.com/questions/316781/how-to-build-query-string-with-javascript
    params = new Object()
    res = process_form_element(params, elem) for elem in form.elements
    return res

#defaultDict = (map, defaul) ->
## from http://stackoverflow.com/a/13059975/419338
#    (key) ->
#        return map[key]  if key of map
#        return defaul(key)  if typeof defaul is "function"
#        return defaul
#
#add_group = ->
#    d = $('#groups-div')
#    d[0].className = "visible_div"
#
#    params = form_params($("#form")[0]) # make a query dict out the the form values
#    for k, v of params
#        myval = "#{k}: #{v}..."
#        myval2 = "<input type=\'button\' class=\'btn btn-success btn-xs\' value=\"#{myval}\"> <nbsp>"
#        d.append(myval2);
#    d.append("<br>");
#    console.log "Added group"
#    return
#
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
                d.append data
                # $("#results-div").html data
                return
    )

    return

#clear_groups = ->
#    $('#groups-div').html ""
#    $('#groups-div')[0].className = "invisible_div"

do_refresh = ->
    params = form_params($("#form")[0]) # make a query dict out the the form values
    suffix = jQuery.param(params) # turn dict into query string
    reload "?".concat(suffix)
    return
