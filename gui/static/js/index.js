// Generated by CoffeeScript 1.7.1
var add_group, do_refresh, form_params, process_form_element, reload;

reload = function(my_suffix) {
  $("#results-div").html("");
  $.ajax({
    url: "/analyse".concat(my_suffix),
    success: function(data) {
      $("#results-div").html(data);
    }
  });
};

process_form_element = function(output_hash, element) {
  var checked, name, tagName, type, value;
  name = element.name;
  value = element.value;
  type = element.type;
  checked = element.checked;
  tagName = element.tagName;
  if (tagName === "TEXTAREA") {
    output_hash[name] = value;
  } else if (tagName === "INPUT") {
    switch (type) {
      case "text":
      case "hidden":
      case "password":
        output_hash[name] = value;
        break;
      case "radio":
      case "checkbox":
        if (checked) {
          if (value) {
            output_hash[name] = value;
          } else {
            output_hash[name] = "on";
          }
        }
    }
  }
  return output_hash;
};

form_params = function(form) {
  var elem, params, res, _i, _len, _ref;
  params = new Object();
  _ref = form.elements;
  for (_i = 0, _len = _ref.length; _i < _len; _i++) {
    elem = _ref[_i];
    res = process_form_element(params, elem);
  }
  return res;
};

add_group = function() {
  var box, checked_boxes, myurl, paramdiv, suffix, _i, _j, _len, _len1, _ref, _ref1;
  checked_boxes = new Object();
  _ref = document.getElementById('form').children;
  for (_i = 0, _len = _ref.length; _i < _len; _i++) {
    paramdiv = _ref[_i];
    checked_boxes[paramdiv.id] = [];
    _ref1 = paramdiv.children;
    for (_j = 0, _len1 = _ref1.length; _j < _len1; _j++) {
      box = _ref1[_j];
      if (box.checked) {
        checked_boxes[paramdiv.id].push(box.name);
      }
    }
  }
  suffix = jQuery.param(checked_boxes);
  suffix = "?".concat(suffix);
  myurl = "/add_group".concat(suffix);
  $.ajax({
    url: myurl,
    success: function(data) {
      var d;
      d = $('#groups-div');
      d[0].className = "visible_div";
      d.append(data);
    }
  });
};

do_refresh = function() {
  var params, suffix;
  params = form_params($("#form")[0]);
  suffix = jQuery.param(params);
  reload("?".concat(suffix));
};

//# sourceMappingURL=index.map
