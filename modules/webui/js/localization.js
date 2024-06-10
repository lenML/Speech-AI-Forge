var re_num = /^[.\d]+$/;

var original_lines = {};
var translated_lines = {};

function hasLocalization() {
  return window.localization && Object.keys(window.localization).length > 0;
}

function textNodesUnder(el) {
  var n,
    a = [],
    walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
  while ((n = walk.nextNode())) a.push(n);
  return a;
}

function canBeTranslated(node, text) {
  if (!text) return false;
  if (!node.parentElement) return false;
  var parentType = node.parentElement.nodeName;
  if (
    parentType == "SCRIPT" ||
    parentType == "STYLE" ||
    parentType == "TEXTAREA"
  )
    return false;
  if (re_num.test(text)) return false;
  return true;
}

function getTranslation(text) {
  if (!text) return undefined;

  if (translated_lines[text] === undefined) {
    original_lines[text] = 1;
  }

  var tl = localization[text];
  if (tl !== undefined) {
    translated_lines[tl] = 1;
  }

  return tl;
}

function processTextNode(node) {
  var text = node.textContent.trim();

  if (!canBeTranslated(node, text)) return;

  var tl = getTranslation(text);
  if (tl !== undefined) {
    node.textContent = tl;
    if (text && node.parentElement) {
      node.parentElement.setAttribute("data-original-text", text);
    }
  }
}

/**
 *
 * @param {HTMLElement} node
 * @returns
 */
function processMDNode(node) {
  const text = node.children[0].textContent.trim();
  let tl = getTranslation(text);

  if (!tl) return;
  if (Array.isArray(tl)) {
    tl = tl.join("\n");
  }
  const md = marked.marked(tl);
  node.innerHTML = md;

  node.setAttribute("data-original-text", text);
}

function is_md_child(node) {
  while (node.parentElement !== document.body) {
    if (node?.classList?.contains("md")) {
      return true;
    }
    node = node.parentElement;
    if (!node) break;
  }
  return false;
}

function processNode(node) {
  if (node.nodeType == 3) {
    processTextNode(node);
    return;
  }
  if (node.classList.contains("md")) {
    processMDNode(node);
    return;
  }
  if (is_md_child(node)) return;

  if (node.title) {
    let tl = getTranslation(node.title);
    if (tl !== undefined) {
      node.title = tl;
    }
  }

  if (node.placeholder) {
    let tl = getTranslation(node.placeholder);
    if (tl !== undefined) {
      node.placeholder = tl;
    }
  }

  textNodesUnder(node).forEach(function (node) {
    if (is_md_child(node)) return;
    processTextNode(node);
  });
}

function refresh_style_localization() {
  processNode(document.querySelector(".style_selections"));
}

function refresh_aspect_ratios_label(value) {
  label = document.querySelector("#aspect_ratios_accordion div span");
  translation = getTranslation("Aspect Ratios");
  if (typeof translation == "undefined") {
    translation = "Aspect Ratios";
  }
  label.textContent = translation + " " + htmlDecode(value);
}

function localizeWholePage() {
  processNode(gradioApp());

  function elem(comp) {
    var elem_id = comp.props.elem_id
      ? comp.props.elem_id
      : "component-" + comp.id;
    return gradioApp().getElementById(elem_id);
  }

  for (var comp of window.gradio_config.components) {
    if (comp.props.webui_tooltip) {
      let e = elem(comp);

      let tl = e ? getTranslation(e.title) : undefined;
      if (tl !== undefined) {
        e.title = tl;
      }
    }
    if (comp.props.placeholder) {
      let e = elem(comp);
      let textbox = e ? e.querySelector("[placeholder]") : null;

      let tl = textbox ? getTranslation(textbox.placeholder) : undefined;
      if (tl !== undefined) {
        textbox.placeholder = tl;
      }
    }
  }
}

/**
 *
 * @param {HTMLElement} node
 */
function isNeedTranslate(node) {
  if (!node) return false;
  if (!(node instanceof HTMLElement)) return true;
  while (node.parentElement !== document.body) {
    if (node.classList.contains("no-translate")) {
      return false;
    }
    node = node.parentElement;
    if (!node) break;
  }
  return true;
}

document.addEventListener("DOMContentLoaded", function () {
  if (!hasLocalization()) {
    return;
  }

  onUiUpdate(function (m) {
    m.forEach(function (mutation) {
      Array.from(mutation.addedNodes)
        .filter(isNeedTranslate)
        .forEach(function (node) {
          processNode(node);
        });
    });
  });

  localizeWholePage();

  if (localization.rtl) {
    // if the language is from right to left,
    new MutationObserver((mutations, observer) => {
      // wait for the style to load
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.tagName === "STYLE") {
            observer.disconnect();

            for (const x of node.sheet.rules) {
              // find all rtl media rules
              if (Array.from(x.media || []).includes("rtl")) {
                x.media.appendMedium("all"); // enable them
              }
            }
          }
        });
      });
    }).observe(gradioApp(), { childList: true });
  }
});
