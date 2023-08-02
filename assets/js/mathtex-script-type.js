// Mostly copied from https://github.com/KaTeX/KaTeX/tree/main/contrib/mathtex-script-type
// Options for global command definitions added by @pdmosses

// import katex from "katex";

const macros = {};

let scripts = document.body.getElementsByTagName("script");
scripts = Array.prototype.slice.call(scripts);
scripts.forEach(function(script) {
    if (!script.type || !script.type.match(/math\/tex/i)) {
        return -1;
    }
    const display =
          (script.type.match(/mode\s*=\s*display(;|\s|\n|$)/) != null);

    const katexElement = document.createElement(display ? "div" : "span");
    katexElement.setAttribute("class",
                              display ? "equation" : "inline-equation");
    try {
        katex.render(script.text, katexElement, {
          displayMode: display,
          globalGroup: true,
          trust: true,
          strict: false,
          throwOnError: false,
          macros
        });
    } catch (err) {
        //console.error(err); linter doesn't like this
        katexElement.textContent = script.text;
    }
    script.parentNode.replaceChild(katexElement, script);
});
