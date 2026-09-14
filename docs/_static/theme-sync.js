// Mirror Furo's theme choice into the landing page's "qarp-theme" key (same
// origin), so the two toggles act as one.  "auto" clears it: the site then
// follows the OS, as it does for a first-time visitor.
document.addEventListener("DOMContentLoaded", function () {
  var buttons = document.getElementsByClassName("theme-toggle");
  Array.prototype.forEach.call(buttons, function (button) {
    button.addEventListener("click", function () {
      var mode = document.body.dataset.theme;
      try {
        if (mode === "dark" || mode === "light") localStorage.setItem("qarp-theme", mode);
        else localStorage.removeItem("qarp-theme");
      } catch (e) { /* private mode */ }
    });
  });
});
