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

// The burger beside the wordmark: click toggles, Escape or a click elsewhere closes.
document.addEventListener("DOMContentLoaded", function () {
  var btn = document.getElementById("menu-toggle"), menu = document.getElementById("site-menu");
  if (!btn || !menu) return;
  function setMenu(open) {
    menu.hidden = !open;
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    btn.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    btn.classList.toggle("open", open);
  }
  btn.addEventListener("click", function () { setMenu(menu.hidden); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape" && !menu.hidden) { setMenu(false); btn.focus(); } });
  document.addEventListener("click", function (e) {
    if (!menu.hidden && !e.target.closest("#site-menu") && !e.target.closest("#menu-toggle")) setMenu(false);
  });
});
