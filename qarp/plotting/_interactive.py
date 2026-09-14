"""Interactive notebook output for circuit plots.

Wraps the matplotlib SVG in a self-contained HTML fragment: vanilla CSS/JS
adds hover tooltips (full gate expression, qubits, index) and dims the rest
of the circuit while a gate is hovered. No widget/comm dependencies — the
interactivity is pure client-side, so it survives ``nbconvert`` to HTML.

Gate artists are tagged with SVG gids of the form ``qarp-<uid>-g<index>-a<n>``
(``m<index>`` for terminal measurements); everything sharing one ``g<index>``
key highlights together.
"""

import json
from typing import Dict, List

from .styles import _theme as theme


def interactive_html(svg: str, payload: Dict[str, Dict], uid: str) -> str:
    """Compose the scrollable, hover-enabled container around an SVG string.

    Args:
        svg: SVG document produced by ``fig.savefig(..., format="svg")``.
        payload: gate key (``g3`` / ``m0``) -> {type, label, displayed, qubits, index}.
        uid: unique id suffix for this plot; scopes ids, CSS, and JS.
    """
    data = json.dumps(payload).replace("<", "\\u003c")
    cid = f"qarp-plot-{uid}"
    return f"""
<div id="{cid}" class="qarp-plot" style="position:relative; overflow-x:auto; width:100%;
     border:1px solid {theme.GRID}; border-radius:8px; background:{theme.CARD}; padding:4px;">
  {svg}
  <div class="qarp-tip" style="position:absolute; display:none; z-index:10; max-width:340px;
       background:{theme.INK}; color:#eef2f8; border-radius:8px; padding:8px 11px;
       font-family:'Spline Sans Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
       font-size:12px; line-height:1.5; pointer-events:none;
       box-shadow:0 6px 20px -6px rgba(16,27,45,0.5);"></div>
</div>
<style>
  #{cid} .qarp-gate {{ cursor: pointer; transition: opacity 120ms ease; }}
  #{cid}.qarp-dim .qarp-gate:not(.qarp-hover) {{ opacity: 0.3; }}
</style>
<script>
(function () {{
  var tries = 0;
  function init() {{
    var root = document.getElementById("{cid}");
    if (!root) {{ if (tries++ < 20) requestAnimationFrame(init); return; }}
    var data = {data};
    var tip = root.querySelector(".qarp-tip");
    var groups = {{}};
    root.querySelectorAll('[id^="qarp-{uid}-"]').forEach(function (el) {{
      var m = el.id.match(/^qarp-{uid}-([gm]\\d+)-/);
      if (!m || !data[m[1]]) return;
      el.classList.add("qarp-gate");
      (groups[m[1]] = groups[m[1]] || []).push(el);
    }});
    function esc(s) {{
      return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }}
    Object.keys(groups).forEach(function (key) {{
      var els = groups[key];
      var d = data[key];
      els.forEach(function (el) {{
        el.addEventListener("mouseenter", function () {{
          root.classList.add("qarp-dim");
          els.forEach(function (e) {{ e.classList.add("qarp-hover"); }});
          var head = "<strong style='color:#fff'>" + esc(d.type) + "</strong>" +
            (d.index !== null ? " <span style='color:#94abd6'>#" + d.index + "</span>" : "");
          var body = "";
          if (d.label && d.label !== d.type) body += "<div>" + esc(d.label) + "</div>";
          body += "<div style='color:#aebdd2'>" + esc(d.qubits.join(", ")) + "</div>";
          tip.innerHTML = head + body;
          tip.style.display = "block";
        }});
        el.addEventListener("mousemove", function (ev) {{
          var r = root.getBoundingClientRect();
          var x = ev.clientX - r.left + root.scrollLeft + 14;
          var y = ev.clientY - r.top + 14;
          var maxX = root.scrollLeft + root.clientWidth - tip.offsetWidth - 8;
          tip.style.left = Math.min(x, Math.max(maxX, 8)) + "px";
          tip.style.top = (y + tip.offsetHeight > root.clientHeight - 8
                           ? y - tip.offsetHeight - 24 : y) + "px";
        }});
        el.addEventListener("mouseleave", function () {{
          root.classList.remove("qarp-dim");
          els.forEach(function (e) {{ e.classList.remove("qarp-hover"); }});
          tip.style.display = "none";
        }});
      }});
    }});
  }}
  init();
}})();
</script>
"""


def build_payload(gate_registry: List, measure_payload: List[Dict]) -> Dict[str, Dict]:
    """Assemble the tooltip payload from the plotter's registries."""
    payload = {}
    for g in gate_registry:
        payload[f"g{g.index}"] = {
            "type": g.gate_type,
            "label": g.full_label,
            "displayed": g.displayed_label,
            "qubits": g.qubits,
            "index": g.index,
        }
    for m in measure_payload:
        payload[m["key"]] = {
            "type": "measure",
            "label": "measure",
            "displayed": "M",
            "qubits": m["qubits"],
            "index": None,
        }
    return payload
