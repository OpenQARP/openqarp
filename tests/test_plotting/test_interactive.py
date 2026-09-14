"""interactive_html / build_payload: pure string and dict assembly."""

import json
from types import SimpleNamespace

from qarp.plotting._interactive import build_payload, interactive_html


def _gate(index, gate_type="Ry", label="Ry(0.5)", displayed="Ry", qubits=("q[0]",)):
    return SimpleNamespace(
        index=index,
        gate_type=gate_type,
        full_label=label,
        displayed_label=displayed,
        qubits=list(qubits),
    )


def test_build_payload_gate_and_measure_entries():
    payload = build_payload(
        [_gate(0), _gate(3, gate_type="CX", qubits=("q[0]", "q[1]"))],
        [{"key": "m0", "qubits": ["q[2]"]}],
    )
    assert set(payload) == {"g0", "g3", "m0"}
    assert payload["g3"]["type"] == "CX"
    assert payload["g3"]["qubits"] == ["q[0]", "q[1]"]
    assert payload["m0"] == {
        "type": "measure",
        "label": "measure",
        "displayed": "M",
        "qubits": ["q[2]"],
        "index": None,
    }


def test_interactive_html_embeds_svg_and_escaped_payload():
    payload = build_payload([_gate(0, label="Ry(<x>)")], [])
    html = interactive_html("<svg>circuit</svg>", payload, uid="abc123")

    assert "<svg>circuit</svg>" in html
    assert 'id="qarp-plot-abc123"' in html
    # Payload is JSON with '<' escaped so it cannot close the script tag.
    assert "\\u003c" in html
    assert "</script>" in html
    # The embedded JSON round-trips.
    start = html.index("var data = ") + len("var data = ")
    end = html.index(";", start)
    assert json.loads(html[start:end]) == payload


def test_interactive_html_scopes_ids_by_uid():
    html = interactive_html("<svg/>", {}, uid="u1")
    assert "qarp-u1-" in html
    assert "qarp-plot-u1" in html
