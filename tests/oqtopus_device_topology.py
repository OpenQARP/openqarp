"""The device-gateway example topology, transcribed verbatim.

Source: https://github.com/oqtopus-team/device-gateway/blob/main/config/example/device_topology.json
(the published ``device_info`` payload shape; a ``.json`` file is a forbidden
type in this repository's pre-commit, hence a Python literal).  Three qubits
whose ``id`` and ``physical_id`` differ, so a builder that reads the wrong
field is caught; two directed couplings.
"""

DEVICE_TOPOLOGY = {
    "name": "anemone",
    "device_id": "anemone",
    "qubits": [
        {
            "id": 0,
            "physical_id": 0,
            "position": {"x": 0, "y": 0.666666666666667},
            "fidelity": 0.9994051214420587,
            "meas_error": {
                "prob_meas1_prep0": 0.2174072265625,
                "prob_meas0_prep1": 0.242431640625,
                "readout_assignment_error": 0.2247314453125,
            },
            "qubit_lifetime": {"t1": 26.45895242063294, "t2": 35.88675637783199},
            "gate_duration": {"rz": 0, "sx": 16, "x": 24},
        },
        {
            "id": 1,
            "physical_id": 2,
            "position": {"x": 0, "y": 0},
            "fidelity": 0.9995370146509399,
            "meas_error": {
                "prob_meas1_prep0": 0.296875,
                "prob_meas0_prep1": 0.2864990234375,
                "readout_assignment_error": 0.2908935546875,
            },
            "qubit_lifetime": {"t1": 40.57616114919819, "t2": 27.79555721607516},
            "gate_duration": {"rz": 0, "sx": 16, "x": 24},
        },
        {
            "id": 2,
            "physical_id": 3,
            "position": {"x": 0.6666666666666665, "y": 0},
            "fidelity": 0.9996114330540922,
            "meas_error": {
                "prob_meas1_prep0": 0.166015625,
                "prob_meas0_prep1": 0.2252197265625,
                "readout_assignment_error": 0.2021484375,
            },
            "qubit_lifetime": {"t1": 16.684034546269388, "t2": 19.34686595199833},
            "gate_duration": {"rz": 0, "sx": 16, "x": 24},
        },
    ],
    "couplings": [
        {
            "control": 0,
            "target": 1,
            "fidelity": 0.9617900609715081,
            "gate_duration": {"rzx90": 272},
        },
        {
            "control": 2,
            "target": 1,
            "fidelity": 0.985511964171477,
            "gate_duration": {"rzx90": 160},
        },
    ],
    "calibrated_at": "2025-04-20T10:03:16.755183Z",
}
