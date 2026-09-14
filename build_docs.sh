#! /bin/bash
set -euo pipefail

# docs/api/ is hand-written and tracked: one page per public package and per
# declared namespace (conventions §15).  autodoc
# honours __all__, so the rendered reference is exactly the declared surface —
# nothing is generated here and no :no-index: patching is needed, because no
# object is documented on two pages any more.
#
# `-W` (warnings as errors) is deliberately NOT set: the one remaining warning
# is an intersphinx fetch of https://docs.python.org/3/objects.inv failing on
# SSL verification in every environment tested (self-signed cert in the chain).
# Flip it on once that is resolved or the intersphinx_mapping is dropped.
cd docs
make clean
make html
cd ..

# Every public package and namespace has a page, and every __all__ name
# rendered into the inventory.
python scripts/ci/check_api_inventory.py docs/_build/html/objects.inv
