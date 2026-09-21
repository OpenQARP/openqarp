# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "OpenQARP"
html_title = "OpenQARP"
copyright = "2026, Fujitsu Limited"
author = "Stefano Scali, Vicente P. Soloviev, Antonio Márquez Romero, Brian Coyle, Giuseppe Buonaiuto, Annie Paine, Jonathan H. Fetherolf, Marcos Díez García, Michal Krompiec, Josh Kirsopp"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",  # comment out to disable source code links
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
]

# Google-style "Attributes:" sections render as :ivar: fields.  As directives
# they would describe every enum member / dataclass field a second time next
# to the :undoc-members: entry on the same api/ page (duplicate object
# description warnings).
napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# sphinx.ext.todo is loaded above but `.. todo::` directives are silently
# dropped from output unless this is set.
todo_include_todos = True

# sphinx.ext.intersphinx is loaded above but had no mapping, making it a
# no-op. Python stdlib only for now — deliberately not numpy/scipy, which
# would need their own vetting against build_docs.sh's warning count.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_baseurl = "https://openqarp.github.io/openqarp/"
# custom.css carries the landing page's tokens onto Furo's role variables and
# loads Geist itself; theme-sync.js keeps the two sites' theme choice as one.
html_css_files = ["custom.css"]
html_js_files = ["theme-sync.js"]
html_show_sourcelink = False
pygments_style = "a11y-light"
pygments_dark_style = "a11y-dark"

# Furo's default sidebar; the brand carries the site-wide burger menu (docs/_templates).
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ]
}
html_theme_options = {
    "source_repository": "https://github.com/OpenQARP/openqarp",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": [],
}

# The docstrings use bra-ket notation — |psi>, <a| H |b> — which RST parses as
# substitution references, producing "start-string without end-string"
# warnings across dozens of files.  Escaping every ket to satisfy the parser
# would disfigure the physics, so the docutils category is suppressed instead.
# This is what keeps the warning count low enough to read, and is a precondition
# for turning on `-W` in build_docs.sh once the intersphinx fetch noted in that
# file stops failing.
suppress_warnings = ["docutils"]
