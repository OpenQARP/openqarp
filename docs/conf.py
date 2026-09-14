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

# Furo's default sidebar with a link row under the wordmark (docs/_templates).
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/site-links.html",
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
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/OpenQARP/openqarp",
            "html": (
                '<svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true">'
                '<path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 '
                "5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49"
                "-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08"
                ".58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64"
                "-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 "
                "2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82"
                ".44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95"
                ".29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 "
                '8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>'
            ),
            "class": "",
        },
    ],
}

# The docstrings use bra-ket notation — |psi>, <a| H |b> — which RST parses as
# substitution references, producing "start-string without end-string"
# warnings across dozens of files.  Escaping every ket to satisfy the parser
# would disfigure the physics, so the docutils category is suppressed instead.
# This is what keeps the warning count low enough to read, and is a precondition
# for turning on `-W` in build_docs.sh once the intersphinx fetch noted in that
# file stops failing.
suppress_warnings = ["docutils"]
