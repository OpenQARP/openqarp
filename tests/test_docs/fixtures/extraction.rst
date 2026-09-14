Extraction fixture
===================

First block, ordinary indentation.

.. code-block:: python

    a = 1
    b = 2

A non-python block that must not be extracted.

.. code-block:: bash

    echo "not python"

A block nested inside another directive (3-space indent, mirrors
docs/source/configuration.rst's ``py:method`` blocks).

.. note::

   .. code-block:: python

      c = 3

The short ``.. code::`` form is a valid alias.

.. code:: python

    d = 4

A bare directive with no language argument inherits the page's
``highlight_language`` (Sphinx defaults to Python) and counts as a python block.

.. code-block::

    e = 5
