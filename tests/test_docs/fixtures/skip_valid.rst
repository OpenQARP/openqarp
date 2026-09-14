Skip fixture
============

This block is deliberately broken and opts out.

.. docs-lint: skip illustrative only, not meant to run

.. code-block:: python

    this_name_is_never_bound_anywhere

This block is not skipped and is also broken, to prove skip is per-block.

.. code-block:: python

    another_unbound_name
