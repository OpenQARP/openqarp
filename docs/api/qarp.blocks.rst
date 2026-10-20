``qarp.blocks``
===============

.. automodule:: qarp.blocks
   :members:
   :undoc-members:
   :show-inheritance:

.. py:data:: QarpBlock
   :module: qarp.blocks

   A block built by ``qarp``: one of the six Python block classes
   (``SimpleBlock``, ``CompositeBlockBase``, ``ControlledBlock``,
   ``MeasureBlock``, ``ResetBlock``, ``ConditionalBlock``), which carry members
   a raw ``qarpx`` block lacks (``symbols``, ``parameter_map``).  It annotates
   parameters that read those members, and ``isinstance(b, QarpBlock)`` rejects a
   raw block.  ``AnyBlock`` is any block, raw ones included.
