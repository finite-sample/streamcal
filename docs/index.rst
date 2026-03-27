streamcal
=========

Streaming probability calibration via multiplicative weights.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Installation
------------

.. code-block:: bash

   pip install streamcal

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from streamcal import MWUCalibrator

   cal = MWUCalibrator(n_buckets=100, eta=0.1)

   # Stream batches
   for p_raw, y in stream:
       p_calibrated = cal.update(p_raw, y)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
