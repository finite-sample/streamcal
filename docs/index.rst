streamcal
=========

Streaming probability calibration with monotonicity guarantee.

**Calibration** means finding a monotonic transformation f such that
E[Y | f(p) = q] = q. The monotonicity requirement is essential—without it,
you destroy the model's discrimination ability.

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

**Streaming Isotonic Calibration (EMA + PAV)**

.. code-block:: python

   import numpy as np
   from streamcal import StreamingIsotonicCalibrator

   cal = StreamingIsotonicCalibrator(n_buckets=100, alpha=0.1)

   # Stream batches
   for p_raw, y in stream:
       p_calibrated = cal.update(p_raw, y)
       # Guaranteed: monotonic output, adapts to drift

Algorithm
---------

``StreamingIsotonicCalibrator`` uses EMA + PAV:

1. Partition probability space into B buckets
2. Maintain EMA of outcome rate per bucket
3. On new batch:

   - Update EMA: ``rate[b] = (1-α)*rate[b] + α*batch_rate[b]``
   - Project onto monotonic cone via PAV

4. Calibrate via interpolation

**Complexity:** O(B) memory, O(B) per batch

**Drift adaptation:** α controls bias-variance tradeoff. Larger α adapts
faster but has more variance.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
