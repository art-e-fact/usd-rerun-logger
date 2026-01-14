.. USD Rerun Logger documentation master file, created by
   sphinx-quickstart on Wed Dec 31 17:44:32 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

USD Rerun Logger
================

Rerun.io logger for USD, NVIDIA Isaac Sim, and Isaac Lab.

.. image:: https://img.shields.io/badge/docs-online-blue
   :target: https://art-e-fact.github.io/usd-rerun-logger/
   :alt: Documentation

.. image:: https://img.shields.io/pypi/v/usd-rerun-logger
   :target: https://pypi.org/project/usd-rerun-logger/
   :alt: PyPI

Use this package to visualize your OpenUSD stages and Omniverse simulations with `Rerun.io <https://rerun.io/>`_.

View the project on GitHub: https://github.com/art-e-fact/usd-rerun-logger

Installation
------------

1. Install the logger from PyPI:

   .. code-block:: bash

      pip install usd-rerun-logger

2. Install OpenUSD (``pxr``). This is a **user-managed dependency** to avoid version conflicts.

   - **Isaac Sim / Isaac Lab**: Skip this step (it's included).
   - **Standalone**:
     
     .. code-block:: bash

        pip install usd-core


.. toctree::
   :maxdepth: 2
   :caption: Contents:


API Reference
=============

Core classes
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~usd_rerun_logger.UsdRerunLogger
   ~usd_rerun_logger.IsaacLabRerunLogger
   ~usd_rerun_logger.LogRerun