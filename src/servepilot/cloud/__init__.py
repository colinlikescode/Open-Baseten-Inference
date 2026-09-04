"""Cloud support.

Two pieces live here:

* ``catalog`` – a small table of instance shapes (GPU model, count, memory, NVLink) so you can
  run ``servepilot plan`` for machines you have not rented yet.
* ``skypilot`` – launching ServePilot on AWS, GCP or Azure, in your own account. All
  provisioning goes through SkyPilot; ServePilot writes a SkyPilot task and calls the ``sky`` CLI.
  There is no other cloud path.
"""
