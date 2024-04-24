Getting Started
===============

Welcome to the Software Defined Elastic Optical Networking Simulator! In this tutorial, we'll guide you through the
process of running your first simulations and visualizing the results.

To begin, head over to the ``ini`` directory and explore the provided examples. These examples, located in the
``example_ini`` directory, are based on default parameters used by two PhD students in their simulations.
Although named after their creators, these examples are fully customizable to suit your specific needs, offering
flexibility in simulation design.

The diagram below illustrates the location:

.. image:: _images/example_ini.png
   :alt: Example ini Image
   :width: 700px
   :height: 600px
   :align: center

.. raw:: html

    <br>

Select an example that suits your requirements and copy it to the ``run_ini`` directory. Rename the copied file to
``config.ini``. This file contains the assumptions used in simulation runs. For detailed information on available
parameters and their validity, refer to the configuration arguments documentation page accessible from the argument
scripts page.

The ``config.ini`` file comprises different sections, including general settings, topology settings, SNR settings,
AI settings, and file settings. This serves as your baseline simulation assumptions or 's1'. Additionally, you may
encounter other sections labeled 's2', 's3', 's4', and so forth in the examples. These represent subsequent simulation
runs with varying assumptions. For instance, adding an 's2' section with 'k_paths=2' indicates a change in the number
of paths from source to destination in 'simulation 2'. Any parameters not specified in these sections default to the
values defined in 's1'. These processes run concurrently, offering flexibility in experimentation, although running
multiple processes simultaneously is optional.

With your configuration file set up, let's proceed to run your first simulation. If you're not using artificial
intelligence, execute the script 'run_sim.py'. For this tutorial, we recommend starting with a small number of
requests (e.g., 10) to ensure quick completion.

During simulation execution, important input data is saved in the 'input' directory within the 'data' directory,
organized by date and time down to the millisecond for easy reference. Once the simulation concludes, navigate to the
'output' directory. Here, you'll find corresponding output data organized by simulation runs ('s1', 's2', etc.), each
containing results specific to the traffic volume simulated.

With your simulations completed, it's time to visualize and interpret the results. Let's move on to plotting and
analyzing the simulator's output.
