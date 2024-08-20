Pylint
======

Our Pylint pipeline plays a crucial role in ensuring the consistency and long-term health of our simulator's code.
It automatically examines our Python code, flagging potential stylistic inconsistencies, enforcing best practices,
and helping prevent subtle errors. This helps us deliver a more robust and maintainable codebase.

To ensure that our code adheres to the established quality standards, our Pylint pipeline requires a successful Pylint run using our custom `.pylintrc` configuration file. This file defines specific coding conventions and style guidelines for our project.

**Example:**

.. code-block:: bash

   pylint --rcfile=./.pylintrc  my_python_script.py other_script.py

* In this example, `./.pylintrc` assumes the configuration file is in the same directory you're running the command. Adjust the path if your `.pylintrc` file is located elsewhere.

For more information on PEP 8 coding style guidelines, see `this resource <https://peps.python.org/pep-0008/>`_.