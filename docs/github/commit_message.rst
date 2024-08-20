Commit Messages
===============

Our commit pipelines act as the first line of defense for code quality. These pipelines execute a battery of tests and
checks whenever new code is introduced, ensuring that changes  meet our standards and won't introduce unexpected errors.

Commit Guidelines
-----------------

To maintain a clear and consistent commit history, please adhere to the following guidelines:

* **Capitalization:** The first letter of your commit message should be capitalized.
* **Character Limit:** Commit messages should aim to stay under 72 characters for readability.
* **Assertive Language:** The first line should use a present-tense, assertive verb (e.g., "Add", "Fix", "Refactor") to clearly describe the change.
* **Additional Description:** If further explanation is required, add a blank line after the first-line summary followed by a more detailed description, possibly formatted as a list.

Examples
---------

**Good Commits:**

* ``Add new simulation output logging``
* ``Fix rendering issue in visualization module``
* ``Refactor unit tests for improved clarity``


**Bad Commits:**

* ``changes to simulation code`` (Too vague)
* ``fixing a bug with the rendering`` (Not capitalized, lacks assertive verb)
* ``Update simulator logic to handle new parameter types this adds a new parameter type for flexibility`` (Exceeds character limit, merges summary with description)

For more comprehensive examples, see `this resource <https://gist.github.com/robertpainsi/b632364184e70900af4ab688decf6f53>`_.