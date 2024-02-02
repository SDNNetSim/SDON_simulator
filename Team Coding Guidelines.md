# ACNL Team Coding Guidelines

## 1. Naming Conventions

1. **Helper Scripts**: Files in the `helper_scripts` directory should be named using the
   pattern `<script_name>_helpers.py`.
2. **Data Structures**: Name variables with their type, e.g., `<name>_list`, `<name>_dict`, `<name>_set`.
3. **Class Properties**: Include a dictionary named `<ClassName>_props` in class constructors for properties.
4. **Inner Classes**: Name classes within a constructor `<ClassName>_Obj` to indicate scope and relationship.

## 2. Directory and File Structure

1. **Argument Scripts**: Place external files with arguments in the `arg_scripts` directory,
   named `<file_name>_args.py`.
2. **Module Naming**: Directories with Python scripts should follow `<name>_scripts` naming convention.

## 3. Coding Practices

1. **Function Names**: Use assertive and descriptive names like `get`, `create`, `update`.
2. **Type Annotations**: Explicitly list variable types in all function parameters.
3. **Commenting and Documentation**:
    - Use `# FIXME:` for areas needing future fixes, with a brief explanation if necessary.
    - Use `# TODO:` for planned enhancements or tasks, with a concise description.
4. **Argument Labeling**: Label arguments explicitly when calling functions.

## 4. Testing and Quality Assurance

1. **Comprehensive Testing**: Test every function and its branches thoroughly.
2. **Formatting**: Use an auto-formatting tool to maintain consistent code formatting, adhering to a style guide like
   PEP 8.

## 5. Additional Considerations

1. **Class and File Naming Alignment**: Ensure class names match their `.py` file names.
2. **Argument Documentation**: Comment each argument in argument scripts to explain its purpose and expected values.
