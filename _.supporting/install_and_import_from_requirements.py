
def install_and_import_from_requirements(requirements_file):
    import pip  
    with open(requirements_file, 'r') as file:
        modules = [line.strip() for line in file]
    # install modules
    for module in modules:
        pip.main(['install', module])
    # import modules
    imported_modules = {}
    for module in modules:
        module_name = module.split('==')[0]  # remove version if it's there
        try:
            imported_modules[module_name] = __import__(module_name)
            logging.info(f'Successfully imported {module_name}')
        except ImportError:
            logging.error(f'Could not import {module_name}. Is it installed?')
    return imported_modules
