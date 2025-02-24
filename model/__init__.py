import os
import pkgutil
import importlib

__all__ = []

package_dir = os.path.dirname(__file__)

# Dynamically import all modules in the package
for _, module_name, _ in pkgutil.iter_modules([package_dir]):
    module = importlib.import_module(f"{__name__}.{module_name}")
    globals()[module_name] = module  # Make module accessible in package namespace
    __all__.append(module_name)

    # Automatically import all symbols from each module
    for attr in dir(module):
        if not attr.startswith("_"):  # Ignore private/internal attributes
            globals()[attr] = getattr(module, attr)
            __all__.append(attr)
