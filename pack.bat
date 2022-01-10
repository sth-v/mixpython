py -m pip install --upgrade pip
py -m pip install --upgrade build
del .\dist
py -m build
py -m pip install --upgrade twine
py -m twine upload dist/*