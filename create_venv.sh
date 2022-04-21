#! /bin/bash
virtualenv -p /usr/bin/python3.8 venv
venv/bin/pip3.8 install ipympl
venv/bin/pip3.8 install pandas
venv/bin/pip3.8 install scipy
venv/bin/pip3.8 install numpy
venv/bin/pip3.8 install stl
venv/bin/pip3.8 install pyquaternion
venv/bin/pip3.8 install jupyterlab
venv/bin/pip3.8 install matplotlib
venv/bin/jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
venv/bin/jupyter labextension install jupyter-matplotlib --no-build
venv/bin/jupyter lab build
