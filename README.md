# Tutorial_ParetoOptimisation

# Setup

Create a venv in the root directory

`
python -m venv venv
`

activate the python venv

`
. venv/bin/activate

install the required libraries

`
pip install scipy matplotlib
`

Set your pythonpath to be able to access the required modules

`
export PYTHONPATH=/path/to/Tutorial_ModelCalibration
`

Then run any of the scripts you want!

# For running in vscode:

After creating your venv as above, create this configuration in vscode

{
  "name": "Python Debugger: Current File",
  "type": "debugpy",
  "request": "launch",
  "program": "${file}",
  "console": "integratedTerminal",
  "cwd": "${workspaceFolder}",
  "env": {
    "PYTHONPATH": "${workspaceFolder}"
  },
  "windows": {
    "python": "${workspaceFolder}\\venv\\Scripts\\python.exe"
  },
  "linux": {
    "python": "${workspaceFolder}/venv/bin/python"
  },
  "osx": {
    "python": "${workspaceFolder}/venv/bin/python"
  }
}

