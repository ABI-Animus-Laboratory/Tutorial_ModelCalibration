# Tutorial_ParetoOptimisation

# Setup

Create a venv in the root directory

`
python -m venv venv
`

activate the python venv

`
. venv/bin/activate

Set your pythonpath to be able to access the required modules

`
export PYTHONPATH=/home/farg967/Documents/git_projects/Tutorial_ModelCalibration
`

Then run any of the scripts you want!

# For running in vscode:

After creating your venv as above, create this configuration in vscode

  "configurations": [
      {
          "name": "Python Debugger: Current File",
          "type": "debugpy",
          "request": "launch",
          "program": "${file}",
          "console": "integratedTerminal",
          "python": "${workspaceFolder}/venv/bin/python",
          "cwd": "${workspaceFolder}",
          "env": {
              "PYTHONPATH": "${workspaceFolder}"
          }
      }
  ]

