#!/bin/bash

# creating virtual environment 
python3 -m venv myenv

# activating virtual environment 
source myenv/bin/activate

# installing dependecies:
pip3 install scikit-learn pandas numpy graphviz

# running script
echo "Run 'deactivate' to exit the virtual environment"
echo "Run 'source myenv/bin/activate & python3 ss_detector.py' to run GUI of ShadowSocks22 detector based on ML"