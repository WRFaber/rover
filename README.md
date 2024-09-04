# Seeker an DRL tutorial in torch

Teaching simulated systems to decide based on RL techniques.



## Setup

1. python3.11 -m venv {venv_name}
2. ./{venv_name}/Scripts/acitvate.bat
3. ensure the environment is activated by checking 'where python' and pip list
    - these should only contain the basic python setup and python version
    - it should point to your local folder 
    - if this is not working switch to bash terminal and run source full path to the environments activate file. 
4. to install GPU enhanced torch follow these steps
    - pip install cuda-python
    - pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 (Note this is due to the fact that the torch has to line up with cuda to get best gpu performance. At the time I am writing this cuda is at 12.2 and 
      torch is at 12.1 support. Thus I am using the latest nightly build rather than the stable version of torch)
