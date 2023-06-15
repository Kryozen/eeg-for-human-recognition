# Bash script to install dependencies if needed
# Dependencies should be already installed if using the virtual environment provided with the project

# Adding general dependencies
## Numpy
pip install numpy

# Adding dependencies for module: loading
## Scipy
pip install scipy

# Adding dependencies for module: preprocessing.py
## Pywt
pip install pywavelets

## Scikit-learn
pip install scikit-learn

# Adding dependencies for module: classification.py
## Tqdm
pip install tqdm
## Tensorflow
pip install tensorflow
pip install tensorflow_decision_forests
## Keras
pip install keras
## Xgboost
pip install xgboost