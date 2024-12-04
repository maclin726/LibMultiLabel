# Name of the virtual environment
if [[ $# == 0 ]]; then
    ENV_NAME="lml_py39"
else
    ENV_NAME=$1
fi

# Creating a new conda environment
echo "Clean and Creating a new conda environment named $ENV_NAME"
conda activate base
conda remove -n $ENV_NAME --all --yes
conda create -n $ENV_NAME python=3.9 -y

# Activating the environment
echo "Activating the environment: $ENV_NAME"
conda activate $ENV_NAME

# Installing libraries in requirements.txt
echo "Installing libraries in requirements.txt"
conda install -y anaconda::mkl
cat requirements.txt | xargs -n 1 -L 1 pip3 install

echo "Setup complete. Activate the conda environment using: conda activate $ENV_NAME"

