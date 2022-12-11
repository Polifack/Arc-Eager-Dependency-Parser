# create conda virtual enviroment and activate it
echo "*** Creating Conda VENV"
conda create -n arc_eager
conda activate arc_eager

# install python libraries
echo "*** Installing Python"
conda install python=3.8.15
echo "*** Installing Libraries"
python -m pip install tensorflow==2.9.2
python -m pip install keras==2.9.0
python -m pip install matplotlib

echo "*** All done"