
mkdir ./projects/
mkdir ./projects/membrane-beta_4state
cd projects/membrane-beta_4state
echo -e "This is a template, follow the steps: " > readme.txt
echo "Notes for individual project" > notes.txt

mkdir ./scripts/ # directory for all the python/R/perl scripts
mkdir ./bash/ # driver scripts that call all other scripts and execute pipelines
touch ./bash/runall.sh # the main driver script
touch ./bash/filter.sh # scripts that I use usually only once to filter input directories and create soft links
mkdir ./input/ # input directories
mkdir ./output/ # output directories
mkdir ./logs # stdout and stderr of runall.sh scripts
touch readme.txt # description of files and scripts in this project folder
touch commands.txt # commands that I run in this directory

mkdir datasets
mkdir general_scripts
mkdir codes
cd codes
echo "#This is a sample code" > samplecode.py
cd ..
mkdir bin
mkdir output
cd

