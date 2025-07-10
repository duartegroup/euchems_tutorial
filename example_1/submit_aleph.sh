#!/bin/bash
#$ -cwd
#$ -l s_rt=360:00:00
#$ -l a40=1 


if [[ -z ${1:-} ]]; then
  echo "Missing name of the scratch folder."
  exit 1
fi

export ORIGIN=$PWD
export SCR=/scratch

# Change directories to the scratch space and copy files

conda activate mlptrain-mace-newest
#Create scratch folder and copy all important files

mkdir -p "$SCR/$1"
cp -r *  "$SCR/$1/"
cd $SCR/$1


python Mg_hexa_xtb.py

wait

# Delete the large files
#rm *.json

# then copy the whole folder back
cd ..
cp -r $1/ $ORIGIN


echo "DONE"

