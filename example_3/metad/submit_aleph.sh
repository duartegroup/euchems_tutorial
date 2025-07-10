#!/bin/bash
#$ -cwd
#$ -l s_rt=360:00:00
#$ -l rtx2080=1
#$ -pe smp 10



if [[ -z ${1:-} ]]; then
  echo "Missing name of the scratch folder."
  exit 1
fi

export ORIGIN=$PWD
export SCR=/scratch

# Change directories to the scratch space and copy files

conda activate mlptrain-mace-newest

module load openmpi4.1.1 
export PATH=/usr/local/orca_5_0_3:$PATH
export CUDA_VISIBLE_DEVICES=$SGE_GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMPI_MCA_hwloc_base_binding_policy=none

#Create scratch folder and copy all important files

mkdir -p "$SCR/$1"
cp -r *  "$SCR/$1/"
cd $SCR/$1


python sn2_metad.py

wait

# Delete the large files
#rm *.json

# then copy the whole folder back
cd ..
cp -r $1/ $ORIGIN


echo "DONE"

