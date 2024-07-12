#!/bin/sh

set -o noglob

### not meant to be run directly; simply a wrapper executable
#export LD_LIBRARY_PATH=${LD_LIB_PATH}

echo "CMSSW directory: ${CMSSW_DIR}"

source /cvmfs/grid.desy.de/etc/profile.d/grid-ui-env.sh
source /cvmfs/cms.cern.ch/cmsset_default.sh

wdir=`pwd -P`

cd ${CMSSW_DIR}/src
eval `scramv1 runtime -sh`
cd ${wdir}

echo 'Job execution starts at '$(date)' on host '${HOSTNAME}
echo

eval "$@"

if [ $? -eq 0 ]; then
    echo 'Job execution ends at '$(date)
fi
