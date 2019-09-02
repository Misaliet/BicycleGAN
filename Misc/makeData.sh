#!/bin/bash

if [ $# != 5 ] ; then 
echo "need 5 parameters (satellite images path / map images path / original picture number / picture mode / digit of file name)"
echo "picture mode: 1 for BtoA and other number for AtoB)"
exit 1; 
fi

# echo "original picture number: $1"
echo "picture mode: $2 (0 for AtoB and 1 for BtoA)"

cd $1
echo "picture mode: $2 (0 for AtoB and 1 for BtoA)"
i=0
for file in `ls -1`
do
    real_list[i]=$1/$file
    i=`expr $i + 1`
done
# echo ${real_list[*]}

cd $2
j=0
for file in `ls -1`
do
    sketch_list[j]=$2/$file
    j=`expr $j + 1`
done
# echo ${sketch_list[*]}

cd /Users/liet/code/Leeds/final/imageProcessing/
for ((k=0; k<$3; ++k))
do
    python /Users/liet/code/Leeds/final/imageProcessing/imageProcess.py -r ${real_list[k]} -s ${sketch_list[k]} -m $4 -d $5
    echo ${real_list[k]}
    echo ${sketch_list[k]}
done