#!/bin/bash
# run this script bash run_eval.sh

########################################
############ Configure this ############
########################################
# dir where you store outputs of all envs
DATA_DIR=/workspace/v2maps
TARTANVO_DIR=/workspace/ws_tartanvo
declare -a SCENES=('abandon_scene1')
# declare -a DIFFICULTY=('Data_easy')
# declare -a StringArray=('P000')

# set this to your env folder
# declare -a SCENES=('industrialhanger_scene1' 'industrialhanger_scene2' 'industrialhanger_scene3' 'modneighbourhood_scene1' 'modneighbourhood_scene2' 'modurbancity_scene1.0')
# dir where your collected data is. i.e. Parent folder of Data_easy, etc
declare -a DIFFICULTY=('Data_easy' 'Data_mid'  'Data_hard')
declare -a StringArray=('P000' 'P001' 'P002' 'P003' 'P004' 'P005' 'P006' 'P007' 'P008' 'P009')

# batch size, num_worker and threshold error for frames; can use default else set this accordingly
BS=1
WORKER=1
THRESHOLD_AVG=0.1
THRESHOLD_TRANS=0.2
THRESHOLD_ROT=0.02


############################################################
############ You should not need to change this ############
############################################################
# model and eval script

EVAL_SCRIPT=$TARTANVO_DIR/vo_trajectory_from_folder.py
MODEL=$TARTANVO_DIR/models/tartanvo_1914.pkl

# eval all data; you should not need to change this
for s in ${SCENES[@]}; do
    SCENE_DATA_DIR=$DATA_DIR/$s
    echo $SCENE_DATA_DIR
    for d in ${DIFFICULTY[@]}; do
        for p in ${StringArray[@]}; do
            TEST_IMAGES=$SCENE_DATA_DIR/$d/$p/image_left
            GT_POSES=$SCENE_DATA_DIR/$d/$p/pose_left.txt
            RESULTS=$SCENE_DATA_DIR/results/$d/$p/

            echo $SCENE_DATA_DIR/$d/$p 
            if [ -d $SCENE_DATA_DIR/$d/$p ]; 
            then
                # create folder if not exists
                mkdir -p $SCENE_DATA_DIR/results/$d/$p
                # eval
                python $EVAL_SCRIPT --traj-num $p\
                                    --model-name $MODEL \
                                    --batch-size $BS \
                                    --worker-num $WORKER \
                                    --test-dir $TEST_IMAGES \
                                    --pose-file $GT_POSES \
                                    --results $RESULTS \
                                    --threshold-error-avg $THRESHOLD_AVG \
                                    --threshold-error-trans $THRESHOLD_TRANS \
                                    --threshold-error-rot $THRESHOLD_ROT \
                                    --scene $s \
                                    --difficulty $d
            else
                echo "Trajectory does not exits."
            fi
        done
    done
done
