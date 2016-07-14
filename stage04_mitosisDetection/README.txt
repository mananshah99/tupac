This folder contains files and scripts used to train a mitosis detector on TUPAC images.

Code listing:
    * draw_nuclei_mitosis.py                    Draws ground truth mitoses over nuclei images
    * evaluate_nuclei.py                        Evaluates mitosis heatmaps (true positive/false negatives) with groun truth
    * stage04-1_genNucleiMasks.py               Generates nuclei masks for all images
    * stage04-2_genTrainingExamples.py          Generates training examples for stage 1 of the mitosis detector (pos from ground truth, neg from other nuclei)
    * stage04-3_genTrainTestLists.py            Generates train and test lists for caffe from training examples generated with stage04-2_genTrainingExamples.py 
    * stage04-4_genTrainingExamplesStage2.py    Generates training examples for stage 2 of the mitosis detector (pos from false neg, neg from false pos)
    * stage04-5_genTrainTestListsStage2.py      Geenrates train and test lists for caffe from training examples generated with stage04-4_genTrainingExamplesStage2.py 


File listing:
    * NOTE: patch size is 100x100

    * Stage 1 (pos from ground truth, neg from everywhere else)
        positive patches
            - location: training_examples/pos
            - number: 30047

        negative patches
            - location: training_examples/neg
            - number: 113104
        
        train image list (balanced classes)
            - location: training_examples/train.lst
            - number: 54084

        validation image list
            - location: training_examples/val.lst  
            - number: 6010 

    * Stage 2 (pos from false neg, neg from false pos)
        positive patches
            - location: training_examples/pos_stage2
            - number: 5910
        
        negative patches
            - location: training_examples/neg_stage2
            - number: 243176
        
        train image list (twice as many negatives as positives)
            - positives from stages 1 and 2
            - negatives from stage 2
            - location: training_examples/train2.lst
            - number: 101583 

        validation image list (twice as many negatives as positives)
            - positives from stages 1 and 2
            - negatives from stage 2
            - location: training_examples/val2.lst
            - number: 11288


Caffe models:
    * Stage 1: P100
    * Stage 2: P100_stage2
