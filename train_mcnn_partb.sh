python main.py --dataset shanghaitechpb \
--model MCNN \
--train-files /home/featurize/work/DIP2021-FinalPJbaseline/TrainingTestingFileLists/ShanghaiTechPartB_full_origin_train.txt \
--val-files /home/featurize/work/DIP2021-FinalPJbaseline/TrainingTestingFileLists/ShanghaiTechPartB_full_origin_val.txt \
--gpu-devices 1 \
--lr 1e-5 \
--train-batch 1 \
--val-batch 1 \
--optim adam \
--loss mseloss \
--checkpoints ./checkpoints/demo \
--summary-write ./runs/demo
