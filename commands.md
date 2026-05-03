
# train model
clearml-task \               
  --project students-demo \
  --name catboost-exp-1 \
  --script train.py \
  --packages clearml pandas scikit-learn catboost matplotlib \
  --args dataset_id=$DATASET_ID iterations=200 depth=3 learning_rate=0.05 \
  --queue students

