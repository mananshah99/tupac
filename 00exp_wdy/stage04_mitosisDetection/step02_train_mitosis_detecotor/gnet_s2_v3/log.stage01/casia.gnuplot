reset
set terminal png
set style data lines
set key right

###### Fields in the data file your_log_name.log.train are
###### Iters Seconds TrainingLoss LearningRate

# Training loss vs. training iterations
set title "Training loss vs. training iterations"
set xlabel "Training loss"
set ylabel "Training iterations"

set output "casia_log_train.png"
plot "tmpinfo.train" using 1:3 title "mnist"

# Training loss vs. training time
set output "casia_log_train_loss.png"
plot "tmpinfo.train" using 2:3 title "mnist"

# Learning rate vs. training iterations;
# plot "tmpinfo.train" using 1:4 title "mnist"

# Learning rate vs. training time;
# plot "tmpinfo.train" using 2:4 title "mnist"


###### Fields in the data file your_log_name.log.test are
###### Iters Seconds TestAccuracy TestLoss

# Test loss vs. test iterations
set output "casia_log_test_loss.png"
plot "tmpinfo.test" using 1:4 title "mnist"

# Test accuracy vs. test iterations
set output "casia_log_test_acc.png"
plot "tmpinfo.test" using 1:3 title "mnist"

# Test loss vs. training time
# plot "tmpinfo.test" using 2:4 title "mnist"

# Test accuracy vs. training time
# plot "tmpinfo.test" using 2:3 title "mnist"
