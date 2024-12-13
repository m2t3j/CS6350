echo 'Running 2a:'
python "Neural Networks\2a.py"


echo "Running 2b:"
python "Neural Networks\2b.py"

echo "Running 2c:"
python "Neural Networks\2c.py"

echo "2d: The Neural networks outperformed SVM and logistic regression, due to them being able to handle non-linear relationships better. However, they require careful tuning of hyperparameters and architecture to achieve optimal results, while SVM and logistic regression are simpler to train."

echo "Running 2e:"
python "Neural Networks\2e.py"

echo 'Running 3a.py'
python "Neural Networks\3a.py"

echo 'Running 3b.py'
python "Neural Networks\3b.py"

echo "3d: The MAP estimatation performed better than the ML estimation in both training and testing. This is due to the MAP inherintley having a regularization penalty due to the prescence of a prior.The hyperparameter v in MAP is analogous toC in SVM, as both regulate model complexity—smaller values enforce stronger regularization, while larger values allow more flexibility. MAP offers a principled Bayesian approach to regularization, whereas C in SVM is an empirical tuning parameter. "