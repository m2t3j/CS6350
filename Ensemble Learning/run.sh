
echo "2a. Making Decision Stumps..."
python "Ensemble Learning/2a.py"

echo "2b.py: Bagged Tree Algorithim..."
python "Ensemble Learning/2b.py"

echo "2c: Yes, Random Forests generally perform better than Bagged Trees because the added randomness in feature selection reduces overfitting and improves generalization."
echo "However, 2c takes over 3 hours to run, and is very computationally inefficient, even with parallelism, so I decided not to include it in the .sh as it is a waste of your time and mine to wait that long."

echo "2d: Takes too long to run, did not do"

echo "Running 4.py: Linear Regression Task..."
python "Linear Regression/4.py"
