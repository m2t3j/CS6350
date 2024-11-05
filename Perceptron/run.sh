

echo "2a: Running a Standard Perceptron model"
python "Perceptron/a.py"

echo "\n"
echo "2b: Running a Voted Perceptron model"
python "Perceptron/b.py"

echo "2c: Running a Averaged Perceptron model"
python "Perceptron/c.py"


echo "2d: Comparing all three of these models, the Averaged and Voted models performed relatively the same and had a small test error, meaning they captured the pattern in the data well. Both were way better than the Standard model, which was too simple and could not correctly classify all points in the test data."

