COSC3P71 Assignment 3
Name: Ray Keating
Student Number: 6510200

Using a Neural Network as a parity bit checking system

-------------------------------------------------------------------------------
                                  Numpy
-------------------------------------------------------------------------------

This project is built with the help of numpy, so if you don't 
already have it installed you can install it using the pip installer.  

Type "pip3 install numpy" or "pip install numpy"

-------------------------------------------------------------------------------
                           Execution Instructions
-------------------------------------------------------------------------------

To run:

1. Navigate to the assignment folder in your terminal
2. Type the command "python3 main.py" or "python main.py"

If numpy isn't found by your terminal, what worked for me (on linux) was the command

"/bin/python3 main.py"

-------------------------------------------------------------------------------
                          Changing Parameters
-------------------------------------------------------------------------------

All of the neural net related parameters can be changed in the main method.

The layer sizes can be changed with the layer_sizes tuple 
(Note: the third value, the output layer size, must be 1)

The training_examples variable is a list of decimal integers to be converted 
into binary.  It represents the binary numbers that will be given to the 
network as inputs during training.  The training algorithm randomly selects 
from these training examples.  Similarly, the testing_examples is the same 
thing, but it represents the inputs that will be given to the network for testing.

The epochs, learning rate, and testing iterations are all pretty self-explanatory.

-------------------------------------------------------------------------------
                                   Note
-------------------------------------------------------------------------------

The network performs really well when the training examples are the same as the
testing examples.  It is able to memorize the training examples with 100% accuracy, 
in most runs with 10000 epochs.  However, when new examples are added in that it 
hasn't seen during training, the number of correct guesses can vary widely 
from run to run.

