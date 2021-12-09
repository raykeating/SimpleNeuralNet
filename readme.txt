COSC3P71 Assignment 3
Name: Ray Keating
Student Number: 6510200
Dec 2nd, 2021

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
into binary and used to generate inputs and expected ouputs to the network during training. 

The testing_examples is the same thing, but for testing.

The epochs and learning rate are pretty self-explanatory.

-------------------------------------------------------------------------------
                                   Note
------------------------------------------------------------------------------
Although the network was able to memorize training examples with 100% accuracy, I 
wasn't able to get it to generalize to work on new examples (testing_data). I'm not really sure why. 
I believe my backpropagation algorithm is working well, since the mean squared error goes 
down epoch-to-epoch.  I've tried multiple things to prevent overfitting, nothing seemed 
to work though.  Anyways, the general idea is there and it does kind of work.

