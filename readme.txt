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
-------------------------------------------------------------------------------

As you will see, the network performs poorly on the testing data.  I wasn't able 
to get it to generalize and I can't figure out why. I believe my backpropagation 
algorithm is working well, since the mean squared error goes down, and it is able to 
memorize the training examples with 100% accuracy.  I've tried multiple things to prevent 
overfitting, including a ton of parameter tweaking, early stoppage, and even a simple 
"dropout" technique.  Nothing seemed to work though.  

If possible, I would be interested to see the working code released at some point 
so I could see what I was doing wrong!

