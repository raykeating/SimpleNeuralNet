import numpy as np
import time

def get_parity(bits:str):
    # a simple method to get the parity of a string of 1's and 0's
    ones = 0
    for bit in bits:
        if bit == '1':
            ones +=1
    parity = ones%2

    return parity

def random_bitstring(examples, length):
    # selects a random decimal number from the examples (should be between 0 and 15 if length is 4), 
    # converts it to binary, 
    # gets it's parity, and returns both as a tuple
    format_options = "{0:0" + str(length) + "b}"
    random_bits = format_options.format(np.random.choice(examples))
    parity = get_parity(random_bits)

    # convert bit string to list of ints 
    random_bits = [int(x) for x in random_bits]

    return (random_bits, parity)

def init_weights(layer_sizes):
    # Initialize the weight matrices. 
    # Matrix dimensions are of the layer_sizes tuple in the parameter
    w1 = np.random.rand(layer_sizes[0],layer_sizes[1])
    w2 = np.random.rand(layer_sizes[1], layer_sizes[2])

    return w1, w2

def sigmoid(input):
    return [1/(1 + np.exp(-x)) for x in input]

def forward_prop(input, w1, w2):
    # forward_prop - given the weight matrices and the input values (vector of 1's and 0's), 
    # calculate the output using the forward propagation algorithm
    input = np.array(input)
    hidden = sigmoid(np.dot(input, w1))
    output = np.dot(hidden, w2)
    y = np.array(sigmoid(output))

    return hidden, y

# Note about the backpropagation functions.  I apologize if this doesn't look exactly like the backprop formula from class/tutorials.  
# I wanted to figure out a way to implement it that made more intuitive sense to me and wasn't just copying off slides.
# However, I won't act like I fully understand all of the math behind it. The algorithms were heavily inspired by this article by Matt Mazur
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/comment-page-16/#comments

# delta_rule - the derivative of the total error with respect to the change in a weight between a hidden/input neuron and the output 
# (described better in the linked article)
def delta_rule(target, output, input):
    return (output-target) * (output*(1-output)) * input

# backprop_output_to_hidden - This function changes the weights of w2 (the weights between the hidden layer and the output layer), 
# based on the calculated error gradient.  "eg" is used as shorthand for error gradient.
def backprop_output_to_hidden(target, hidden, output, w2, learning_rate):
    for i in range(w2.shape[0]):
        # i - for each connection from the hidden layer to the output
        hidden_to_output_eg = (learning_rate*(delta_rule(target, output[0], hidden[i])))
        w2[i] = w2[i] - hidden_to_output_eg

    return w2

# backprop_hidden_to_input - This function changes the weights of w1 (the weights between the input layer and the hidden layer), 
# based on the calculated error gradient.  "eg" is used as shorthand for error gradient.
def backprop_hidden_to_input(target, input, hidden, output, w1, w2, learning_rate):
    for i in range(len(w1)): # for each input neuron
        for j in range(len(w1[i])): # for each connection from the input neuron to the hidden layer
            hidden_to_output_eg = delta_rule(target, output[0], w2[j])
            sigmoid_derivative = hidden[j] * (1-hidden[j])
            input_to_hidden_eg = input[i]
            total_eg_wrt_w1_ij = hidden_to_output_eg * sigmoid_derivative * input_to_hidden_eg
            w1[i][j] = w1[i][j] - (learning_rate*total_eg_wrt_w1_ij)

    return w1

# train - This function trains the neural net based on the given parameters.  Returns the computed weights.
def train(layer_sizes, training_examples, epochs, learning_rate):
    # layer_sizes - a tuple describing the size of each layer (the output layer should have a size of 1)
    # training_examples - a list of integers that will be converted to binary strings for training.
    # epochs - the number of times the network adjusts its weight to fit a training example

    if layer_sizes[2] != 1:
        return ValueError("The output layer size must be 1")

    w1, w2 = init_weights(layer_sizes) # initialize the weights to random values between 0 and 1
    iterations = 0
    total_error = 0

    print('Training...')

    for i in range(epochs+1):
        # select a random bit string from the training examples.  bits[0] holds the bit string, bits[1] holds the parity
        bits = random_bitstring(examples=training_examples, length=layer_sizes[0]) 
        error=1
        counter = 0
        # once the model adjusts the error so it is below 0.01, move to the next example.
        while error > 0.01:
            hidden, output = forward_prop(bits[0], w1, w2)
            w2 = backprop_output_to_hidden(bits[1], hidden, output, w2, learning_rate)
            w1 = backprop_hidden_to_input(bits[1], bits[0], hidden, output, w1, w2, learning_rate)
            error = (bits[1] - output)**2 
            iterations += 1
            total_error += error
            mean_squared_error = (total_error/iterations)
            if i % 200 == 0:
                if counter == 0:
                    print(f'MSE at epoch {i}:', mean_squared_error)
                    counter += 1

    # once training is complete, output the results
    print(f'''\nTraining Complete
    \n-------------------------------------
    \nNumber of Hidden Nodes: {layer_sizes[1]}
    \nLearning Rate: {learning_rate}
    \nFinal Mean Squared Error: {mean_squared_error}
    \n-------------------------------------\n''')

    time.sleep(1)
    print("Testing starting in 3 seconds...\n")
    time.sleep(3)

    return w1, w2

def test(testing_iterations, testing_examples, w1, w2):

    testing_iterations = 100
    wrong_guesses = 0

    for i in range(testing_iterations):
        bits = random_bitstring(examples=testing_examples, length=len(w1))
        hidden, output = forward_prop(bits[0], w1, w2)
        expected_output = bits[1]
        guess = round(output[0])
        if guess != expected_output:
            wrong_guesses += 1
            print('wrong guess!')
            print(bits[0])
        # display results 
        print(
            '|', 'example bitstring: ' , bits[0] , '|',
            'parity: ', bits[1], '|',
            'output: ', '{:.4f}'.format(output[0]), '|', 
            'guess: ', round(output[0]), '|'
            )

    print('\nnumber of wrong guesses: ', wrong_guesses, '\n')

def main():

    np.random.seed(0)

    ##-------TRAINING STAGE-------##

    # training parameters
    layer_sizes = (4,8,1) # input, hidden, and output layer sizes. (output layer size should remain 1)
    training_examples = list(range(0,15)) # the decimal numbers to use for training (will be converted to binary)
    epochs = 10000
    learning_rate = 0.8

    # train the network using the given training parameters and return the weights
    # w1: weights from input layer to hidden, w2: weights from hidden layer to output
    w1, w2 = train(layer_sizes, training_examples, epochs, learning_rate) 

    ##-------TESTING STAGE-------##

    # testing parameters
    testing_examples = list(range(0,15)) # the decimal numbers to use for testing (will be converted to binary)
    testing_iterations = 100

    # test the network using the given testing parameters
    test(testing_iterations, testing_examples, w1, w2)

if __name__ == "__main__":
    main()