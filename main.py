import numpy as np

## ----------------------------------------- ##
## ------------HELPER FUNCTIONS------------- ##
## ----------------------------------------- ##

def get_parity(bits:str):
    # a simple method to get the parity of a string of 1's and 0's
    ones = 0
    for bit in bits:
        if bit == '1':
            ones +=1
    parity = ones%2

    return parity

def parse_bitstring(number, number_of_input_neurons=4):
    # selects a random decimal number from the examples (should be between 0 and 15 if length is 4), 
    # converts it to binary, 
    # gets it's parity, and returns both as a tuple
    
    # convert number to binary string
    format_options = "{0:0" + str(number_of_input_neurons) + "b}"
    bits = format_options.format(number)

    bit_list = [int(x) for x in bits]

    parity = get_parity(bits)

    return (bit_list, parity)

def init_weights(layer_sizes):
    # Initialize the weight matrices. 
    # Matrix dimensions are of the layer_sizes tuple in the parameter
    w1 = np.random.normal(size=(layer_sizes[0],layer_sizes[1]))
    w2 = np.random.normal(size=(layer_sizes[1], layer_sizes[2]))

    return w1, w2

def sigmoid(input):
    try:
        return [1/(1 + np.exp(-x)) for x in input]
    except TypeError: # if input is not iterable
        return 1/(1+np.exp(-input))

def sigmoid_derivative(input):
    try:
        return [sigmoid(x)*(1-sigmoid(x)) for x in input]
    except TypeError: # if input is not iterable
        return sigmoid(input) * (1-sigmoid(input))

## ----------------------------------------- ##
## -----------FORWARD PROPAGATION----------- ##
## ----------------------------------------- ##

def forward_prop(input, w1, w2):
    # forward_prop - given the weight matrices and the input values (vector of 1's and 0's), 
    # calculate the output using the forward propagation algorithm
    input = np.array(input)
    hidden = sigmoid(np.dot(input, w1))
    output = np.dot(hidden, w2)
    output = np.array(sigmoid(output))

    return hidden, output

## ----------------------------------------- ##
## --------BACKPROPAGATION FUNCTIONS-------- ##
## ----------------------------------------- ##

# Note about the backpropagation functions.  I apologize if this doesn't look exactly like the backprop formula from class/tutorials.  
# I wanted to figure out a way to implement it that made more intuitive sense to me and wasn't just copying off slides.
# However, I won't act like I fully understand all of the calculus behind it. The algorithms were heavily inspired by this article by Matt Mazur
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

## ----------------------------------------- ##
## ------------TESTING FUNCTION------------- ##
## ----------------------------------------- ##

def test(testing_examples, w1, w2, show_results=False):

    wrong_guesses = 0

    for i in range(len(testing_examples)):
        bits = parse_bitstring(testing_examples[i])
        hidden, output = forward_prop(bits[0], w1, w2)
        expected_output = bits[1]
        guess = round(output[0])
        
        
        if guess != expected_output:
            wrong_guesses += 1

        error_rate = (wrong_guesses/len(testing_examples))

        if show_results:
        # display results 
            if guess != expected_output:
                print('wrong guess!')
                print(bits[0])
            print(
                '|', 'input: ' , bits[0] , '|',
                'parity: ', bits[1], '|',
                'output: ', '{:.4f}'.format(output[0]), '|', 
                'guess: ', round(output[0]), '|'
                )
    
    if show_results:
        print('\nnumber of wrong guesses: ', wrong_guesses, '\naccuracy: ', f'{100*(1-error_rate)}%')

    return error_rate

## ----------------------------------------- ##
## ------------TRAINING FUNCTION------------ ##
## ----------------------------------------- ##

# train - This function trains the neural net based on the given parameters.  Returns the computed weights.
def train(layer_sizes, training_examples, testing_examples, epochs, learning_rate):
    # layer_sizes - a tuple describing the size of each layer (the output layer should have a size of 1)
    # training_examples - a list of integers that will be converted to binary strings for training.
    # epochs - the number of times the network adjusts its weight to fit a training example

    if layer_sizes[2] != 1:
        return ValueError("The output layer size must be 1")

    w1, w2 = init_weights(layer_sizes) # initialize the weights to random values between 0 and 1

    print('Training...')

    for i in range(epochs):
        # to prevent overfitting (probably would only happen in rare cases), 
        # stop training if both the training and testing error is 0.  
        # Otherwise, continue training until the epoch limit has been reached. 
        # (note, checking the accuracy of the testing set has no impact on the weights)
        testing_error_rate = 1.0
        training_error_rate = 1.0
        epoch_error = 0

        testing_error_rate = test(testing_examples, w1, w2, show_results=False)
        training_error_rate = test(training_examples, w1, w2, show_results=False)
        if (testing_error_rate == 0 and training_error_rate == 0):
            break
        else:
            pass
        
        # training happens in this loop
        for j in range(len(training_examples)): 
            # one epoch - an iteration over the training examples
            # select the next bitstring from the training examples
            # parse so that bits[0] holds the bit string, bits[1] holds the parity
            bits = parse_bitstring(number=training_examples[j], number_of_input_neurons=layer_sizes[0]) 
        
            hidden, output = forward_prop(bits[0], w1, w2)

            w2 = backprop_output_to_hidden(bits[1], hidden, output, w2, learning_rate)
            w1 = backprop_hidden_to_input(bits[1], bits[0], hidden, output, w1, w2, learning_rate)
            
            epoch_error += (bits[1] - output)**2 

        if i % 200 == 0:
            print(f'MSE at epoch {i}: ', (epoch_error/len(training_examples)))          

    # once training is complete, output the results
    print(f'''\nTraining Complete
    \n-------------------------------------
    \nNumber of Hidden Nodes: {layer_sizes[1]}
    \nLearning Rate: {learning_rate}
    \nFinal Mean Squared Error: {(epoch_error/len(training_examples))}
    \n-------------------------------------\n''')

    return w1, w2

def main():

## ----------------------------------------- ##
## ---------------MAIN METHOD--------------- ##
## ----------------------------------------- ##

    ##-------TRAINING STAGE-------##

    # parameters
    layer_sizes = (4,8,1) # input, hidden, and output layer sizes. (output layer size should remain 1)
    training_examples = [0,1,2,3,5,7,8,9,12,13,14,15] # numbers to be used for training (will be converted to binary)
    testing_examples  = [4,6,10,11] # numbers to be used for testing (will be converted to binary)
    
    #epochs can be turned up for higher accuracy on the training examples.  I turned it down so your computer wouldn't explode.
    epochs = 1000
    learning_rate = 0.8

    # train the network using the given training parameters and return the weights
    # w1: weights from input layer to hidden, w2: weights from hidden layer to output
    w1, w2 = train(layer_sizes, training_examples, testing_examples, epochs, learning_rate) 

    ##-------TESTING STAGE-------##

    # test the network using the weights obtained from training, show results
    print('Results on training data\n')
    error = test(training_examples, w1, w2, show_results=True)

    # test the network using the weights obtained from testing, show results
    print('\nResults on testing data\n')
    error = test(testing_examples, w1, w2, show_results=True) 

if __name__ == "__main__":
    main()
