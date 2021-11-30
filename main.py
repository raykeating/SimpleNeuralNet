from typing import Type
import numpy as np
import random

# shorthands: eg -> error gradient

def get_parity(bits:str):
    ones = 0
    for bit in bits:
        if bit == '1':
            ones +=1
    parity = ones%2
    return parity

def random_bitstring():
    # generate a random 4 digit binary number, 
    # get it's parity, return both as tuple
    random_bits = '{0:04b}'.format(random.randint(0,15))
    parity = get_parity(random_bits)
    random_bits = [int(x) for x in random_bits]
    return (random_bits, parity)

def init_weights(layer_sizes=(4,8,1)):
    w1 = np.random.rand(layer_sizes[0],layer_sizes[1])
    w2 = np.random.rand(layer_sizes[1], layer_sizes[2])

    return w1, w2

def sigmoid(input):
    return [1/(1 + np.exp(-x)) for x in input]
    

def forward_prop(input, w1, w2):
    input = np.array(input)
    hidden = sigmoid(np.dot(input, w1))
    output = np.dot(hidden, w2)
    y = np.array(sigmoid(output))
    return hidden, y

def delta_rule(target, output, hidden):
    return (output-target) * (output*(1-output)) * hidden

def backprop_output_to_hidden(target, hidden, output, w2, learning_rate):
    for i in range(w2.shape[0]):
        # i - for each connection from the hidden layer to the output
        hidden_to_output_eg = (learning_rate*(delta_rule(target, output[0], hidden[i])))
        w2[i] = w2[i] - hidden_to_output_eg
    return w2

def backprop_hidden_to_input(target, input, hidden, output, w1, w2, learning_rate):
    for i in range(len(w1)): # for each input neuron
        for j in range(len(w1[i])): # for each connection from the input neuron to the hidden layer
            hidden_to_output_eg = delta_rule(target, output[0], w2[j])
            sigmoid_derivative = hidden[j] * (1-hidden[j])
            input_to_hidden_eg = input[i]
            total_eg_wrt_w1_ij = hidden_to_output_eg * sigmoid_derivative * input_to_hidden_eg
            w1[i][j] = w1[i][j] - (learning_rate*total_eg_wrt_w1_ij)
    return w1

def main():

    # training stage
    w1, w2 = init_weights() # w1: weights from input layer to hidden, w2: weights from hidden layer to output
    for i in range(1000):
        bits = random_bitstring()
        for j in range(100):
            hidden, output = forward_prop(bits[0], w1, w2)
            w2 = backprop_output_to_hidden(bits[1], hidden, output, w2, learning_rate=0.8)
            w1 = backprop_hidden_to_input(bits[1], bits[0], hidden, output, w1, w2, learning_rate=0.8)
            loss = (bits[1] - output)**2
            print(f'loss at generation {j+1}: ', loss)
    
    # testing stage
    wrong_guesses = 0
    for i in range(1000):
        bits = random_bitstring()
        hidden, output = forward_prop(bits[0], w1, w2)
        expected_output = bits[1]
        guess = round(output[0])
        if guess != expected_output:
            wrong_guesses += 1
            print('wrong guess')
            print(bits[0])
        print('expected: ', bits[1], 'actual output: ', output[0], 'guess: ', round(output[0]))
    
    print('number of wrong guesses: ', wrong_guesses)

if __name__ == "__main__":
    main()