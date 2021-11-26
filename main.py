import numpy as np
import random

def get_parity(bits:str):
    ones = 0
    for bit in bits:
        if bit == '1':
            ones +=1
    
    parity = ones%2
    
    return parity

# generate_dataset - generate training and testing data
def generate_dataset():
    
    training_data_size = 1000
    training_data = []
    testing_data_size = 100
    testing_data = []

    for i in range(training_data_size):
        # generate a random 4 digit binary number, 
        # get it's parity, store in training_data as tuple
        random_bits = '{0:04b}'.format(random.randint(0,15))
        parity = get_parity(random_bits)
        training_data.append((random_bits, parity))
        

    for i in range(testing_data_size):
        # do the same thing but for testing data set
        random_bits = '{0:04b}'.format(random.randint(0,15))
        parity = get_parity(random_bits)
        testing_data.append((random_bits, parity))

    return training_data, testing_data

layer_sizes = (4,5,1)
input_layer = np.array([])
training_data, testing_data = generate_dataset()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_activation_layer(input, layer_size):
    for i in range(layer_size):
        input[i] = sigmoid(input[i])
    return input

def main():
    bits = training_data[0][0]
    expected_output = training_data[0][1]
    x1 = list(bits)
    x1 = [int(x) for x in x1]
    a1 = get_activation_layer(x1, layer_sizes[0])
    a1 = np.array(a1)
    print('a1')
    print(a1)

    w2 = np.random.normal(size=(layer_sizes[1],layer_sizes[0]))
    print('w2')
    print(w2)

    x2 = np.matmul(w2,a1)
    print('x2')
    print(x2)
    
    a2 = get_activation_layer(x2, 5)
    print('a2')
    print(a2)

    w3 = np.random.normal(size=(layer_sizes[2], layer_sizes[1]))
    print('w3')
    print(w3)

    x3 = np.matmul(w3,a2)
    print('x3')
    print(x3)

    output = sigmoid(x3[0])
    print(output)


if __name__ == "__main__":
    main()