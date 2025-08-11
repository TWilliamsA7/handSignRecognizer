from DataCapture.capture import capture_images, capture_all_images, capture_test_data
from DataCapture.preprocess import preprocess_all_images, preprocess_label
from DataCapture.balance import balance_labels
from DataCapture import labels
import sys

command = ""

print("Initialized Data Creation Module. Please Input Commands <help>")

while True:
    
    command = input()

    command = "DataCapture " + command

    commandProps = command.split(" ")

    match commandProps[1]:
        case 'capture':
            match len(commandProps):
                case 2:
                    print("Please specifiy a number of samples")
                case 3:
                    try:
                        numSamples = int(commandProps[2])
                        capture_all_images(numSamples, False)
                    except:
                        print("Invalid Command Line Argument!")
                case 4:
                    try:
                        numSamples = int(commandProps[2])
                        if commandProps[3] == 'lighting':
                            capture_all_images(numSamples, True)
                        elif commandProps[3] in labels:
                            capture_images(commandProps[3], numSamples, False)
                        else:
                            print("Invalid Command Line Argument")
                    except:
                        print("Invalid Command Argument")
                case 5:
                    try:
                        numSamples = int(commandProps[2])
                        if commandProps[3] in labels and commandProps[4] == 'lighting':
                            capture_images(commandProps[3], numSamples, True)
                        else:
                            print("Invalid Command Line Argument")
                    except:
                        print("Invalid Command Line Argument")
                case _:
                    print("Too many command line Arguments!")

        case 'preprocess':
            match len(commandProps):
                case 2:
                    preprocess_all_images()
                case 3:
                    if commandProps[2] in labels:
                        preprocess_label(commandProps[2])
                    else:
                        print("Invalid Label Choice")
                case _:
                    print('Too many command line arguments')

        case 'createtest':
            match len(commandProps):
                case 2:
                    print("Specify number of samples per label")
                case 3:
                    try:
                        capture_test_data(int(commandProps[2]))
                    except:
                        print("Invalid Command Line argument")
                case _:
                    print('Too many command line arguments')

            
        case 'balance':
            if len(commandProps) < 3:
                print("Please specify maximum samples expected")
            balance_labels(int(commandProps[2]))

        case 'help':
            print('''The purpose of this file is to provide an all-in-one service for generating data using command-line arguments
                This is a list of the desired functionality from the script

                balance:
                    This argument would determine the label with the least amount of data and then randomly remove
                    data from each label until there is an even amount across the board

                capture <num_samples> <label>:
                    This argument can be used to generate num_samples samples of data for the given label

                capture <num_samples>:
                    This argument will capture data for all labels as defined in the __init__.py file of the DataCapture Module

                Adding <lighting> to the end of the above capture arguments:
                    This will cause 2 pauses in the data capture to switch lighting for more variance in data

                preprocess:
                    This argument will process the data samples under each label and return an dictionary with the number of samples per label

                preprocess <label>
            The same as above but for one label''')

        case 'exit':
            print("Terminating Program...")
            sys.exit(0)
            
        case _:
            print('Invalid Command Line Argument')