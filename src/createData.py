from DataCapture.capture import capture_images, capture_all_images
from DataCapture.preprocess import preprocess_all_images, preprocess_label
from DataCapture.balance import balance_labels
from DataCapture import labels
import sys

'''

The purpose of this file is to provide an all-in-one service for generating data using command-line arguments
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
    The same as above but for one label


'''

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} [capture, preprocess, balance] [label? | num_samples?] [num_samples? | lighting?] [lighting?]")
    sys.exit(1)

match sys.argv[1]:
    case 'capture':
        match len(sys.argv):
            case 2:
                print("Please specifiy a number of samples")
                sys.exit(1)
            case 3:
                try:
                    numSamples = int(sys.argv[2])
                    capture_all_images(numSamples, False)
                except:
                    print("Invalid Command Line Argument!")
                    sys.exit(1)
            case 4:
                try:
                    numSamples = int(sys.argv[2])
                    if sys.argv[3] == 'lighting':
                        capture_all_images(numSamples, True)
                    elif sys.argv[3] in labels:
                        capture_images(sys.argv[3], numSamples, False)
                    else:
                        print("Invalid Command Line Argument")
                        sys.exit(1)
                except:
                    print("Invalid Command Argument")
            case 5:
                try:
                    numSamples = int(sys.argv[2])
                    if sys.argv[3] in labels and sys.argv[4] == 'lighting':
                        capture_images(sys.argv[3], numSamples, True)
                    else:
                        print("Invalid Command Line Argument")
                        sys.exit(1)   
                except:
                    print("Invalid Command Line Argument")
                    sys.exit(1)
            case _:
                print("Too many command line Arguments!")
                sys.exit(1)

    case 'preprocess':
        match len(sys.argv):
            case 2:
                preprocess_all_images()
            case 3:
                if sys.argv[2] in labels:
                    preprocess_label(sys.argv[2])
                else:
                    print("Invalid Label Choice")
                    sys.exit(1)
            case _:
                print('Too many command line arguments')
        
    case 'balance':
        if len(sys.argv) < 3:
            print("Please specify maximum samples expected")
            sys.exit(1)
        balance_labels(int(sys.argv[2]))
        
    case _:
        print('Invalid Command Line Argument')
        sys.exit(1)