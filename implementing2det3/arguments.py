# -*- coding: utf-8 -*-
#!/usr/bin/env python3 
import argparse  # Importing the argparse module for command-line argument parsing.
from argparse import ArgumentParser  # This line seems to be unnecessary, as 'argparse' was already imported.

class Arguments(object):
    def __init__(self, lines=None):
        try:
            assert(lines!=None), " arguments = Arguments({'i':{'name':'input','default':'/home/dev','help': 'This is a example!'}})arg = arguments.get()print(arg)"
            # The line above seems to contain an entire program within an assert statement, which is unusual and not recommended.

            arguments = argparse.ArgumentParser()  # Create an ArgumentParser object.
            for key, values in lines.items():  # Loop through the provided dictionary of arguments.
                arguments.add_argument("-"+key, "--"+str(values['name']),default=values['default'],help=values['help'])
                # Add arguments to the ArgumentParser based on the provided dictionary.

            self.arg = vars(arguments.parse_args())  # Parse the command-line arguments and store them in 'self.arg'.

        except ValueError as error:
            print(error)  # Print any ValueErrors that occur.

    def get(self):
        return self.arg  # Return the parsed arguments.

if __name__ == '__main__':
    # If this script is executed directly (not imported as a module), the following code will run:

    arguments = Arguments(
        {
            'i':{
                'name':'input',
                'default':'/home/dev',
                'help': 'This is a example!'
            }
        }
    )
    arg = arguments.get()  # Get the parsed arguments.
    print(arg)  # Print the parsed arguments.
