import os

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

os.chdir("../dataset")
print("Current working directory: {0}".format(os.getcwd()))