This code solves a simple optimization problem, and intended to be an learning exercise.


# HOW TO RUN THE CODE

The code "q1.py" calculates an objective function and outputs the results with the lowest values.

In "test_q1.py", the unit test routines are provided.

These codes have a few dependencies that are required: numpy, argparse, multiprocessing, and unittest

##  Running the code via "pipenv" tool, which will create a virtualenv to run the code.

First, make sure "pipenv" is installed in your system. On Ubuntu linux, this can be done via:

  $ pip install --user pipenv

On MacOS, pipenv can be installed using "homebrew" as discussed in "https://pypi.org/project/pipenv/"

Once, pipenv is installed, one can start the virtual environment shell, and run the code by doing the followings in the project folder.

```sh
  $ pipenv shell
  $ ./q1.py input.txt
```

##  Running the code via the system installation of python

Make sure the above mentioned dependencies are installed in your system. On Ubuntu Linux, one would do:

  $ sudo apt install python-numpy

Afterwards, code is run via:

  $ ./q1.py input.txt


##  Script options

To see the help page, do:

  $ ./q1.py -h

The script requires an input file. Optionally output file name can be indicated:

  $ ./q1.py --output somefile.txt input.txt

In order to run the parallel version of the calculation, use the '-p' or '--parallel' flag:

  $ ./q1.py -p input.txt


## Running unit tests

Unit tests can be run via:

  $ python test_q1.py

