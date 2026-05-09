import argparse

from hdc_approximation_tuner.ApproxTypes import *



def main():
    print("Starting the HDC Approximation Tuner")

    aTy = ApproxTypes("abc", "Desc")
    print(aTy)

# This ensures the code runs when the script is executed, but not when imported
if __name__ == "__main__":
    main()
