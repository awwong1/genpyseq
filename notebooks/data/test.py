"""Predict Test"""
import sys
from os import getcwd

def main():
    sys.stdout.write(getcwd())
    for i in range(0, 10):
        print("{} : Boop".format(i), i)
    return False

if __name__ == "__main__":
    main()
