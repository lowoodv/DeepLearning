# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 07:35:37 2017

@author: ylu56
"""

import sys

from project_main import project_main
import params #params.py

def main(argv):
    project_main(params, argv)    
if __name__ == "__main__":
    main(sys.argv[1:])