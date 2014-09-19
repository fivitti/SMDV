# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:42:14 2014

@author: HP
"""

import scipy.io


if __name__ == '__main__':
    ### Stałe programu ###
    plikMacierzy = 'Macierz_4x4.mtx'
#    plikMacierzy = 'Macierz_8x8.mtx'
#    plikMacierzy = 'Macierz_9x9.mtx'
#    plikMacierzy = 'Macierz_128x128.mtx'
#    plikMacierzy = 'Macierz_2048x2048_2.mtx'

    folderMacierzy = "../../Matrices/Generated/"
    

    ###
    macierz = scipy.io.mmread(folderMacierzy + plikMacierzy)
    print macierz

        
    

    
        
    
    
    
    