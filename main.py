from cv2 import THRESH_BINARY
import numpy as np
import cv2
import random
import gc

usebackgroundcolor = 0

#array of 256 brightness values to ascii characters
chararray=[' ',' ',' ','.','.','.','.','\'','\'','\'','`','`','`','`','^','^','^','“','“','“','“',',',',',',',':',':',':',':',';',';',';','I','I','I','I','l','l','l','!','!','!','!','I','I','I','>','>','>','>','<','<','<','~','~','~','~','+','+','+','_','_','_','_','-','-','-','?','?','?','?',']',']',']','[','[','[','[','{','{','{','}','}','}','}','1','1','1',')',')',')',')','(','(','(','|','|','|','/','/','/','\\','\\','\\','\\','t','t','t','f','f','f','f','j','j','j','r','r','r','r','x','x','x','n','n','n','n','u','u','u','v','v','v','v','c','c','c','z','z','z','z','X','X','X','Y','Y','Y','Y','U','U','U','J','J','J','J','C','C','C','L','L','L','L','Q','Q','Q','0','0','0','0','O','O','O','Z','Z','Z','Z','m','m','m','w','w','w','w','q','q','q','p','p','p','p','d','d','d','d','b','b','b','k','k','k','k','h','h','h','a','a','a','a','o','o','o','*','*','*','*','#','#','#','M','M','M','M','W','W','W','&','&','&','&','8','8','8','8','%','%','%','B','B','B','B','@','@','@','$','$','$','$','$','$','$','$','$','$','$','$','$','$','$'
]
#array of 16 characters
contourarray=[' ','.',',','=','\'',']','/','d','`','\\','[','b','"','Y','P','8']

#array of hues to ansi escape codes
fghuetocolorarray=['\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[93m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[92m','\u001b[96m','\u001b[96m','\u001b[96m','\u001b[96m','\u001b[96m','\u001b[96m','\u001b[96m','\u001b[96m','\u001b[96m','\u001b[96m','\u001b[96m','\u001b[96m','\u001b[96m','\u001b[96m','\u001b[96m','\u001b[96m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[94m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[95m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m','\u001b[91m'
]
bghuetocolorarray=['\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[43m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[42m','\u001b[46m','\u001b[46m','\u001b[46m','\u001b[46m','\u001b[46m','\u001b[46m','\u001b[46m','\u001b[46m','\u001b[46m','\u001b[46m','\u001b[46m','\u001b[46m','\u001b[46m','\u001b[46m','\u001b[46m','\u001b[46m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[44m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[45m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m','\u001b[41m'
]

#array of values to ansi escape codes
fgvaltocolorarray=['\u001b[30m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[90m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[37m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m','\u001b[97m'
]
bgvaltocolorarray=['\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[40m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[100m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m','\u001b[47m'
]


#take an image, downscale it, greyscale it, then print that to the terminal 
img = cv2.imread('image.png', 1)
cv2.imshow('original',img)

img_resize = cv2.resize(img, (80,25))
img_termsize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
cv2.imshow('resised', img_resize)
cv2.imshow('grey', img_termsize)

text_valuemap = ''
for x in img_termsize:
    linetext = ''
    for y in x:
       linetext = linetext + chararray[y]
    text_valuemap = text_valuemap + linetext


#get hue and saturation of image
img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
img_huemap = cv2.extractChannel(img_resize,0)
img_satmap = cv2.extractChannel(img_resize,1)
cv2.imshow('hue', img_huemap)
cv2.imshow('saturation', img_satmap)
cv2.imshow('resised_hsv', img_resize)






#generate a laplacian transform for the input image
#img_contrastmap = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img_contrastmap = cv2.Laplacian(img_contrastmap, cv2.CV_8U)
#(t,img_contrastmap) = cv2.threshold(img_contrastmap, 15,255, cv2.THRESH_BINARY)
#img_contrastmap = cv2.resize(img_contrastmap, (80,25))
#cv2.imshow('contrast map',img_contrastmap)


#generate a 1 bit image and place characters based on groups of 4 adjacent pixels
img_doublesize = cv2.resize(img, (160,50))
img_doublesize = cv2.cvtColor(img_doublesize, cv2.COLOR_BGR2GRAY)
img_doublesize = cv2.adaptiveThreshold(img_doublesize,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('contrast map resize',img_doublesize)

#take the 4 pixels to the right, bottom, and bottom right and print a character based on that
text_contourmap = ''
x_index = 0
for x in img_doublesize:
    linetext = ''
    y_index = 0
    x_index = x_index + 1

    if (x_index % 2) != 0:
        for y in x:
            y_index = y_index + 1
            if (y_index % 2) != 0:
                z = 0
                #z = str(img_doublesize[x_index,y_index])
                if str(img_doublesize[x_index-1,y_index-1]) != '0':#it breaks if i dont do this and i dont know why
                    z = z+1
                if str(img_doublesize[x_index,y_index-1]) != '0':
                    z = z+2
                if str(img_doublesize[x_index-1,y_index]) != '0':
                    z = z+4
                if str(img_doublesize[x_index,y_index]) != '0':
                    z = z+8
                linetext = linetext + contourarray[z]
            else:
                continue

            
        text_contourmap = text_contourmap + linetext
    else:
        continue
    
del x_index
gc.collect()



#generate contrast map based on contour
img_contrastmap = cv2.Laplacian(img_termsize, cv2.CV_8U)
img_contrastmap = cv2.resize(img_contrastmap, (80,25))
(t,img_contrastmap) = cv2.threshold(img_contrastmap, 30,255, cv2.THRESH_BINARY)
cv2.imshow('contrast map',img_contrastmap)

#print to terminal based on contrast map
z_index = 0
x_index = 0
for x in img_contrastmap:
    linetext = ''
    x_index = x_index + 1
    y_index = 0
    for y in x:
        y_index = y_index + 1

        #background color
        if (usebackgroundcolor == 1):
            if img_satmap[x_index-1,y_index-1] >= 60:
                #use hue
                linetext = linetext + bghuetocolorarray[(img_huemap[x_index-1,y_index-1])]
            else:
                #use val
                linetext = linetext + bgvaltocolorarray[(img_termsize[x_index-1,y_index-1])]

        #forground color
        if img_satmap[x_index-1,y_index-1] >= 40:
            #use hue
            linetext = linetext + fghuetocolorarray[(img_huemap[x_index-1,y_index-1])]
        else:
            #use val
            linetext = linetext + fgvaltocolorarray[(img_termsize[x_index-1,y_index-1])]

        #value
        if y == 255:
           linetext = linetext + str(text_contourmap[z_index])
        else:
            linetext = linetext + str(text_valuemap[z_index])
        z_index = z_index + 1


    print(linetext + '\u001b[0m')

cv2.waitKey(0)
cv2.destroyAllWindows()
