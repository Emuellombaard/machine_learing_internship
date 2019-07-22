from datetime import datetime
import os

def compare_loss(quantlearn, unquantlearn):
    unquant = unquantlearn.validate()[0]
    quant = quantlearn.validate()[0]
    inc = ((quant-unquant)/unquant)*100
    print("Unquantized loss:\t{:.4f}\nQuantized Loss:\t\t{:.4f}\nPercentage Increase:\t{:.4f}%".format(unquant,quant,inc))

def print_loss(uncomplearn, complearn):
    uncomp = uncomplearn.validate()[0]
    comp = uncomplearn.validate()[0]
    print("Original model loss:\t{:.4f}\nLoaded model loss:\t{:.4f}\n".format(uncomp,comp))
    
def compare_size(filepath1, filepath2):
    file1 = os.path.getsize(filepath1)/1024/1024
    file2 = os.path.getsize(filepath2)/1024/1024
    difference = (file2 / file1) * 100
    print("Unqauntized Model Size:\t{:.0f}Mb\nQuantized Model Size:\t{:.2f}Mb\nQuantized size is {:.2f}% of the unquantized size".format(file1,file2,difference))