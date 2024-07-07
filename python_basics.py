# Python basics 


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#type conversion
a = int(5.)
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

#contains element from i to j 
test_list[i:j] 
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#sublist from i to j with step k 
test_list[i:j:k]
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#number of elements in list
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
len(test_list)
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#return the index of x in the list
test_list.index(x)
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#counts total occurences of x 
test_list.count(x)
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#range immutable sequence of numbers from 1 to 10 step 2
test_list(range(1,10,2))
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

#documenting what a function does:

def func(x):
    '''Calcola il quadrato di x
    
    Args:
        x(float): numero
        
    Returns:
        (float): il quadrato di x
    '''
    return x**2
>>>help(func)
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


#Drawing functions 
x_coord = np.linspace(0, 2*np.pi, 10000)
y_coord = np.sin(x_coord)


#loading text

list = np.loadtxt("File_name.txt",unpack=True)

numpy.loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding=None, max_rows=None, *, quotechar=None, like=None)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

###Function pdf of a exponential###
def exp_pdf(x,tau):
    if tau ==0.: return 1.
    else: return np.exp(-1*x/tau)/tau
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Read data from a .txt file###
with open('data.txt') as o:
    sample = [float(x) for x in o.readlines()]
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# Load data from "data.txt" as float values using numpy 
data = np.genfromtxt("data.txt")
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
##Load data using numpy if data.txt are CSV format
data = np.loadtxt("data.txt", delimiter=",")