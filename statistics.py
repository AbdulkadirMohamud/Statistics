Python for Statistical Analysis
Statistics
Statistics is the discipline that studies the collection, organization, displaying, analysing, interpretation and presentation of data. Statistics is a branch of Mathematics that is recommended to be a prerequisite for data science and machine learning. Statistics is a very broad field but we will focus in this section only on the most relevant part. After completing this challenge, you may go onto the web development, data analysis, machine learning and data science path. Whatever path you may follow, at some point in your career you will get data which you may work on. Having some statistical knowledge will help you to make decisions based on data, data tells as they say.

Data
What is data? Data is any set of characters that is gathered and translated for some purpose, usually analysis. It can be any character, including text and numbers, pictures, sound, or video. If data is not put in a context, it doesn't make any sense to a human or computer. To make sense from data we need to work on the data using different tools.

The work flow of data analysis, data science or machine learning starts from data. Data can be provided from some data source or it can be created. There are structured and unstructured data.

Data can be found in small or big format. Most of the data types we will get have been covered in the file handling section.

Statistics Module
The Python statistics module provides functions for calculating mathematical statistics of numerical data. The module is not intended to be a competitor to third-party libraries such as NumPy, SciPy, or proprietary full-featured statistics packages aimed at professional statisticians such as Minitab, SAS and Matlab. It is aimed at the level of graphing and scientific calculators.

NumPy
In the first section we defined Python as a great general-purpose programming language on its own, but with the help of other popular libraries as(numpy, scipy, matplotlib, pandas etc) it becomes a powerful environment for scientific computing.

NumPy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with arrays.

Importing NumPy
Jupyter notebook is available if your are in favor of jupyter notebook

    # How to import numpy
    import numpy as np
    # How to check the version of the numpy package
    print('numpy:', np.__version__)
    # Checking the available methods
    print(dir(np))

    Creating numpy array using
Creating int numpy arrays
    # Creating python List
    python_list = [1,2,3,4,5]

    # Checking data types
    print('Type:', type (python_list)) # <class 'list'>
    #
    print(python_list) # [1, 2, 3, 4, 5]

    two_dimensional_list = [[0,1,2], [3,4,5], [6,7,8]]

    print(two_dimensional_list)  # [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    # Creating Numpy(Numerical Python) array from python list

    numpy_array_from_list = np.array(python_list)
    print(type (numpy_array_from_list))   # <class 'numpy.ndarray'>
    print(numpy_array_from_list) # array([1, 2, 3, 4, 5])

    Creating float numpy arrays
Creating a float numpy array from list with a float data type parameter

    # Python list
    python_list = [1,2,3,4,5]

    numy_array_from_list2 = np.array(python_list, dtype=float)
    print(numy_array_from_list2) # array([1., 2., 3., 4., 5.])

    Creating boolean numpy arrays
Creating a boolean a numpy array from list

    numpy_bool_array = np.array([0, 1, -1, 0, 0], dtype=bool)
    print(numpy_bool_array) # array([False,  True,  True, False, False])

    Creating multidimensional array using numpy
A numpy array may have one or multiple rows and columns

    two_dimensional_list = [[0,1,2], [3,4,5], [6,7,8]]
    numpy_two_dimensional_list = np.array(two_dimensional_list)
    print(type (numpy_two_dimensional_list))
    print(numpy_two_dimensional_list)

        <class 'numpy.ndarray'>
    [[0 1 2]
     [3 4 5]
     [6 7 8]]

     Converting numpy array to list
# We can always convert an array back to a python list using tolist().
np_to_list = numpy_array_from_list.tolist()
print(type (np_to_list))
print('one dimensional array:', np_to_list)
print('two dimensional array: ', numpy_two_dimensional_list.tolist())

Creating numpy array from tuple
# Numpy array from tuple
# Creating tuple in Python
python_tuple = (1,2,3,4,5)
print(type (python_tuple)) # <class 'tuple'>
print('python_tuple: ', python_tuple) # python_tuple:  (1, 2, 3, 4, 5)

numpy_array_from_tuple = np.array(python_tuple)
print(type (numpy_array_from_tuple)) # <class 'numpy.ndarray'>
print('numpy_array_from_tuple: ', numpy_array_from_tuple) # numpy_array_from_tuple:  [1 2 3 4 5]
Shape of numpy array
The shape method provide the shape of the array as a tuple. The first is the row and the second is the column. If the array is just one dimensional it returns the size of the array.

    nums = np.array([1, 2, 3, 4, 5])
    print(nums)
    print('shape of nums: ', nums.shape)
    print(numpy_two_dimensional_list)
    print('shape of numpy_two_dimensional_list: ', numpy_two_dimensional_list.shape)
    three_by_four_array = np.array([[0, 1, 2, 3],
        [4,5,6,7],
        [8,9,10, 11]])
    print(three_by_four_array.shape)
    [1 2 3 4 5]

    Data type of numpy array
Type of data types: str, int, float, complex, bool, list, None

int_lists = [-3, -2, -1, 0, 1, 2,3]
int_array = np.array(int_lists)
float_array = np.array(int_lists, dtype=float)

print(int_array)
print(int_array.dtype)
print(float_array)
print(float_array.dtype)

Mathematical Operation using numpy
NumPy array is not like exactly like python list. To do mathematical operation in Python list we have to loop through the items but numpy can allow to do any mathematical operation without looping. Mathematical Operation:

Addition (+)
Subtraction (-)
Multiplication (*)
Division (/)
Modules (%)
Floor Division(//)
Exponential(**)

NumPy Matrix Multiplication with np.matmul()
### Matmul: matruc product of two arrays
h = [[1,2],[3,4]]
i = [[5,6],[7,8]]
### 1*5+2*7 = 19
np.matmul(h, i)
    array([[19, 22],
           [43, 50]])
## Determinant 2*2 matrix
### 5*8-7*6np.linalg.det(i)
np.linalg.det(i)
-1.999999999999999
Z = np.zeros((8,8))
Z[1::2,::2] = 1
Z[::2,1::2] = 1
Z
array([[0., 1., 0., 1., 0., 1., 0., 1.],
       [1., 0., 1., 0., 1., 0., 1., 0.],
       [0., 1., 0., 1., 0., 1., 0., 1.],
       [1., 0., 1., 0., 1., 0., 1., 0.],
       [0., 1., 0., 1., 0., 1., 0., 1.],
       [1., 0., 1., 0., 1., 0., 1., 0.],
       [0., 1., 0., 1., 0., 1., 0., 1.],
       [1., 0., 1., 0., 1., 0., 1., 0.]])
new_list = [ x + 2 for x in range(0, 11)]
new_list
[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
np_arr = np.array(range(0, 11))
np_arr + 2
array([ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

We use linear equation for quantities which have linear relationship. Let's see the example below:

temp = np.array([1,2,3,4,5])
pressure = temp * 2 + 5
pressure
array([ 7, 9, 11, 13, 15])

plt.plot(temp,pressure)
plt.xlabel('Temperature in oC')
plt.ylabel('Pressure in atm')
plt.title('Temperature vs Pressure')
plt.xticks(np.arange(0, 6, step=0.5))
plt.show()
png

To draw the Gaussian normal distribution using numpy. As you can see below, the numpy can generate random numbers. To create random sample, we need the mean(mu), sigma(standard deviation), mumber of data points.

mu = 28
sigma = 15
samples = 100000

x = np.random.normal(mu, sigma, samples)
ax = sns.distplot(x);
ax.set(xlabel="x", ylabel='y')
plt.show()
png