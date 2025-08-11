import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import random
import math
# random data set of 50 numbers

# X_data = np.random.random(500) * 100 #the 50 means 50 numbers
# Y_data = np.random.random(500) * 100 

plt.figure()

years = [2006 + x for x in range(16) if x > 5] #for loop inline with conditionals
weights = [80 + x for x in range (10)]

x = ["c++", "c#", "python", "java", "assembly"]
y = [20, 30, 100, 10, 45]

#scatter plot
#plt.plot(X_data, Y_data, c = "#000", marker = "*") 


# len function is what it sounds like
# print (len(years) == len(weights)) 

# line plot
#plt.plot(years, weights, c ='g', lw = 3, linestyle = "--")

plt.bar(x, y)
plt.title("preferred languages")

plt.xlabel("languages")
plt.ylabel("votes")

plt.show()