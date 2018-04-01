#
# Script to display the histogram (assumed to be 'hist.out' in the local directory).
#
import matplotlib.pyplot as plt

# Extract the data from the file.
grey, value = [], []
for line in open("hist.out","r").readlines():
	g, v = [ int(str) for str in line.split() ]
	grey.append( g )
	value.append( v )

# Display
plt.bar( grey, value )
plt.xlabel( "Greyscale value" )
plt.ylabel( "Count" )
plt.show()
