# tubefilter
Image analysis for filamentous networks
Analysis written for Jupyter Notebook 6.1.4

Start the tube_filter.ipynb via the Jupyter Notebook, tube_filter.py has to be placed in the same folder.

1. set folder path (r"XXX")
2. set basefilename (r'\XXX')
3. define varaiable parameters for CLAHE and the Gaussian filter

	#CLAHE
		ntiles_x = XXX (eg 32)
		ntiles_y = XXX (eg 32)
		cliplimit=XXX (eg 0.01)

	#Gaussian filter

		Sigma=XXX (eg 0.2)

4. run skript

5. Analysis saves a skeleton.png and a node.txt (containing x an y coordinates of the respective nodes) 
