# Comp9517_project

## Task 1: Detect and Track Cells
Develop a Python program to detect and track all the cells within the image sequences. This means the program needs to perform the following steps:  

1-1. Detect all the cells and draw a bounding box around each of them. For each image in a
sequence the program should show the bounding boxes for that image only.  
1-2. Draw the trajectory (also called track or path) of each cell. For each image in a sequence
the program should show for each cell the past trajectory up to that time point.  
1-3. Print (either as an output to the terminal or directly in the image window) the real-time count of the cells detected in each image of the sequence.

## Task 2: Detect Cell Divisions
Extend the program so that it can detect cell division (also called mitosis) events. For each dividing cell, the process of splitting of the mother cell into two daughter cells may take multiple time points to complete. The program should output the following:

2-1. Change the colour or shape (your choice) of the cell’s bounding box for those time
points during which the cell is in the process of dividing. After the division is complete,
the program should track the two daughter cells as new cells.  
2-2. Print (either as an output to the terminal or directly in the image window) the real-time
count of the cells that are dividing at each time point.

## Task 3: Analyse Cell Motion
Further extend the program so that it can analyse the motion of a selected cell. At any time point, the user should be able to select a cell, and the program should output the following (either to the terminal or directly in the image window):

3-1. Speed of the cell at that time point. This can be estimated by taking the Euclidean
distance (in pixels) between the coordinates of the cell’s bounding box center in the current time point and the previous time point, divided by the time difference (the latter is simply 1 frame, so the unit of speed is pixels/frame). Notice this means for the first time point of a cell’s trajectory, no speed estimate can be computed.  
3-2. Total distance travelled up to that time point. This is the sum of the Euclidean distances (in pixels) computed from the first time point of a cell’s trajectory to the second, from the second to the third, and so on, until the current time point.  
3-3. Net distance travelled up to that time point. This is the Euclidean distance (in pixels) directly between the cell’s coordinates in the current time point and its coordinates in the first time point of its trajectory.  
3-4. Confinement ratio of the cell motion. This is the ratio between the total distance
travelled by the cell up to the current time point (computed in 3-2) and the net distance travelled up to the current time point (computed in 3-3).
