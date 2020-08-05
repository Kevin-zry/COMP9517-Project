## s1_raw:
**numpy of raw images in sequence 1**  

## s1_pre:
**numpy of predictions(probability format) in sequence1**  

## s1_draw_res:
**numpy of results after drawing contours in sequence 1**  
It is a dictionary:  
{ 0:  
{'frame': 0,  
 'draw_img': an image array,  
 'cell_num': number of cells,  
 'box': box parameters of box boundary, (top_left_corner_x, y , width, height)  
 'centers': a list of cell centers,  
}}  
