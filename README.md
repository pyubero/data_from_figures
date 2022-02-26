# data from figures
 This repo contains some Python scripts that I have been using to reverse-engineer figures (mainly from papers, or datasheets) to extract quantitatively their data. So far they only work for a selected number of figure types: line plots, scatter, bargraphs and heatmaps. Fortunately, these cover the vast majority of figures found around. The data is exported in a .csv file.


# Use
1 Their usage is fairly simple. From the console execute any file as `python export_scatter.py -i input_filepath -o output_filepath`. And perhaps some of the optional parameters:

    a. `-sz VALUE` to resize the image previews. Value can be any decimal number. Default = 1
    b. `-th VALUE` to change the color similarity threshold that recognizes poits/lines. Value can be any number in [0,1]. Default = 0.95
	c. `-p  VALUE` to specify the output precision in numbers of decimals. Value can be any integer >1. Default=2  

2 The image will pop up and you need to specify the axis limits by click-and-dragging the bounding box.

3 Follow the questions asked in the terminal.

4 Enjoy your results in the csv file!

## Example results of line plots
![](https://github.com/pyubero/data_from_figures/blob/main/results_plot.png "Results for line plots.")

<img src="https://github.com/pyubero/data_from_figures/blob/main/results_plot.png" width="200"  />