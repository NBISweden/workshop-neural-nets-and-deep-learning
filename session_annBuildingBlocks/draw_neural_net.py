## Gist originally developed by @craffel and improved by @ljhuang2017

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import numpy as np

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, coefs_, intercepts_, n_iter_, loss_):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    radius = h_spacing /8.
    
    # Input-Arrows
    layer_top_0 = v_spacing*(layer_sizes[0] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[0]):
        xtail = left-2*radius 
        ytail = layer_top_0 - m*v_spacing
        dx= radius # h_spacing - v_spacing/8. #0.3*h_spacing
        dy = 0
        xhead = xtail + dx
        yhead = ytail + dy
        arrow = mpatches.FancyArrowPatch((xtail, ytail), (xhead, yhead),
                                         mutation_scale=25)
        ax.add_patch(arrow)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            x_node = n*h_spacing + left
            y_node = layer_top - m*v_spacing
            circle = plt.Circle((x_node, y_node), radius, #v_spacing/8.,
                                color='w', ec='k', zorder=4)
            txt = r'z{m}->a{m}'.format(m=m+1)                                  # Change format of hidden  node text here
            if n == 0:
                plt.text(left-radius*2-0.03, #0.125,
                         y_node, r'$X_{'+str(m+1)+'}$', fontsize=15)
                txt = r'   i{}  '.format(m+1)                                  # Change format of inout  node text here
            elif (n_layers == 3) & (n == 1):
                plt.text(x_node + 0.00, y_node + (v_spacing/8.+0.01*v_spacing), r'$H_{'+str(m+1)+'}$', fontsize=15)
            elif n == n_layers -1:
                plt.text(right+2*radius+0.01, #n*h_spacing + left+ 2 * radius + 0.03, #0.10,
                         y_node, r'$y_{'+str(m+1)+'}$', fontsize=15)
                #txt = r'   o{}  '.format(m+1)                                  # Change format of output  node text here
            ax.add_artist(circle)
            plt.text(x_node-0.035, y_node-0.01, txt, fontsize=15, zorder=8, color="blue") # Change position of bias text here
    # Bias-Nodes
    for n, layer_size in enumerate(layer_sizes):
        if n < n_layers -1:
            x_bias = (n+0.5)*h_spacing + left
            y_bias = top - 0.005
            circle = plt.Circle((x_bias, y_bias), radius, #v_spacing/8.,
                                label="b",
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            plt.text(x_bias-0.015, y_bias-0.01, r'$b${}'.format(n+1), fontsize=15, zorder=8, color="green") # Change format of bias text here

            
    # Edges
    # Edges between nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)
                xm = (n*h_spacing + left)
                xo = ((n + 1)*h_spacing + left)
                ym = (layer_top_a - m*v_spacing)
                yo = (layer_top_b - o*v_spacing)
                rot_mo_rad = np.arctan((yo-ym)/(xo-xm))
                rot_mo_deg = rot_mo_rad*180./np.pi
                label = str(round(coefs_[n][m, o],4)),
                letterwidth= 0.07

                delta_x = v_spacing/8.# + radius
                delta_y = delta_x*abs(np.tan(rot_mo_rad))
                epsilon = 0.01/abs(np.cos(rot_mo_rad))
                xm1 = xm + delta_x 
                #if n == 0:
                if yo > ym:
                    label_skew = letterwidth * abs(np.sin(rot_mo_rad))
                    ym1 = ym + delta_y + epsilon
                elif yo < ym:
                    label_skew = len(label)* letterwidth * abs(np.sin(rot_mo_rad))
                    ym1 = ym - label_skew - delta_y + epsilon
                else:
                    ym1 = ym + epsilon
                plt.text( xm1, ym1,
                         str(round(coefs_[n][m, o],4)),
                         rotation = rot_mo_deg, 
                         fontsize = 10)


    # Edges between bias and nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if n < n_layers-1:
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        x_bias = (n+0.5)*h_spacing + left
        y_bias = top + 0.005 
        for o in range(layer_size_b):
            line = plt.Line2D([x_bias, (n + 1)*h_spacing + left],
                          [y_bias, layer_top_b - o*v_spacing], c='k')
            ax.add_artist(line)
            xo = ((n + 1)*h_spacing + left)
            yo = (layer_top_b - o*v_spacing)
            rot_bo_rad = np.arctan((yo-y_bias)/(xo-x_bias))
            rot_bo_deg = rot_bo_rad*180./np.pi
            label = str(round(intercepts_[n][o],4)),
            
            letterwidth= 0.07
            label_skew = len(label)* letterwidth * abs(np.sin(rot_bo_rad))            
            delta_x = v_spacing/8.# + radius
            delta_y = delta_x * abs(np.tan(rot_bo_rad))
            epsilon = 0.01/abs(np.cos(rot_bo_rad))

            
            xo1 = xo - delta_x #(v_spacing/8.+0.01)*np.cos(rot_bo_rad)
            yo1 = yo -label_skew + delta_y +epsilon
            plt.text( xo1, yo1,\
                 str(round(intercepts_[n][o],4)),\
                 rotation = rot_bo_deg, \
                 fontsize = 10)    
                
    # Output-Arrows
    layer_top_0 = v_spacing*(layer_sizes[-1] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[-1]):
        xtail = right+radius #0.015
        ytail = layer_top_0 - m*v_spacing
        dx = radius #0.2*h_spacing
        dy = 0
        xhead = xtail + dx
        yhead = ytail + dy
        arrow = mpatches.FancyArrowPatch((xtail, ytail), (xhead, yhead),
                                         mutation_scale=25)
        ax.add_patch(arrow)
    # Record the n_iter_ and loss
    plt.text(left + (right-left)/3., bottom - 0.005*v_spacing, \
             'Steps:'+str(n_iter_)+'    Loss: ' + str(round(loss_, 6)), fontsize = 15)

