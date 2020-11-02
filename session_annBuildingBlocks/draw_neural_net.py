## Gist originally developed by @craffel and improved by @ljhuang2017

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import numpy as np
import numbers


def lenMathString(s):
    ret = 0
    inMath = False
    for i in s:
        if inMath:
            if i.isspace():
                ret += 3
                inMath=False
        elif i == "\\":
           inMath = True
        elif i not in [ "_", "^" ]:
            ret += 1
    return ret

def draw_neural_net(ax,
                    left= 0.1, right= 0.9, bottom= 0.1, top= 0.9,
                    layerSizes= [2,3,1], weights = None, biases = None,
                    epoch = "", loss = "",
                    inPrefix = "x", outPrefix = "y", nodePrefix = r"z_{m}\rightarrow a_{m}",
                    hideInOutPutNodes = False, inNodePrefix = "I", outNodePrefix = "O",
                    weightPrefix = "w", biasPrefix = "b", hiddenNodePrefix = "H", 
                    showLayerIndex = True, hideBias = False, 
                    nodeColor= "lightgreen", biasNodeColor = "lightblue",
                    edgeColor = "black", biasEdgeColor = "gray",
                    weightsColor = "green", biasColor = "blue",
                    nodeFontSize = 15, edgeFontSize = 10
                    ):
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
        - layerSizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layerSizes)
    v_spacing = (top - bottom)/float(max(layerSizes))
    h_spacing = (right - left)/float(len(layerSizes) - 1)

    letterWidth= 0.004
    if "{" in nodePrefix:
        node_txt = '${}$'.format(nodePrefix.format(m=9))
    else:
        node_txt = '${}$'.format(nodePrefix)
    radius = max(h_spacing /8., (lenMathString(node_txt)+2) * letterWidth)
    
    # Input-Arrows
    layer_top_0 = v_spacing*(layerSizes[0] - 1)/2. + (top + bottom)/2.
    for m in range(layerSizes[0]):
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
    for n, layer_size in enumerate(layerSizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            x_node = n*h_spacing + left
            y_node = layer_top - m*v_spacing
            circle = mpatches.Circle((x_node, y_node), radius, #v_spacing/8.,
                                     facecolor = nodeColor, edgecolor = 'k', zorder=4)
#            circle = plt.Circle((x_node, y_node), radius, #v_spacing/8.,
#                                facecolor = nodeColor, edgecolor = 'k', zorder=0)
            if "{" in nodePrefix:
                txt = r'${}$'.format(nodePrefix.format(m=m+1))              # Change format of hidden  node text here
            else:
                txt = nodePrefix                                            # Change format of hidden  node text here
            x_label = x_node-lenMathString(txt)*letterWidth
            y_label = y_node-0.01
            layerTxt = '${}$'.format(inNodePrefix)
            if n == 0:
                plt.text(left-radius*2-0.03, #0.125,
                         y_node, r'${}_{}$'.format(inPrefix, m+1), fontsize=15)
                txt = r'$i_{}$'.format(m+1)                                  # Change format of inout  node text here
                if not hideInOutPutNodes:
                    ax.add_artist(circle)
                    x_label = x_node - lenMathString(txt) * letterWidth
                    plt.text(x_label, y_label, 
                             txt, fontsize=nodeFontSize, zorder=8, color='k')            # Change txt position here
            else:
                if n == n_layers - 1 :
                    layerTxt = r"${}$".format(outNodePrefix)   
                    plt.text(right+2*radius+0.01, y_node, r'${}_{}$'.format(outPrefix, m+1), fontsize=15)
                    #txt = r'o_{}'.format(m+1)                                  # Change format of output  node text here
                    if not hideInOutPutNodes:
                        ax.add_artist(circle)
                        #x_label = x_node-lenMathString(txt) * letterWidth
                        plt.text(x_label, y_label, 
                                 txt, fontsize=nodeFontSize, zorder=8, color='k')           # Change txt position here
                else:
                    layerTxt = r'$'+hiddenNodePrefix+'_{'+"{}".format(n)+'}$'
                    ax.add_artist(circle)
                    plt.text(x_label, y_label, 
                             txt, fontsize=15, zorder=8, color='k')               # Change txt position here
            if showLayerIndex and m == 0: 
                plt.text(x_node + 0.00, y_node + max(v_spacing/8.+0.01*v_spacing, radius+0.01),
                         layerTxt, zorder=8, fontsize=nodeFontSize)

    # Bias-Nodes
    if not hideBias:
        for n, layer_size in enumerate(layerSizes):
            skip = 1
            if hideInOutPutNodes:
                skip = 2
            if n < n_layers -skip:
                x_bias = (n+0.5)*h_spacing + left
                y_bias = top - 0.005
                circle = plt.Circle((x_bias, y_bias), radius, #v_spacing/8.,
                                    label="b",
                                    facecolor=biasNodeColor, edgecolor='k', zorder=4)
                ax.add_artist(circle)
                plt.text(x_bias-0.015, y_bias-0.01, r'$b${}'.format(n+1), fontsize=15, zorder=8, color='k') # Change format of bias text here

            
    # Edges
    # Edges between nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layerSizes[:-1], layerSizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                xm = n * h_spacing + left
                xo = (n + 1) * h_spacing + left
                ym = layer_top_a - m*v_spacing
                yo = (layer_top_b - o*v_spacing)
                delta_x = xo - xm
                delta_y = yo - ym
                length = np.sqrt(delta_x**2+delta_y**2)
                
                line1 = plt.annotate("", xy=(xo, yo), xytext=(xm, ym), xycoords = 'data',
                                     arrowprops=dict(arrowstyle=mpatches.ArrowStyle("->",head_length=2, head_width=1), shrinkB=radius *700,
                                     color=edgeColor)) #*(radius+2)/length)) #, head_length=0.8, head_width=0.8))#fc = edgeColor)
                ax.add_artist(line1)
                if weights != None:
                    rot_mo_rad = np.arctan((yo-ym)/(xo-xm))
                    rot_mo_deg = rot_mo_rad*180./np.pi
                    label = weights[n][m, o]
                    if isinstance(label, numbers.Number):
                        label = round(label,4)
                    label = "${}$".format(label)
                    letterwidth= 0.01

                    delta_x = v_spacing/8.# + radius
                    delta_y = delta_x * abs(np.tan(rot_mo_rad))
                    epsilon = 0.01/abs(np.cos(rot_mo_rad))
                    xm1 = xm + max(delta_x, radius + 0.001)
                    if yo > ym:
                        label_skew = letterwidth * abs(np.sin(rot_mo_rad))
                        ym1 = ym + label_skew + delta_y + epsilon
                    elif yo < ym:
                        label_skew = len(label)* letterwidth * abs(np.sin(rot_mo_rad))
                        ym1 = ym - label_skew - delta_y + epsilon
                    else:
                        ym1 = ym + epsilon

                    plt.text(xm1, ym1, label,
                             rotation = rot_mo_deg, 
                             fontsize = edgeFontSize, color = weightsColor)


    # Edges between bias and nodes
    if not hideBias:
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layerSizes[:-1], layerSizes[1:])):
            if hideInOutPutNodes and n == n_layers - 2:
                continue
            if n < n_layers-1:
                layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
                layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
            x_bias = (n+0.5)*h_spacing + left
            y_bias = top + 0.005 
            for o in range(layer_size_b):
                xo = left + (n + 1)*h_spacing
                yo = (layer_top_b - o*v_spacing)
                line = plt.Line2D([x_bias, xo], #(n + 1)*h_spacing + left],
                                  [y_bias, yo], #layer_top_b - o*v_spacing],
                                  c = biasEdgeColor)
                ax.add_artist(line)
                if biases != None:
                    rot_bo_rad = np.arctan((yo-y_bias)/(xo-x_bias))
                    rot_bo_deg = rot_bo_rad*180./np.pi
                    label = biases[n][o]
                    if isinstance(label, numbers.Number):
                        label = round(label,4)
                    label = "${}$".format(label)

                    letterwidth= 0.01
                    label_skew = len(label)* letterwidth * abs(np.sin(rot_bo_rad))            
                    delta_x = max(v_spacing/8., + radius+0.001)
                    delta_y = delta_x * abs(np.tan(rot_bo_rad))
                    epsilon = 0.01/abs(np.cos(rot_bo_rad))


                    xo1 = xo - delta_x #(v_spacing/8.+0.01)*np.cos(rot_bo_rad)
                    yo1 = yo -label_skew + delta_y +epsilon
                    plt.text( xo1, yo1, label,
                              rotation = rot_bo_deg, 
                              fontsize = edgeFontSize, color=biasColor)    
                
    # Output-Arrows
    layer_top_0 = v_spacing*(layerSizes[-1] - 1)/2. + (top + bottom)/2.
    for m in range(layerSizes[-1]):
        xtail = right+radius #0.015
        ytail = layer_top_0 - m*v_spacing
        dx = radius #0.2*h_spacing
        dy = 0
        xhead = xtail + dx
        yhead = ytail + dy
        arrow = mpatches.FancyArrowPatch((xtail, ytail), (xhead, yhead),
                                         mutation_scale=25)
        ax.add_patch(arrow)
    # Record the epoch and loss
    if isinstance(epoch, numbers.Number):
        round(epoch, 6)
    if isinstance(loss, numbers.Number):
        round(loss, 6)
    
    plt.text(left + (right-left)/3., bottom - 0.005*v_spacing, \
             'Steps:' + "{}".format(epoch) + '    Loss: ' + "{}".format(loss), fontsize = 15)

