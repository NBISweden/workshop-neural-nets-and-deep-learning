 ## Gist originally developed by @craffel and improved by @ljhuang2017 and @bsennblad

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import numpy as np
import numbers
import re


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
        elif i not in [ "_", "^", "$", "{", "}", "," ]:
            ret += 1
    return ret

def draw_neural_net(ax,
                    left= 0.0, right= 1., bottom= 0.0, top= 1.,
                    layerSizes= [2,3,1],
                    inputPrefix = "x", outputPrefix = "\hat{y}_{m}", 
                    inLayerPrefix = "I", outLayerPrefix = "O", hiddenLayerPrefix = "H", 
                    inNodePrefix = "i", otherNodePrefix = r"z_{m}\rightarrow a_{m}",
                    biasNodePrefix = r"b_{m}", 
                    weights = None, biases = None,
                    epoch = "", loss = "",
                    hideInOutPutNodes = False, hideBias = False, showLayerIndex = True, 
                    inputOutputColor = "blue",
                    nodeColor= "lightgreen", biasNodeColor = "lightcyan",
                    edgeColor = "black", biasEdgeColor = "gray",
                    weightsColor = "green", biasColor = "purple",
                    nodeFontSize = 15, edgeFontSize = 10, edgeWidth = 1
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
        - inputPrefix : string
            Prefix of input; a prefix p will show as $p_i$ where i is an index
        - outputPrefix : string
            Prefix of output; a prefix p will show as $p_i$ where i is an index
        - inLayerPrefix : string
            string used to denote input layer
        - outLayerPrefix : string
            string used to denote output layer
        - hiddenLayerPrefix : string
            string used to denote hidden layers
        - inNodePrefix : string
            string used for text in input nodes.; a prefix p will show as $p_i$ where i is an index
        - otherNodePrefix : string or list of list of strings
            Prefix used for text in all nodes but the input and bias nodes.
            If string, this will be reused for all nodes; to get automatic indexing 
            include "_{m}" (must use 'm').
            If list of list, then outer list must conform to number of layers (excepting 
            input layer) and inner lists to the number of nodes in the respective layers;
            NB! use raw strings when including latex math notation
        - biasNodePrefix : string
            string used for text in bias nodes.; to get automatic indexing include "_{m}" 
            (must use 'm'). NB! use raw strings when including latex math notation
        - weights : None or list of numpy.array of strings or floats
            If list of list, then outer list must conform to number of layers (excepting 
            input layer) and inner numpy.array must take an indexing 'from node', 'to node',
            denoting the weight for that edge; strings may include latex math notation; 
            numbers will be rounded to 4 decimals.
        - bias : None or list of lists of strings or floats
            If list of list, then outer list must conform to number of layers (excepting 
            input layer) and inner lists to the number of nodes in the respective layers.
        - epoch : int
            The epoch number
        - loss : float
            The value of the Loss/Cost function
        - hideInOutPutNodes : True/False
            Hide inout and output nodes (e.g., when drawing only a nenuron
        - hideBias : True/False
            Hide bias nodes and edges
        - showLayerIndex = True/False
            Whether to show layer names 
        - inputOutputColor : valid MatPlotLib color name
            Color of input output arrows
        - nodeColor : valid MatPlotLib color name
            Background color of layer nodes
        - biasNodeColor : valid MatPlotLib color name
            Background color of bias nodes
        - edgeColor : valid MatPlotLib color name
            Color of weight text
        - biasEdgeColor : valid MatPlotLib color name
            Color of bias text
        - nodeFontSize : int
            Fontsize text inside nodes
        - edgeFontSize : int
            Fontsize of edge text
        - edgeWidth : int
            Width of edge lines

    '''
    n_layers = len(layerSizes)
    vSpacing = (top - bottom)/float(max(layerSizes))
    hSpacing = (right - left)/float(len(layerSizes) - 1)
    
    input = inputPrefix
    if not isinstance(input, list): 
        input = [ r'${}_{}$'.format(inputPrefix, m+1) for m in range(layerSizes[0]) ]
    
    output = outputPrefix
    if not isinstance(output, list): 
        output = [ r'${}$'.format(re.sub("m", "{}".format(m+1), outputPrefix)) for m in range(layerSizes[-1]) ]

    hidden = otherNodePrefix
    if not isinstance(hidden, list): 
        hidden = [ [ r'${}$'.format(otherNodePrefix.format(m=m+1)) if "{" in otherNodePrefix else  otherNodePrefix for m in list(range(layerSizes[n])) ] for n in list(range(len(layerSizes))) ]
    
    nodeLetterWidth= 0.0007 * nodeFontSize
    edgeLetterWidth= 0.0007 * edgeFontSize
    nodeRadius =  max(hSpacing /8.,
                      (max([ lenMathString(max(x,key=lenMathString))for x in hidden ])+1)*nodeLetterWidth/2)
    #nodeRadius = max(hSpacing /8., (lenMathString(node_txt))/2 * nodeLetterWidth)
    biasRadius = max(hSpacing /12., (lenMathString(biasNodePrefix)+1)/2 * nodeLetterWidth)

    nodePlusArrow = 2*nodeRadius
    if hideInOutPutNodes:
        nodePlusArrow = 0
    # adjust left, right, bottom, top and spacing to fit input text
    inPad = (lenMathString(max(input,key=lenMathString))+2)*nodeLetterWidth
    left = max(nodePlusArrow+inPad, left)
    
    outPad = (lenMathString(max(output,key=lenMathString))+2)*nodeLetterWidth
    right = min(1.0-nodePlusArrow-outPad, right)

    bottom = max(nodeRadius, bottom)
    top = min(1.0-nodeRadius, top)
    
    vSpacing = (top - bottom)/float(max(layerSizes))
    hSpacing = (right - left)/float(len(layerSizes) - 1)

 
    # Input-Arrows
    if not hideInOutPutNodes:
        layer_top_0 = vSpacing*(layerSizes[0] - 1)/2. + (top + bottom)/2.
        for m in range(layerSizes[0]):
            xhead = left-nodeRadius 
            yhead = layer_top_0 - m*vSpacing
            dx= nodeRadius # hSpacing - vSpacing/8. #0.3*hSpacing
            dy = 0 #2*nodeLetterWidth
            xtail = xhead - dx
            ytail = yhead - dy
            # arrow = mpatches.FancyArrowPatch((xtail, ytail), (xhead, yhead),
            #                                  mutation_scale=25, zorder = 10)
            # ax.add_patch(arrow)
            line1 = plt.annotate("", xy=(xhead, yhead), xytext=(xtail, ytail), xycoords = 'data',
                                     arrowprops=dict(arrowstyle=mpatches.ArrowStyle("simple",
                                                                                    head_length=0.4,
                                                                                    head_width=0.4
                                                                                    ),
                                                     color=inputOutputColor, lw=edgeWidth)) 
            ax.add_artist(line1)
    # Nodes
    for n, layer_size in enumerate(layerSizes):
        layer_top = vSpacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            x_node = n*hSpacing + left
            y_node = layer_top - m*vSpacing
            circle = mpatches.Circle((x_node, y_node), nodeRadius, #vSpacing/8.,
                                     facecolor = nodeColor, edgecolor = 'k', zorder=4)
#            circle = plt.Circle((x_node, y_node), nodeRadius, #vSpacing/8.,
#                                facecolor = nodeColor, edgecolor = 'k', zorder=0)
            txt = hidden[n][m]
            x_label = x_node-lenMathString(txt)*nodeLetterWidth/2
            y_label = y_node-0.01
            layerTxt = ""
            inputOutputPad = nodeRadius * 2 
            if hideInOutPutNodes:
                inputOutputPad = 0
            
            if n == 0:
                if inLayerPrefix != "":
                    layerTxt = '${}$'.format(inLayerPrefix)
                plt.text(left - inputOutputPad - inPad, #left-nodeRadius*2-0.03, #0.125,
                         y_node - nodeLetterWidth,
                         input[m], #r'${}_{}$'.format(inputPrefix, m+1),
                         fontsize=nodeFontSize, zorder=2)
                txt = r'${}_{}$'.format(inNodePrefix, m+1) if inNodePrefix != "" else inNodePrefix
                if not hideInOutPutNodes:
                    ax.add_artist(circle)
                    x_label = x_node - lenMathString(txt) * nodeLetterWidth/2
                    plt.text(x_label, y_label, 
                             txt, fontsize=nodeFontSize, zorder=8, color='k')# Change txt position here
            else:
                if n == n_layers - 1 :
                    if outLayerPrefix != "":
                        layerTxt = r"${}$".format(outLayerPrefix)
                    plt.text(right + inputOutputPad, # +outPad/2, #+ 0.01, #right+2*nodeRadius+0.01,
                             y_node - nodeLetterWidth,
                             output[m], #r'${}_{}$'.format(outputPrefix, m+1),
                             fontsize=nodeFontSize)
                    #txt = r'o_{}'.format(m+1)                                  # Change format of output  node text here
                    if not hideInOutPutNodes:
                        ax.add_artist(circle)
                        #x_label = x_node-lenMathString(txt) * nodeLetterWidth
                        plt.text(x_label, y_label, 
                                 txt, fontsize=nodeFontSize, zorder=8, color='k') # Change txt position here
                else:
                    if hiddenLayerPrefix != "":
                        layerTxt = r'$'+hiddenLayerPrefix+'_{'+"{}".format(n)+'}$'
                    ax.add_artist(circle)
                    plt.text(x_label, y_label, 
                             txt, fontsize=nodeFontSize, zorder=8, color='k')      # Change txt position here
            if showLayerIndex and m == 0: 
                plt.text(x_node + 0.00, y_node + max(vSpacing/8.+0.01*vSpacing, nodeRadius+0.01),
                         layerTxt, zorder=8, fontsize=nodeFontSize)

    # Bias-Nodes
    if not hideBias:
        for n, layer_size in enumerate(layerSizes):
            skip = 1
            if hideInOutPutNodes:
                skip = 2
            if n < n_layers -skip:
                x_bias = (n+0.5)*hSpacing + left
                y_bias = top - 0.005
                circle = plt.Circle((x_bias, y_bias), biasRadius, #vSpacing/8.,
#                                    label="b",
                                    facecolor=biasNodeColor, edgecolor='k', zorder=4)
                ax.add_artist(circle)
                txt = biasNodePrefix
                if "{" in biasNodePrefix:
                    txt = r'${}$'.format(biasNodePrefix.format(m=m+1))              # Change format of hidden  node text here
                r'$b${}'.format(n+1)
                plt.text(x_bias-0.015, y_bias-0.01, txt, fontsize=nodeFontSize, zorder=8, color='k') # Change format of bias text here

            
    # Edges
    # Edges between nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layerSizes[:-1], layerSizes[1:])):
        layer_top_a = vSpacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = vSpacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                xm = n * hSpacing + left
                xo = (n + 1) * hSpacing + left
                ym = layer_top_a - m*vSpacing
                yo = (layer_top_b - o*vSpacing)
                delta_x = xo - xm
                delta_y = yo - ym
                length = np.sqrt(delta_x**2+delta_y**2)
                
                line1 = plt.annotate("", xy=(xo, yo), xytext=(xm, ym), xycoords = 'data',
                                     arrowprops=dict(arrowstyle=mpatches.ArrowStyle("->",
                                                                                    head_length=10*min(0.2, hSpacing/5.),
                                                                                    head_width=10*min(0.1, hSpacing/5.)),
                                                     shrinkB=nodeRadius *700,
                                                     color=edgeColor, lw=edgeWidth))
                ax.add_artist(line1)
                if weights != None:
                    rot_mo_rad = np.arctan((yo-ym)/(xo-xm))
                    rot_mo_deg = rot_mo_rad*180./np.pi
                    label = weights[n][m, o]
                    if label != "":
                        if isinstance(label, numbers.Number):
                            label = round(label,4)
                        label = "${}$".format(label)

                    delta_x = vSpacing/8.# + nodeRadius
                    delta_x = max(delta_x, nodeRadius + 0.001)
                    delta_y = delta_x * abs(np.tan(rot_mo_rad))
                    epsilon = 0.01/abs(np.cos(rot_mo_rad))
                    xm1 = xm + delta_x
                    if yo > ym:
                        label_skew = edgeLetterWidth * abs(np.sin(rot_mo_rad))
                        ym1 = ym + label_skew + delta_y + epsilon 
                    elif yo < ym:
                        label_skew = lenMathString(label)* edgeLetterWidth * abs(np.sin(rot_mo_rad))
                        ym1 = ym - label_skew - delta_y + epsilon
                    else:
                        ym1 = ym + epsilon

                    plt.text(xm1, ym1, label,
                             rotation = rot_mo_deg, 
                             fontsize = edgeFontSize, color = weightsColor, zorder=10)


    # Edges between bias and nodes
    if not hideBias:
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layerSizes[:-1], layerSizes[1:])):
            if hideInOutPutNodes and n == n_layers - 2:
                continue
            if n < n_layers-1:
                layer_top_a = vSpacing*(layer_size_a - 1)/2. + (top + bottom)/2.
                layer_top_b = vSpacing*(layer_size_b - 1)/2. + (top + bottom)/2.
            x_bias = (n+0.5)*hSpacing + left
            y_bias = top + 0.005 
            for o in range(layer_size_b):
                xo = left + (n + 1)*hSpacing
                yo = (layer_top_b - o*vSpacing)
                line = plt.Line2D([x_bias, xo], #(n + 1)*hSpacing + left],
                                  [y_bias, yo], #layer_top_b - o*vSpacing],
                                  c = biasEdgeColor, lw=edgeWidth)
                ax.add_artist(line)
                if biases != None:
                    rot_bo_rad = np.arctan((yo-y_bias)/(xo-x_bias))
                    rot_bo_deg = rot_bo_rad*180./np.pi
                    label = biases[n][o]
                    if isinstance(label, numbers.Number):
                        label = round(label,4)
                    label = "${}$".format(label)


                    label_skew = len(label)* edgeLetterWidth * abs(np.sin(rot_bo_rad))            
                    delta_x = max(vSpacing/8., + nodeRadius+0.001)
                    delta_y = delta_x * abs(np.tan(rot_bo_rad))
                    epsilon = 0.01/abs(np.cos(rot_bo_rad))


                    xo1 = xo - delta_x #(vSpacing/8.+0.01)*np.cos(rot_bo_rad)
                    yo1 = yo -label_skew + delta_y +epsilon
                    plt.text( xo1, yo1, label,
                              rotation = rot_bo_deg, 
                              fontsize = edgeFontSize, color=biasColor)    
                
    # Output-Arrows
    if not hideInOutPutNodes:
        layer_top_0 = vSpacing*(layerSizes[-1] - 1)/2. + (top + bottom)/2.
        for m in range(layerSizes[-1]):
            xtail = right+nodeRadius #0.015
            ytail = layer_top_0 - m*vSpacing
            dx = nodeRadius #0.2*hSpacing
            dy = 0 #-2*nodeLetterWidth
            xhead = xtail + dx
            yhead = ytail + dy
            # arrow = mpatches.FancyArrowPatch((xtail, ytail), (xhead, yhead),
            #                                  mutation_scale=25, zorder=8)
            # ax.add_patch(arrow)
            line1 = plt.annotate("", fontsize=nodeFontSize, xy=(xhead, yhead), xytext=(xtail, ytail), xycoords = 'data',
                                     arrowprops=dict(arrowstyle=mpatches.ArrowStyle("simple",
                                                                                    head_length=0.4,
                                                                                    head_width=0.4
                                                                                    ),
                                                     color=inputOutputColor, lw=edgeWidth), zorder=0) 
            ax.add_artist(line1)
        
    # Record the epoch and loss
    if isinstance(epoch, numbers.Number):
        round(epoch, 6)
    if isinstance(loss, numbers.Number):
        round(loss, 6)
    txt = ""
    if epoch != "":
        txt = "Steps: {}".format(epoch)
    if loss != "":
        txt = "{}    Loss: {}".format(txt,loss)
    plt.text(left + (right-left)/3., bottom - 0.005*vSpacing, \
             txt,
             #'Steps:' + "{}".format(epoch) + '    Loss: ' + "{}".format(loss),
             fontsize = nodeFontSize)

