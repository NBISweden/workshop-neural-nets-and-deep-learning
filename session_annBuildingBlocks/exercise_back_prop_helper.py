# Cosmetic defs and functions, please ignore!
import sys, os
import numpy as np
import pandas as pd

# Define logistic/sigm function
def sigma(z):
  return 1/(1+np.exp(-z))



# Some styling for the table
props1 = [
    ("font-size", "18px"),
    ('max-width', '2.5cm'),
    ('text-align', 'left')
#    ('padding', '2.5cm)')
#    ("font-size", "24px"),
#    ('max-width', '2cm'),
#    ('text-align', 'center'),
#    ('padding', '5cm)')
]
styles1 = [
    dict(selector="th", props=props1),
    dict(selector="td", props=props1),
]
colNames1 = { 
        "x" : r"$x$",
        "y" : r"$y$",
        "w1" : r"$w_1$",
        "b1" : r"$b_1$",
        "i1" : r"$i_1$", 
        "z1" : r"$z_1$", 
        "a1" : r"$a_1$",
        "w2" : r"$w_2$",
        "b2" : r"$b_2$",
        "z2" : r"$z_2$",
        "a2" : r"$a_2$",
        "haty" : r"$\hat{y}$",
        "L" : "$L(w,b|x)$"
}

# Known variables
# Initialize variables to values in figure
# Code below is what is needed to update table at each step
def updateTable1(x, y, w1, b1, w2, b2, i1, z1, a1, z2, a2, haty,L):
    tab = pd.Series(
        { 
            "x" : format(5.0, ".3g"), # Use this format to display nicely in table
            "y" : format(y, ".3g"),
            "w1" : format(w1, ".3g"), 
            "b1" : format(b1, ".3g"), 
            "i1" : "" if isinstance(i1, str) else format(i1, ".3g"), 
            "z1" : "" if isinstance(z1, str) else format(z1, ".3g"), 
            "a1" : "" if isinstance(a1, str) else format(a1, ".3g"), 
            "w2" : format(w2, ".3g"), 
            "b2" : format(b2, ".3g"),        
            "z2" : "" if isinstance(z2, str) else format(z2, ".3g"),
            "a2" : "" if isinstance(a2, str) else format(a2, ".3g"),
            "haty" : "" if isinstance(haty, str) else format(haty, ".3g"),
            "L" : "" if isinstance(L, str) else format(L, ".3g").center(12, '\u00A0')
        }
    )
    return pd.DataFrame.rename(tab.to_frame().T, columns = colNames1).style.hide_index().set_table_styles(styles1)


colNames2 = { 
    "dz2dw2" : r"$\frac{\partial z_2}{\partial w_2}$",
    "da2dz2" : r"$\frac{\partial a_2}{\partial z_2}$",
    "dLda2" : r"$\frac{\partial L(b,w|x)}{\partial a_2}$",
    "dLdw2" : r"$\frac{\partial L(b,w|a)}{\partial w_2}$",
    "eta": r"$\eta$",
    "w2new": r"$w'_2$"
}



# Code below is what is needed to update table at each step
def updateTable2(dLda2,  da2dz2, dz2dw2, dLdw2, eta, w2new):
    tab = pd.Series(
        { 
            "dz2dw2" : format(dz2dw2, ".3g"), 
            "da2dz2" : format(da2dz2, ".3g"),
            "dLda2" : format(dLda2, ".3g").center(12, '\u00A0'), # Use this format to display nicely in table
            "dLdw2" : format(dLdw2, ".3g").center(12, '\u00A0'), # Use this format to display nicely in table
            "eta" : format(eta, ".3g"), 
            "w2new" : format(w2new, ".3g")
        }
    )
    return pd.DataFrame.rename(tab.to_frame().T, columns = colNames2).style.hide_index().set_table_styles(styles1)
    
props2 = [
    ("font-size", "18px"),
    ('max-width', '3cm'),
    ('text-align', 'center')
]
styles2 = [
    dict(selector="th", props = props2),
    dict(selector="td", props=props2)
]

colNames3 = { 
    "w1new" : r"$w'_1$",
    "eta" : r"$\eta$",
    "dLdw1" : r"$\frac{\partial L(w,b|x)}{\partial w_1}$",
    "dz1dw1" : r"$\frac{\partial z_1}{\partial w_1}$",
    "da1dz1" : r"$\frac{\partial a_1}{\partial z_1}$",
    "dz2da1" : r"$\frac{\partial z_2}{\partial a_1}$",
    "da2dz2" : r"$\frac{\partial a_2}{\partial z_2}$",
    "dLda2" : r"$\frac{\partial L(w,b|x)}{\partial a_2}$"
}

# Code below is what is needed to update table at each step
def updateTable3(dLda2, da2dz2, dz2da1, da1dz1, dz1dw1, dLdw1, eta, w1new):
    tab3 = pd.Series(
        { 
            "w1new" : "" if isinstance(w1new, str) else format(w1new, ".3g"),
            "eta" : "" if isinstance(eta, str) else format(eta, ".3g"),
            "dLdw1" : "" if isinstance(dLdw1, str) else format(dLdw1, ".3g"), 
            "dz1dw1" : "" if isinstance(dz1dw1, str) else format(dz1dw1, ".3g"),
            "da1dz1" : "" if isinstance(da1dz1, str) else format(da1dz1, ".3g"), 
            "dz2da1" : "" if isinstance(dz2da1, str) else format(dz2da1, ".3g"),
            "da2dz2" : "" if isinstance(da2dz2, str) else format(da2dz2, ".3g"),
            "dLda2" : "" if isinstance(dLda2, str) else format(dLda2, ".3g").center(12, '\u00A0')
        }
    )



    return pd.DataFrame.rename(tab3.to_frame().T, columns = colNames3).style.hide_index().set_table_styles(styles1)


