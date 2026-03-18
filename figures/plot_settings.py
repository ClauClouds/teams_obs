import cmcrameri.cm as cm

# define colormap global variable
CMAP = cm.batlow_r


# var dict for plotting
VAR_DICT = {
    "lwp": {
        "label": "LWP [g/m$^2$]",
        "units": "g/m$^2$",
        "cmap": CMAP,
        "vmin": 0,
        "vmax": 500,
    },
    "iwv": {
        "label": "IWV [kg/m$^2$]",  
        "units": "kg/m$^2$",
        "cmap": CMAP,
        "vmin": 0,
        "vmax": 50,
    }
}   