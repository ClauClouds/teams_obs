import cmcrameri.cm as cm

# define colormap global variable
CMAP1 = cm.batlow_r
CMAP2 = cm.imola
CMAP3= cm.vik

# var dict for plotting
VAR_DICT = {
    "lwp": {
        "label": "LWP [g/m$^2$]",
        "units": "g/m$^2$",
        "cmap": CMAP1,
        "vmin": 0,
        "vmax": 500,
        "color_step": 2.0,
        "tick_step": 20.0,
    },
    "iwv": {
        "label": "IWV [kg/m$^2$]",  
        "units": "kg/m$^2$",
        "cmap": CMAP2,
        "vmin": 0,
        "vmax": 50,
        "color_step": 2.0,
        "tick_step": 5.0,
    },
    "IWV_deviation": {
        "label": "IWV deviation [kgm$^{-2}$]",
        "units": "kgm$^{-2}$",
        "cmap": CMAP3,
        "vmin": -10,
        "vmax": 10, 
        "color_step": 0.1,
        "tick_step": 1.0,
        }
}   