"""
Including all the ploting functions, 2D, 3D, dots
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import my_functions.functions_general as fg

# standard values for fonts
ticksFontSize = 18
xyLabelFontSize = 20
legendFontSize = 20


def plot_2D(field, x=None, y=None, xname='', yname='', map='jet', vmin=None, vmax=None, title='',
            ticksFontSize=ticksFontSize, xyLabelFontSize=xyLabelFontSize,
            axis_equal=False, xlim=None, ylim=None, ax=None, show=True, ijToXY=True, origin='lower',
            **kwargs):
    fieldToPlot = field
    if ijToXY:
        origin = 'lower'
        fieldToPlot = np.copy(field).transpose()
    if x is None:
        x = range(np.shape(fieldToPlot)[0])
    if y is None:
        y = range(np.shape(fieldToPlot)[1])
    if ax is None:
        if axis_equal:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
    image = plt.imshow(fieldToPlot,
                       interpolation='bilinear', cmap=map,
                       origin=origin, aspect='auto',  # aspect ration of the axes
                       extent=[y[0], y[-1], x[0], x[-1]],
                       vmin=vmin, vmax=vmax, label='sdfsd', **kwargs)
    cbr = plt.colorbar(image, shrink=0.8, pad=0.02, fraction=0.1)
    cbr.ax.tick_params(labelsize=ticksFontSize)
    plt.xticks(fontsize=ticksFontSize)
    plt.yticks(fontsize=ticksFontSize)
    ax.set_xlabel(xname, fontsize=xyLabelFontSize)
    ax.set_ylabel(yname, fontsize=xyLabelFontSize)
    plt.title(title, fontweight="bold", fontsize=26)
    if axis_equal:
        ax.set_aspect('equal', adjustable='box')
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_plane_go(z, mesh, fig=None, opacity=0.6, show=False,
                  colorscale=([0, '#aa9ce2'], [1, '#aa9ce2']), **kwargs):
    """
    plotting the cross-section XY plane in 3d go figure
    :param z: z coordinate of the plane
     :param colorscale: need values for 0 and 1 (the same), or something like 'RdBu'
    :param kwargs: for go.Surface
    :return: fig
    """
    xyz = fg.arrays_from_mesh(mesh)

    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Surface(x=xyz[0], y=xyz[1], z=z,
                             opacity=opacity, colorscale=colorscale, showscale=False, **kwargs))
    if show:
        fig.show()
    return fig


def plot_3D_dots_go(dots, mode='markers', marker=None, fig=None, xyzNames=None, show=False):
    """
    plotting dots in the interactive window in browser using plotly.graph_objects
    :param dots: [[x,y,z],...]
    :param show: True if you want to show it instantly
    :return: fig
    """
    if marker is None:
        marker = {'size': 8, 'color': 'black'}
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=dots[:, 0], y=dots[:, 1], z=dots[:, 2],
                               mode=mode, marker=marker))
    if show:
        fig.show()
    return fig


def plot_3D_density(E, resDecrease=(1, 1, 1), mesh=None,
                    xMinMax=None, yMinMax=None, zMinMax=None,
                    surface_count=20, show=True,
                    opacity=0.5, colorscale='RdBu',
                    opacityscale=None, fig=None, **kwargs):
    """
    Function plots 3d density in the browser
    :param E: anything in real number to plot
    :param resDecrease: [a, b, c] steps in each direction
    :param xMinMax: values along x [xMinMax[0], xMinMax[1]] (boundaries)
    :param yMinMax: values along y [yMinMax[0], yMinMax[1]] (boundaries)
    :param zMinMax: values along z [zMinMax[0], zMinMax[1]] (boundaries)
    :param surface_count: numbers of layers to show. more layers - better resolution
                          but harder to plot it. High number can lead to an error
    :param opacity: needs to be small to see through all surfaces
    :param opacityscale: custom opacity scale [...] (google)
    :param kwargs: extra params for go.Figure
    :return: nothing since it's in the browser (not ax or fig)
    """
    if mesh is None:
        shape = np.array(np.shape(E))
        if resDecrease is not None:
            shape = (shape // resDecrease)
        if zMinMax is None:
            zMinMax = [0, shape[0]]
        if yMinMax is None:
            yMinMax = [0, shape[1]]
        if xMinMax is None:
            xMinMax = [0, shape[2]]

        X, Y, Z = np.mgrid[
                  xMinMax[0]:xMinMax[1]:shape[0] * 1j,
                  yMinMax[0]:yMinMax[1]:shape[1] * 1j,
                  zMinMax[0]:zMinMax[1]:shape[2] * 1j
                  ]
    else:
        X, Y, Z = mesh
    values = E[::resDecrease[0], ::resDecrease[1], ::resDecrease[2]]
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Volume(
        x=X.flatten(),  # collapsed into 1 dimension
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=values.min(),
        isomax=values.max(),
        opacity=opacity,  # needs to be small to see through all surfaces
        opacityscale=opacityscale,
        surface_count=surface_count,  # needs to be a large number for good volume rendering
        colorscale=colorscale,
        **kwargs
    ))
    if show:
        fig.show()
    return fig


def plot_scatter_3D(X, Y, Z, ax=None, size=plt.rcParams['lines.markersize'] ** 2, color=None,
                    viewAngles=(70, 0), show=True, **kwargs):
    """
    ploting dots using plt.scatter
    :param ax: if you want multiple plots in one ax
    :param size: dots size. Use >100 for a better look
    :param color: color of the dots. Default for a single plot is blue
    :param viewAngles: (70, 0) (phi, theta)
    :param kwargs: extra parameters for plt.scatter
    :return: ax
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, s=size, color=color, **kwargs)  # plot the point (2,3,4) on the figure
    ax.view_init(*viewAngles)
    if show:
        plt.show()
    return ax
