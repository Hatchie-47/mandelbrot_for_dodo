import sys, getopt, os
import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb
from matplotlib.animation import FuncAnimation, PillowWriter

def n_for_point(xr, xi, n_max=20):
    """
    For point on complex plain count at which iteration (up until n_max) it's absolute value is higher then 2
    :param xr: real part of point
    :param xi: imaginary part of point
    :param n_max: max number of iterations
    :return: number of iterations between 1 and n_max+1 - in case even n_max doesn't yield abs value higher than 2
    """
    n = 1
    c = complex(xr, xi)
    z = complex(0, 0)
    while n <= n_max:
        z = z*z+c
        if abs(z) > 2:
            break
        n += 1
    return n

def calc_frame(x_shape, y_shape, res, n_max):
    """
    Create data for a frame of mandelbrot
    :param x_shape: min and max values on x axis
    :param y_shape: min and max values on y axis
    :param res: resolution o the frame
    :param n_max: max number of iterations
    :return: x_plain and y_plain as x and y for countourf, ns as results
    """
    x_vector = np.linspace(x_shape[0], x_shape[1], res)
    y_vector = np.linspace(y_shape[0], y_shape[1], res)
    x_plain, y_plain = np.meshgrid(x_vector, y_vector)
    coords = np.c_[x_plain.ravel(), y_plain.ravel()]
    ns = np.array([n_for_point(x, y, n_max) for x, y in coords]).reshape(x_plain.shape)
    return ns


def main():
    """
    Program that creates mandelbrot zooming animation:
    
    :param -n int: max number of iterations for point, default 75
    :param -p string: matplotlib cmap to be used, default 'plasma' - in case of invalid cmap default will be used
    :param -r int: resolution of the plot, can only be square so only one number, default 500
    :param -f int: number of animation frames, fps is 15 so this also controls length of the animation, default 90
    :param -s string: filename of the saved gif (without .gif extension), default 'mandelbrot'
    :param -m float: max zoom e.g. what portion of the original frame remains in the last frame, default 1e-6
    """

    # Handling arguments
    opts = ''
    def_cmap = 'plasma'
    if len(sys.argv) > 1:
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'n:p:r:f:s:m:', ['help'])
        except getopt.GetoptError:
            print(main.__doc__)
            sys.exit()

    opts = dict(opts)

    if '--help' in opts:
        print(main.__doc__)
        sys.exit()

    res = int(opts['-r']) if '-r' in opts else 500
    n_max = int(opts['-n']) if '-n' in opts else 75
    fcount = int(opts['-f']) if '-f' in opts else 90
    cmap = opts['-p'] if '-p' in opts else def_cmap
    zoom_max = float(opts['-m']) if '-m' in opts else 1e-6
    filename = opts['-s'] if '-s' in opts else 'mandelbrot'
    filename = os.path.join('.', filename + '.gif')
    zoom_on = (-.75,.1)
    x_dist_l = abs(-2 - zoom_on[0])
    x_dist_h = abs(2 - zoom_on[0])
    y_dist_l = abs(-2 - zoom_on[1])
    y_dist_h = abs(2 - zoom_on[1])

    if cmap not in plt.colormaps():
        print(f'Argument -p {cmap} is not a matplotlib colormap, "{def_cmap}" will be used as default...')
        cmap = def_cmap

    # Creating the figure
    fig, ax = plt.subplots()
    ax.set_title(f'Mandelbrot set with max iteration value {n_max} and resolution {res}x{res}')
    ax.axis('off')

    # Progressbar
    pbar = pb.ProgressBar(widgets=['Frames processed: ', pb.SimpleProgress(), pb.Bar(), pb.ETA()], maxval=fcount).start()
    pbar.update(0)

    # Make next frame of the animation
    def update_graph(i):
        x_shape = (-2+((x_dist_l*(1-zoom_max)*i)/fcount), 2-((x_dist_h*(1-zoom_max)*i)/fcount))
        y_shape = (-2+((y_dist_l*(1-zoom_max)*i)/fcount), 2-((y_dist_h*(1-zoom_max)*i)/fcount))
        ns = calc_frame(x_shape, y_shape, res, n_max)
        ax.imshow(ns, cmap=cmap)
        pbar.update(i+1)


    # Creating animation
    animation = FuncAnimation(fig, update_graph, fcount)
    writer = PillowWriter(fps=15)

    # Save and end
    animation.save(filename, writer=writer)
    pbar.finish()
    print(f'Animation created and saved as {filename}.')
    plt.close()

if __name__ == '__main__':
    main()