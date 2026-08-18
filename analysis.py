import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import utilities

import csv
import os
import numpy as np

def save_resolution_efficiency(vertex_err, proton_count, total_events, outfile,  target_thickness, filename="res_eff.csv"):

    vertex_err = np.asarray(vertex_err, dtype=float)

    popt = analyse_vertex(vertex_err, target_thickness)

    # contains only z values
    sigma = popt[2]

    efficiency = proton_count / total_events

    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow([
                "sigma_z_mm",
                "efficiency",
                "reconstructed_events",
                "total_events"
            ])

        writer.writerow([
            sigma,
            efficiency,
            proton_count,
            total_events,
            outfile
        ])

    print(f"Sigma z = {sigma:.4f} mm")
    print(f"Efficiency = {efficiency:.4f}")

def vertex3D(positions2):
    """
    Plot reconstructed 3D vertex positions.

    Parameters
    ----------
    positions2 : list or np.ndarray
        List of reconstructed vertices.
        Each element should be [x,y,z]
    """

    positions2 = np.asarray(positions2, dtype=float)

    if positions2.shape[1] != 3:
        raise ValueError(
            "positions2 must contain [x,y,z] vertices"
        )


    x = positions2[:,0]
    y = positions2[:,1]
    z = positions2[:,2]


    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(
        x,
        y,
        z,
        s=5,
        alpha=0.5
    )


    # Draw beam spot cylinder

    beam_radius = np.max(
        np.sqrt(x*x+y*y)
    )

    theta = np.linspace(0,2*np.pi,100)

    zmin = np.min(z)
    zmax = np.max(z)

    X = beam_radius*np.cos(theta)
    Y = beam_radius*np.sin(theta)

    for zi in [zmin,zmax]:

        ax.plot(
            X,
            Y,
            np.ones_like(theta)*zi,
            linewidth=1
        )


    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")


    ax.set_title(
        "3D Vertex Reconstruction"
    )


    # Equal aspect ratio

    max_range = np.array([
        x.max()-x.min(),
        y.max()-y.min(),
        z.max()-z.min()
    ]).max()/2.0


    mid_x = (x.max()+x.min())/2
    mid_y = (y.max()+y.min())/2
    mid_z = (z.max()+z.min())/2


    ax.set_xlim(
        -2,
        2
    )

    ax.set_ylim(
       -2,
        2
    )

    ax.set_zlim(
       -25,
        25
    )


    plt.show()

def analyse_vertex(vertex_err, target_thickness):

    # Histogram
    counts, edges = np.histogram(
        vertex_err,
        bins=1000,
        range=(-target_thickness/2, target_thickness/2)
    )

    centers = 0.5 * (edges[:-1] + edges[1:])


    # Select fit region
    mask = (
        (centers >= -target_thickness/2) &
        (centers <= target_thickness/2)
    )

    xfit = centers[mask]
    yfit = counts[mask]


    # Initial parameter guesses
    A0 = np.max(yfit)
    mu0 = xfit[np.argmax(yfit)]
    sigma0 = 5.0


    # Gaussian fit
    popt, pcov = curve_fit(
        utilities.gaussian,
        xfit,
        yfit,
        p0=[A0, mu0, sigma0]
    )


    A, mu, sigma = popt

    A_err, mu_err, sigma_err = np.sqrt(
        np.diag(pcov)
    )


    # Plot
    plt.figure(figsize=(8,5))

    plt.hist(
        vertex_err,
        bins=1000,
        range=(-target_thickness/2,target_thickness/2),
        histtype='step',
        color='black',
        label='Data'
    )


    x = np.linspace(-target_thickness/2,target_thickness/2,1000)

    plt.plot(
        x,
        utilities.gaussian(x,*popt),
        'r-',
        lw=2,
        label=fr'Gaussian Fit\n'
        fr'$\mu$ = {mu:.3f} ± {mu_err:.3f}\n'
        fr'$\sigma$ = {sigma:.3f} ± {sigma_err:.3f}'
    )


    #plt.xlabel("Value")
    #plt.ylabel("Counts")
    #plt.title("Gaussian Fit")
    #plt.legend()
    #plt.show()

    print(f"Mean     = {mu:.6f} ± {mu_err:.6f}")
    print(f"Sigma    = {sigma:.6f} ± {sigma_err:.6f}")
    print(f"Amplitude= {A:.6f} ± {A_err:.6f}")

    return popt
