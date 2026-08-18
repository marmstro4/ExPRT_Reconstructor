import utilities
import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from itertools import combinations
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
import csv
import sys
from scipy.optimize import least_squares

def shift(number, std_dev):
    shift = np.random.normal(0, std_dev)  # Generate Gaussian shift
    return number + shift

def group3(points):
    if len(points) != 3:
        raise ValueError("Function expects exactly 3 points")

    min_pair = None
    min_dist = float('inf')

    for p1, p2 in combinations(points, 2):
        d = utilities.distance(p1, p2)
        if d < min_dist:
            min_dist = d
            min_pair = (p1, p2)

    return min_pair

def group4(points):
    if len(points) != 4:
        raise ValueError("Function expects exactly 4 points")

    # Find the pair with the smallest distance
    min_pair = None
    min_dist = float('inf')

    for p1, p2 in combinations(points, 2):
        d = utilities.distance(p1, p2)
        if d < min_dist:
            min_dist = d
            min_pair = (p1, p2)

    # Group 1: closest pair
    group1 = list(min_pair)

    # Group 2: the remaining two points
    group2 = [p for p in points if p not in group1]

    return group1, group2

def get_hits(line, numbers, simresx, simresy, simresz):
    hits = 0
    points = []

    # Check for up to four valid hits

    if -200 <= numbers[line][1] <= 200:
        point1 = utilities.Point3D(
            shift(numbers[line][1], simresx),
            shift(numbers[line][2], simresy),
            shift(numbers[line][3], simresz)
        )
        points.append(point1)
        hits += 1

    if -200 <= numbers[line][4] <= 200:
        point1 = utilities.Point3D(
            shift(numbers[line][4], simresx),
            shift(numbers[line][5], simresy),
            shift(numbers[line][6], simresz)
        )
        points.append(point1)
        hits += 1

    if -200 <= numbers[line][7] <= 200:
        point1 = utilities.Point3D(
            shift(numbers[line][7], simresx),
            shift(numbers[line][8], simresy),
            shift(numbers[line][9], simresz)
        )
        points.append(point1)
        hits += 1

    if -200 <= numbers[line][10] <= 200:
        point1 = utilities.Point3D(
            shift(numbers[line][10], simresx),
            shift(numbers[line][11], simresy),
            shift(numbers[line][12], simresz)
        )
        points.append(point1)
        hits += 1

    hit = hits > 1

    return points, hit

def FitGroupsAdv(points, target_thickness, beam_spot):
    # -------------------------------------------------
    # Convert points
    # -------------------------------------------------

    points_np = np.array([
        [p.x,p.y,p.z]
        for p in points
    ], dtype=float)



    # -------------------------------------------------
    # Group hits
    # -------------------------------------------------

    group1, group2 = group4(points)

    group1 = np.array([
        [p.x,p.y,p.z] for p in group1
    ])

    group2 = np.array([
        [p.x,p.y,p.z] for p in group2
    ])



    # -------------------------------------------------
    # Initial directions
    # -------------------------------------------------

    u1 = group1[1]-group1[0]
    u2 = group2[1]-group2[0]

    u1 /= np.linalg.norm(u1)
    u2 /= np.linalg.norm(u2)



    # -------------------------------------------------
    # Initial vertex from closest approach
    # -------------------------------------------------

    A1 = np.eye(3)-np.outer(u1,u1)
    A2 = np.eye(3)-np.outer(u2,u2)

    try:

        vertex0 = np.linalg.solve(
            A1+A2,
            A1@group1[0]+A2@group2[0]
        )

    except:

        vertex0=np.mean(points_np,axis=0)



    # -------------------------------------------------
    # Convert directions to angles
    # -------------------------------------------------

    def get_angles(u):

        theta=np.arccos(
            np.clip(u[2],-1,1)
        )

        phi=np.arctan2(
            u[1],
            u[0]
        )

        return theta,phi



    theta1,phi1=get_angles(u1)
    theta2,phi2=get_angles(u2)



    initial=np.array([

        vertex0[0],
        vertex0[1],
        vertex0[2],

        theta1,
        phi1,

        theta2,
        phi2

    ])



    # -------------------------------------------------
    # Direction function
    # -------------------------------------------------

    def direction(theta,phi):

        return np.array([

            np.sin(theta)*np.cos(phi),
            np.sin(theta)*np.sin(phi),
            np.cos(theta)

        ])




    # -------------------------------------------------
    # Residual function
    # -------------------------------------------------

    def residuals(params):

        vertex=params[:3]


        d1=direction(
            params[3],
            params[4]
        )

        d2=direction(
            params[5],
            params[6]
        )


        residual=[]


        # proton 1 hits

        for p in group1:

            diff=p-vertex

            perp=diff-np.dot(diff,d1)*d1

            residual.append(
                np.linalg.norm(perp)
            )


        # proton 2 hits

        for p in group2:

            diff=p-vertex

            perp=diff-np.dot(diff,d2)*d2

            residual.append(
                np.linalg.norm(perp)
            )



        # ---------------------------------------------
        # Beam spot likelihood
        # ---------------------------------------------

        r2 = (
            vertex[0]**2 +
            vertex[1]**2
        )


        # Gaussian prior:
        #
        # chi2 = r^2/sigma^2

        beam_penalty = np.sqrt(
            r2/(beam_spot**2)
        )


        residual.append(
            beam_penalty
        )


        # z target constraint as weak penalty

        half = target_thickness/2.0

        if abs(vertex[2]) > half:

            residual.append(
                (abs(vertex[2])-half)
            )


        else:

            residual.append(0.0)



        return np.array(residual)



    # -------------------------------------------------
    # Fit
    # -------------------------------------------------

    result = least_squares(
        residuals,
        initial,
        max_nfev=100
    )



    vertex=result.x[:3]


    return vertex

def direction(theta,phi):

    return np.array([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ])

def Fit2Hits(points,target_thickness,beam_spot):


    p1=np.array([
        points[0].x,
        points[0].y,
        points[0].z
    ])

    p2=np.array([
        points[1].x,
        points[1].y,
        points[1].z
    ])


    u=p2-p1
    u/=np.linalg.norm(u)


    def residual(t):

        vertex=p1+t[0]*u


        # beam likelihood
        beam_term = np.sqrt(
            (vertex[0]**2+vertex[1]**2)
            /
            beam_spot**2
        )


        return np.array([
            beam_term
        ])


    result=least_squares(
        residual,
        [0.0]
    )


    vertex=p1+result.x[0]*u


    vertex[2]=np.clip(
        vertex[2],
        -target_thickness/2,
        target_thickness/2
    )


    return vertex

def group3(points):
    """
    Returns the closest pair,
    assumed to be one proton track.
    """

    if len(points)!=3:
        raise ValueError(
            "Function expects exactly 3 points"
        )

    min_pair=None
    min_dist=float('inf')

    for p1,p2 in combinations(points,2):

        d=utilities.distance(p1,p2)

        if d < min_dist:
            min_dist=d
            min_pair=(p1,p2)

    return min_pair

def Fit3Hits(points,target_thickness,beam_spot):


    # find proton pair

    proton1=group3(points)


    proton2=[
        p for p in points
        if p not in proton1
    ][0]


    group1=np.array([
        [p.x,p.y,p.z]
        for p in proton1
    ])


    p3=np.array([
        proton2.x,
        proton2.y,
        proton2.z
    ])



    # initial track directions

    u1=group1[1]-group1[0]
    u1/=np.linalg.norm(u1)


    u2=p3-group1[0]
    u2/=np.linalg.norm(u2)


    # initial vertex:
    # closest approach of lines


    A1=np.eye(3)-np.outer(u1,u1)

    A2=np.eye(3)-np.outer(u2,u2)


    try:

        vertex0=np.linalg.solve(
            A1+A2,
            A1@group1[0]+A2@p3
        )

    except:

        vertex0=np.mean(
            np.array([
                group1[0],
                group1[1],
                p3
            ]),
            axis=0
        )



    def angles(u):

        return (
            np.arccos(u[2]),
            np.arctan2(u[1],u[0])
        )


    theta1,phi1=angles(u1)
    theta2,phi2=angles(u2)



    initial=np.array([
        vertex0[0],
        vertex0[1],
        vertex0[2],
        theta1,
        phi1,
        theta2,
        phi2
    ])



    def residual(params):

        vertex=params[:3]


        d1=direction(
            params[3],
            params[4]
        )


        d2=direction(
            params[5],
            params[6]
        )


        residuals=[]


        # proton 1 hits

        for p in group1:

            diff=p-vertex

            perp=diff-np.dot(diff,d1)*d1

            residuals.extend(perp)


        # proton 2 hit

        diff=p3-vertex

        perp=diff-np.dot(diff,d2)*d2

        residuals.extend(perp)



        # beam profile likelihood

        residuals.append(
            vertex[0]/beam_spot
        )

        residuals.append(
            vertex[1]/beam_spot
        )


        return np.array(residuals)



    result=least_squares(
        residual,
        initial,
        max_nfev=100
    )


    vertex=result.x[:3]


    vertex[2]=np.clip(
        vertex[2],
        -target_thickness/2,
        target_thickness/2
    )


    return vertex

def vertex_reco(line, numbers, target_thickness, beam_spot,simresx,simresy,simresz):

    points, hit = get_hits(line, numbers,simresx,simresy,simresz)

    fitcent = None

    if len(points) == 2:

        fitcent = Fit2Hits(
            points,
            target_thickness,
            beam_spot
        )

    elif len(points) == 3:

        fitcent = Fit3Hits(
            points,
            target_thickness,
            beam_spot
        )

    elif len(points) == 4:

        fitcent = FitGroupsAdv(
            points,
            target_thickness,
            beam_spot
        )

    return fitcent
