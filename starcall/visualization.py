""" Contains functions to visualize read calling data
"""

import pandas
import numpy as np

from . import utils
from . import reads


def find_basis():
    points = np.eye(4)

    point = points[0]
    vec1 = points[1] - point
    vec1 /= np.linalg.norm(vec1)

    point += np.sum((points[2] - point) * vec1) * vec1
    vec2 = points[2] - point
    vec2 /= np.linalg.norm(vec2)

    point += np.sum((points[3] - point) * vec2) * vec2
    vec3 = points[3] - point
    vec3 /= np.linalg.norm(vec3)

    return np.array([vec1, vec2, vec3])



def project_triangle(values):
    basis = find_basis()
    #values = values[:,:3]

    np.clip(values, 0, None, out=values)

    normed = values / np.maximum(0.0000000001, values.sum(axis=1).reshape(-1,1))
    #return normed
    return np.dot(basis, normed.T).T

    angle1 = np.arctan2(values[:,0], values[:,1]) / np.pi * 2
    length1 = np.linalg.norm(values[:,:2], axis=1)

    angle2 = np.arctan2(values[:,2], length1) / np.pi * 2
    angle1 = angle1 * (1 - angle2) + (angle2 / 2)
    angle2 = angle2 * np.sqrt(3) / 2
    length2 = np.linalg.norm([values[:,2], length1], axis=0)

    angle3 = np.arctan2(values[:,3], length2) / np.pi * 2
    angle1 = angle1 * (1 - angle3) + (angle3 / 2)
    angle2 = angle2 * (1 - angle3) + (angle3 / 2)
    angle3 = angle3 * np.sqrt(3) / 2
    length3 = np.linalg.norm([values[:,3], length2], axis=0)

    return np.array([angle1, angle2, angle3, length3]).T

def plot_testset(values, barcodes, path):

    #fig, axes = plt.subplots(nrows=values.shape[1], ncols=4, figsize=(6*4, 6*values.shape[1]))
    fig, axes = plt.subplots(nrows=values.shape[1], ncols=4, figsize=(6*4, 6*values.shape[1]), subplot_kw=dict(projection='3d'))

    #values -= values.mean(axis=1).reshape(values.shape[0], 1, values.shape[2])
    #values /= np.maximum(0.000001, values.std(axis=1).reshape(values.shape[0], 1, values.shape[2]))

    for cycle in range(values.shape[1]):
        #if cycle not in (0, 3, 6, 9, 11):
            #continue
        print ('cycle', cycle)
        clusters = np.argmax(values[:,cycle], axis=1)
        clusters = np.array(['GTAC'.index(barc[cycle]) if cycle < len(barc) else -1 for barc in barcodes])
        #values[cycle] /= np.linalg.norm(values[cycle], axis=0).reshape(1,-1)
        cur_values = values[:,cycle]
        print (cur_values.shape)
        correct_let = clusters == np.argmax(cur_values, axis=1)

        miscalled_vals = cur_values[~correct_let]
        miscalled_clusters = clusters[~correct_let]

        #normed = cur_values[: / np.linalg.norm(cur_values, axis=1)

        angle1, angle2, angle3, length = project_triangle(cur_values).T
        test_x, test_y, test_z = project_triangle(np.eye(4)).T

        #linspace = np.linspace(0, 1, 5)
        #test_coords = np.array(np.meshgrid(linspace, linspace, linspace, linspace))
        #test_coords = test_coords.reshape(4, -1).T
        #test_x, test_y, test_z, test_len = project_triangle(test_coords).T

        angles = np.linspace(20, 70, 4)
        for i in range(4):
            #axes[cycle,i].scatter(test_x, test_y, test_z, c=test_coords)
            axes[cycle,i].scatter(test_x, test_y, test_z, c=['purple', 'blue', 'green', 'red'])

            axes[cycle,i].scatter(angle1, angle2, angle3, c=clusters, s=1)

            axes[cycle,i].axes.set_aspect('equal')
            axes[cycle,i].view_init(30, angles[i])

    fig.savefig(path + '_dot_value_clusters_3d.png')

def plot_basevalues_plotly(path, table, cycle=None, line_plot=False):
    import plotly.graph_objects as go
    import plotly.express

    values, barcodes = table.reads.values, table.reads.sequences_array

    cycle_indices = np.arange(values.shape[1]).reshape(1, -1)
    cycle_indices = np.broadcast_to(cycle_indices, values.shape[:2])
    read_indices = np.arange(values.shape[0]).reshape(-1, 1)
    read_indices = np.broadcast_to(read_indices, values.shape[:2])

    if cycle is not None:
        values = values[:,cycle]
        cycle_indices = cycle_indices[:,cycle]
        read_indices = read_indices[:,cycle]
        barcodes = barcodes[:,cycle]

    values = values.reshape(-1, values.shape[-1])
    cycle_indices = cycle_indices.reshape(-1)
    read_indices = read_indices.reshape(-1)
    barcodes = barcodes.reshape(-1)

    #values -= values.mean(axis=1).reshape(values.shape[0], 1, values.shape[2])
    #values /= np.maximum(0.000001, values.std(axis=1).reshape(values.shape[0], 1, values.shape[2]))

    #clusters = np.argmax(values[:,cycle], axis=1)
    #clusters = np.array(['GTAC'.index(barc[cycle]) if cycle < len(barc) else -1 for barc in barcodes])
    #clusters = np.array([barc[cycle] if cycle < len(barc) else '?' for barc in barcodes])
    #values[cycle] /= np.linalg.norm(values[cycle], axis=0).reshape(1,-1)
    #cur_values = values[:,cycle]
    #print (cur_values.shape)
    #correct_let = clusters == np.argmax(cur_values, axis=1)

    #miscalled_vals = cur_values[~correct_let]
    #miscalled_clusters = clusters[~correct_let]

    #cur_values = cur_values[:100000]
    #clusters = clusters[:100000]

    x, y, z = project_triangle(values).T
    length = np.linalg.norm(values, axis=1)
    test_x, test_y, test_z = project_triangle(np.eye(4)).T

    linspace = np.linspace(0, 1, 5)
    test_coords = np.array(np.meshgrid(linspace, linspace, linspace, linspace))
    test_coords = test_coords.reshape(4, -1).T
    test_x, test_y, test_z = project_triangle(test_coords).T
    test_length = np.linalg.norm(test_coords, axis=1)

    #fig = go.Figure(data=[go.Scatter3d(x=test_x, y=test_y, z=test_z, symbol=clusters)])
    #fig = plotly.express.scatter_3d(x=test_x, y=test_y, z=test_z, color=test_length)
    np.clip(length, None, np.percentile(length, 99), out=length)

    plot_data = pandas.DataFrame(dict(
        x=x, y=y, z=z, length=length, symbol=barcodes,
        cycle=cycle_indices, index=read_indices,
        g=values[:,0], t=values[:,1], a=values[:,2], c=values[:,3]))

    #fig = plotly.express.scatter(x=x, y=y, color=length, symbol=barcodes, animation_frame=cycle_indices, animation_group=read_indices)
    #fig = plotly.express.scatter_3d(
            #x=x, y=y, z=z, color=length, symbol=barcodes,
            #animation_frame=cycle_indices, animation_group=read_indices,
            #range_x=[-0.7, 0.7], range_y=[-0.5, 0.9], range_z=[-0.5, 0.9])
    if line_plot:
        fig = plotly.express.line_3d(plot_data,
                x='x', y='y', z='z', color='index', symbol='cycle', markers=True,
                #animation_frame='cycle', animation_group='index',
                hover_data=['index', 'cycle', 'g', 't', 'a', 'c'],
                range_x=[-0.7, 0.7], range_y=[-0.5, 0.9], range_z=[-0.5, 0.9])
        #fig.update_traces(marker=dict(size=1))
    else:
        fig = plotly.express.scatter_3d(plot_data,
                x='x', y='y', z='z', color='length', symbol='symbol',
                animation_frame='cycle', animation_group='index',
                hover_data=['index', 'cycle', 'g', 't', 'a', 'c'],
                range_x=[-0.7, 0.7], range_y=[-0.5, 0.9], range_z=[-0.5, 0.9])
        fig.update_traces(marker=dict(size=1))

    #fig.write_html(path + '_dot_value_clusters_3d_cycle{:02}.html'.format(cycle))
    fig.write_html(path, auto_play=False)

    #fig.savefig(path + '_dot_value_clusters_3d.png')

