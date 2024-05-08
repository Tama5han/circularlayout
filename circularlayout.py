import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from math import radians, cos, sin


class CircularLayout:
    def plot(self, node_data, edge_data,
                ax = None,
                with_labels = False,
                node_size = 20,
                node_radius = 1,
                node_props = None,
                label_offset = 0.05,
                label_props = None,
                edge_width = 1,
                arrow_style = '-',
                edge_props = None,
                **kwargs
            ):

        
        if (show := ax is None):
            fig, ax = plt.subplots(**kwargs)
            
            plt.tick_params(
                labelbottom = False,
                labelleft = False,
                labelright = False,
                labeltop = False,
                bottom = False,
                left = False,
                right = False,
                top = False
            )

        
        nodes = self.node_layout(node_data, with_labels, node_size=node_size)
        edges = self.edge_layout(nodes, edge_data, edge_width=edge_width)
        
        self.draw_nodes(
            ax, nodes, with_labels,
            radius = node_radius,
            offset = label_offset,
            node_props = node_props,
            label_props = label_props
        )
        
        self.draw_edges(
            ax, edges,
            arrow_style = arrow_style,
            edge_props = edge_props
        )
        
        if show:
            plt.show()
    
    
    def plot_from_nx(
                self,
                G,
                label = None,
                group = None,
                size = None,
                width = None,
                **kwargs
            ):
        
        # Create node and edge data
        
        node_columns = dict()
        
        if label is not None: node_columns['label'] = label
        if group is not None: node_columns['group'] = group
        if size  is not None: node_columns['size']  = size
        
        node_data = {'node': []} | { col: [] for col in node_columns.keys() }

        
        for n, ndata in G.nodes.items():
            node_data['node'].append(n)
            
            for col, attr in node_columns.items():
                node_data[col].append(ndata[attr])
        
        node_data = pl.DataFrame(node_data)

        
        # Edge part
        
        edge_columns = dict()
        
        if width is not None: edge_columns['width'] = width
        
        edge_data = {'from': [], 'to': []} | { col: [] for col in edge_columns.keys() }
        
        for e, edata in G.edges.items():
            edge_data['from'].append(e[0])
            edge_data['to'].append(e[1])
            
            for col, attr in edge_columns.items():
                edge_data[col].append(edata[attr])
        
        edge_data = pl.DataFrame(edge_data)

        
        self.plot(node_data, edge_data, **kwargs)
    
    
    def node_layout(self, node_data, with_labels, node_size=20):
        assert 'node' in node_data.columns, 'Not found column "node"'
        
        # Add column "angle"
        nodes = node_data.with_columns(
            pl.Series('angle', np.linspace(0, 360, len(node_data) + 1)[:-1])
        )
        
        if 'size' not in node_data.columns:
            nodes = nodes.with_columns(pl.lit(node_size).alias('size'))
        
        # Add columns related to label
        if with_labels:
            if 'label' not in node_data.columns:
                nodes = nodes.with_columns(
                    pl.col('node').cast(pl.Utf8).alias('label')
                )
            
            nodes = nodes.with_columns(
                pl.when((pl.col('angle') >= 90) & (pl.col('angle') < 270))
                    .then(pl.col('angle') - 180)
                    .otherwise(pl.col('angle'))
                    .alias('rotation'),
                pl.when((pl.col('angle') < 90) | (pl.col('angle') >= 270))
                    .then(pl.lit('left'))
                    .otherwise(pl.lit('right'))
                    .alias('align')
            )
        
        return nodes
    
    
    def edge_layout(self, nodes, edge_data, edge_width=1):
        assert 'from' in edge_data.columns, 'Not found column "from"'
        assert 'to' in edge_data.columns, 'Not found column "to"'
        
        # Delete self-loops
        edges = edge_data.filter(pl.col('from') != pl.col('to'))

        # Add node angles
        edges = (
            edges
            .join(nodes, left_on='from', right_on='node')
            .rename({'angle': 'start'})
            .join(nodes, left_on='to', right_on='node')
            .rename({'angle': 'end'})
        )

        if 'width' not in edge_data.columns:
            edges = edges.with_columns(pl.lit(edge_width).alias('width'))

        return edges
    
    
    def draw_nodes(self, ax, nodes, with_labels, radius=1, offset=0.05, node_props=None, label_props=None):
        if node_props is None:
            node_props = dict()
        
        # Node positions
        angles = (np.pi / 180) * nodes['angle'].to_numpy()
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        sizes = nodes['size'].to_numpy()
        
        # Grouping nodes
        if 'group' in nodes.columns:
            groups = nodes['group'].to_numpy()
            for g in np.unique(groups):
                indices = np.where(groups == g)[0]
                ax.scatter(xs[indices], ys[indices], s=sizes[indices], label=str(group), **node_props)
        else:
            ax.scatter(xs, ys, s=sizes, **node_props)
        
        # Labeling
        if with_labels:
            if label_props is None:
                label_props = dict()
            
            xs = (radius + offset) * xs
            ys = (radius + offset) * ys
            for i, row in enumerate(nodes.iter_rows(named=True)):
                ax.text(
                    xs[i], ys[i], row['label'],
                    rotation_mode = 'anchor',
                    rotation = row['rotation'],
                    horizontalalignment = row['align'],
                    verticalalignment = 'center',
                    **label_props
                )
    
    
    def draw_edges(self, ax, edges, radius=1, arrow_style='-', edge_props=None):
        if edge_props is None:
            edge_props = dict()
        
        for row in edges.iter_rows(named=True):
            start = row['start']
            end = row['end']
            angle_s = radians(start)
            angle_e = radians(end)
            
            xy_s = (radius * cos(angle_s), radius * sin(angle_s))
            xy_e = (radius * cos(angle_e), radius * sin(angle_e))
            width = row['width']
            
            diff = abs(start - end)
            if abs(diff - 180) <= 1e-10:
                ax.annotate(
                    '', xy=xy_e, xytext=xy_s,
                    arrowprops = dict(
                        arrowstyle = arrow_style,
                        linewidth = width),
                    **edge_props
                )
                continue
            
            angle_d = radians(diff / 2)
            radius_m = abs((1 - sin(angle_d)) / cos(angle_d)) * radius
            
            mid = (start + end) / 2
            if diff < 180:
                angle_m = radians(mid)
            else:
                angle_m = radians(mid + 180)
            
            xy_m = (radius_m * cos(angle_m), radius_m * sin(angle_m))
            
            ax.annotate(
                '', xy=xy_m, xytext=xy_s,
                arrowprops = dict(
                    arrowstyle = '-',
                    linewidth = width,
                    shrinkB = 0,
                    connectionstyle = f'angle3,angleA={start},angleB={mid - 90}'),
                **edge_props
            )
            
            ax.annotate(
                '', xy=xy_e, xytext=xy_m,
                arrowprops = dict(
                    arrowstyle = arrow_style,
                    linewidth = width,
                    shrinkA = 0,
                    connectionstyle = f'angle3,angleA={mid - 90},angleB={end}'),
                **edge_props
            )
