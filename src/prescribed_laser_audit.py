"""Point evaluation of inspected notebook laser expressions, without FEniCS."""
import ast
import copy
import json

import numpy as np


class _ExpressionBase:
    def __init__(self, **kwargs):
        self.expression_metadata = kwargs


class NotebookLaser:
    """Extract only reviewed path definitions and laser classes, never solver cells.

    Input notebooks are trusted, manually inspected research code. This is not a
    general sandbox for arbitrary notebook functions. The original eval method
    is retained; a copy of its path prefix exposes its prescribed focus.
    """
    def __init__(self, path, family, interval):
        book = json.loads(path.read_text())
        self.sources = {i: ''.join(c['source']) for i, c in enumerate(book['cells'])
                        if c['cell_type'] == 'code'}
        tree = ast.parse(self.sources[6])
        assert isinstance(tree.body[0], ast.For)
        body = tree.body[0].body
        self.parameters = {}
        self.locations = {}
        for cell, nodes in [(3, ast.parse(self.sources[3]).body), (6, body)]:
            for node in nodes:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Name):
                        try:
                            value = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            continue
                        self.parameters[target.id] = value
                        self.locations[target.id] = (cell, node.lineno, node.end_lineno)
                if isinstance(node, ast.FunctionDef) and node.name in ('rho', 'Cp', 'k'):
                    self.parameters[node.name] = ast.literal_eval(node.body[0].value)
                    self.locations[node.name] = (cell, node.lineno, node.end_lineno)
        batch = ast.parse(self.sources[5]).body[0]
        assert interval in ast.literal_eval(batch.value)
        names = {
            'Diagonal': ('generate_zigzag_path_with_linespace', 'SmoothZigZagLaser'),
            'Spiral': ('generate_archimedean_spiral_path_and_length', 'SmoothLaserPath'),
        }[family]
        definitions = [copy.deepcopy(n) for n in body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert len(definitions) == 2
        cls = next(n for n in definitions if isinstance(n, ast.ClassDef))
        evaluator = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'eval')
        focus = copy.deepcopy(evaluator)
        focus.name = 'prescribed_focus'
        prefix_end = next(i for i, n in enumerate(focus.body)
                          if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                          and n.targets[0].id == 'r_focus')
        focus.body = focus.body[:prefix_end] + [ast.Return(value=ast.Tuple(
            elts=[ast.Name(id='x_focus', ctx=ast.Load()), ast.Name(id='y_focus', ctx=ast.Load())],
            ctx=ast.Load()))]
        cls.body.append(focus)
        namespace = dict(np=np, UserExpression=_ExpressionBase,
                         power=self.parameters['power'], sigma=self.parameters['sigma'])
        module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
        exec(compile(module, str(path) + ':cell6:selected-definitions', 'exec'), namespace)
        p = self.parameters
        if family == 'Diagonal':
            x, y = namespace[names[0]](p['length_x'], p['length_y'], p['length_y']/interval)
            self.distance = sum(np.sqrt((x[i]-x[i-1])**2+(y[i]-y[i-1])**2)
                                for i in range(1, len(x)))
            self.arguments = dict(paths_x=x, paths_y=y, velocity=p['velocity'])
        else:
            x, y, lengths, cumulative, self.distance = namespace[names[0]](
                N=interval, L=p['length_x'], dt_path=p['dt_path'])
            self.arguments = dict(paths_x=x, paths_y=y, segment_lengths=lengths,
                                  cumulative_lengths=cumulative, total_distance=self.distance,
                                  velocity=p['velocity'])
        self.expression_class = namespace[names[1]]
        self.path = np.column_stack([x, y])
        self.frame_count = int(np.ceil(self.distance/p['velocity']/p['dt'])) - 1
        t = 0.
        times = []
        for _ in range(self.frame_count):
            t += p['dt']
            times.append(t)
        self.times = np.asarray(times)

    def expression(self, time):
        return self.expression_class(t=time, degree=2, **self.arguments)

    def focus(self, time):
        return np.asarray(self.expression(time).prescribed_focus(None, None))

    def evaluate(self, points, time):
        values = [None]
        self.expression(time).eval(values, np.asarray(points).T)
        return np.asarray(values[0])


def flux_metrics(g, kg, weights):
    """Signed error is exported conductivity-gradient minus prescribed flux."""
    g, kg, w = map(np.asarray, (g, kg, weights))
    if w.sum() <= 0 or np.sum(w*g*g) <= 0:
        raise ValueError('Region must have positive area and prescribed-flux norm')
    error = kg-g
    avg = lambda x: float(np.sum(w*x)/w.sum())
    return dict(area_m2=float(w.sum()), bias_W_m2=avg(error),
                rmse_W_m2=np.sqrt(avg(error**2)), mae_W_m2=avg(abs(error)),
                relative_L2=np.sqrt(np.sum(w*error**2)/np.sum(w*g*g)),
                prescribed_integral_W=float(np.sum(w*g)), exported_integral_W=float(np.sum(w*kg)),
                integral_ratio=float(np.sum(w*kg)/np.sum(w*g)),
                negative_exported_integral_W=float(np.sum(w*np.minimum(kg, 0))),
                negative_exported_area_fraction=avg(kg < 0))
