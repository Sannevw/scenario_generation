import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle as pkl
from abc import ABC, abstractmethod
from sklearn.manifold import TSNE
from scipy.spatial.transform import Rotation as R
import copy as copy
import inspect
from gigalib.utils import *
import random


class SpaTiaLDataset(ABC):

    def __init__(self, N: int, objects: int, fixed_seed=True, seed=1):
        self.N = N
        self.objects = objects
        self.seed = seed

        # set seed
        if fixed_seed:
            np.random.seed(self.seed)

    @property
    def shape(self):
        return self.X.shape

    @abstractmethod
    def save_to_file(self, path):
        pass

    @abstractmethod
    def load_from_file(cls, path):
        pass

    @abstractmethod
    def plot(self, x):
        pass

    def torch_dataloader(self, **kwargs):
        my_dataset = TensorDataset(self.X, self.Y)  # create your datset
        return DataLoader(my_dataset, **kwargs)  # create your dataloader

    def plot_tsne(self, step=1):
        if self.N > 1:
            trainX, trainY = self.numpy
            # do tsne
            Y = TSNE(n_components=2, init='random').fit_transform(np.array(trainX[::step, :]))
            # plot
            plt.scatter(Y[:, 0], Y[:, 1], marker='o', c=trainY[::step], s=150)
            plt.colorbar()
            plt.autoscale()
            plt.xlabel('latent 1')
            plt.ylabel('latent 2')
            plt.show()
        else:
            print('TSNE requires minimum 2 samples to run!')

    @property
    def numpy(self):
        return self.X.cpu().detach().numpy(), self.Y.cpu().detach().numpy()


class SpaTiaLBBoxDataset(SpaTiaLDataset):
    X_MAX = 1
    Y_MAX = 1

    def __init__(self, N: int, objects: int, dimensions: list, fixed_seed=True, seed=1, rotation=False):
        super(SpaTiaLBBoxDataset, self).__init__(N, objects, fixed_seed, seed)
        self.dimensions = dimensions if type(dimensions) == np.ndarray else np.array(dimensions)
        self.rotation = rotation

        assert objects == len(dimensions)

        if N > 0:
            self.X, self.Y = self.create_dataset()

    def save_to_file(self, path):
        data = dict()
        data['X'] = self.X.cpu().detach().numpy()
        data['Y'] = self.Y.cpu().detach().numpy()
        data['N'] = self.N
        data['O'] = self.objects
        data['D'] = self.dimensions
        with open(path, 'wb') as f:
            pkl.dump(data, f)

    @classmethod
    def load_from_file(cls, path):
        with open(path, 'rb') as f:
            data = pkl.load(f)
        d = SpaTiaLBBoxDataset(0, 0, [])
        d.N = data['N']
        d.X = torch.Tensor(data['X'])
        d.Y = torch.Tensor(data['Y'])
        d.objects = data['O']
        d.dimensions = data['D']
        return d

    def create_dataset(self):
        # create example dataset

        def corners(p, w, l, theta=None):
            corners = np.array([[-w / 2, l / 2], [-w / 2, -l / 2], [w / 2, -l / 2], [w / 2, l / 2]])
            if theta is not None:
                rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                                      [np.sin(theta), np.cos(theta)]])
                corners = np.array([np.dot(c, rotMatrix) for c in corners])
            return (p + corners).flatten()

        objects = None
        ####
        # Shape is:
        #
        # Sample 0: vertices for each object after another
        # Sample 1: vertices for each object after another
        # Sample 2: vertices for each object after another
        # ....
        ####
        Y = None
        for i in range(self.objects):
            l_x = self.dimensions[i, 0]
            l_y = self.dimensions[i, 1]
            p_o = np.random.rand(self.N, 2) * [self.X_MAX - l_x, self.Y_MAX - l_y] + [l_x / 2., l_y / 2.]
            rotation = None
            if self.rotation:
                rotation = np.random.rand(self.N) * 4 * np.pi - 2 * np.pi

            if i == 0:
                Y = copy.copy(p_o[:, 0])
            if i == 1:
                Y -= p_o[:, 0]

            block = np.vstack([corners(p_o[j], self.dimensions[i, 0], self.dimensions[i, 1],
                                       theta=rotation[j] if self.rotation else None) for j in range(len(p_o))])
            objects = np.concatenate((objects, block), axis=1) if objects is not None else block

        return torch.Tensor(objects), torch.Tensor(np.vstack(Y))

    def compute_score(self, poses):
        def create_object_list(poses):
            return poses.reshape(self.objects, 2)

        return None

    def plot_dataset(self, step=1, marker='', linestyle='-', color=None, **kwargs):
        for i in range(0, self.N, step):
            self.plot(self.X[i], marker, linestyle, color, **kwargs)

    def legend(self):
        cm = plt.cm.get_cmap('tab10')
        lp = lambda i: plt.plot([], color=cm.colors[i], ms=10, mec="none",
                                label="Object {:g}".format(i), ls="", marker="o")[0]
        handles = [lp(i) for i in range(self.objects)]
        plt.legend(handles=handles)

    def plot(self, x, marker='', linestyle='-', color=None, **kwargs):
        cm = plt.cm.get_cmap('tab10')
        data = x.cpu().detach().numpy() if type(x) == torch.Tensor else x
        for i in range(self.objects):
            self.plot_block(data[i * 8:(i + 1) * 8].reshape((4, 2)), marker, linestyle,
                            color if color is not None else cm.colors[i], **kwargs)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])

    def plot_block(self, x, marker, linestyle, color, **kwargs):
        plt.plot(x[:, 0], x[:, 1], marker=marker, color=color, linestyle=linestyle, **kwargs)


class SpaTiaLPoseDataset(SpaTiaLDataset):
    X_MAX = 1
    Y_MAX = 1
    THETA_MAX = 2 * np.pi

    def __init__(self, N: int, objects: int, specification, safety_spec, fixed_seed=True, seed=1, debug=True):
        super(SpaTiaLPoseDataset, self).__init__(N, objects, fixed_seed, seed)

        self.specification = specification
        self.safety_spec = safety_spec
        self.safety = None

        if debug:
            print('Got task specification: {}'.format(self.spec2text(self.specification)))
            print('Got safety specification: {}'.format(self.spec2text(self.safety_spec)))

        if N > 0:
            self.X, self.Y = self.create_dataset()

    def boost_positives(self, factor, eps_p=0.001):
        if self.safety_spec is not None:
            idx_t = np.where(self.numpy[1][:, 0] >= 0. - self.y_offset_task)[0]
            idx_s = np.where(self.numpy[1][:, 1] >= 0. - self.y_offset_safety)[0]
            idx = set(idx_t.flatten()).intersection(set(idx_s.flatten()))
            idx = list(idx)
        else:
            idx = np.where(self.numpy[1][:, 0] >= 0. - self.y_offset_task)[0]

        x_new = self.numpy[0].tolist()
        y_new = self.numpy[1].tolist()
        for i in idx:
            pert, sc = self.pertubate([x_new[i]] * factor, eps_p=eps_p)
            for i in range(len(pert)):
                if self.safety_spec is None:
                    checkScore = sc[i] + self.y_offset_task
                    if all(checkScore >= 0.):
                        x_new.append(pert[i])
                        y_new.append(sc[i])
                else:
                    checkScore = sc[i] + [self.y_offset_task, self.y_offset_safety]
                    if any(checkScore >= 0.):
                        x_new.append(pert[i])
                        y_new.append(sc[i])

        self.X = torch.Tensor(np.array(x_new))
        self.Y = torch.Tensor(np.array(y_new))

    def balance(self, n_max, STEPS=0.025, boost=False):
        x_new = list()
        y_new = list()
        X, Y = self.numpy

        values = np.arange(Y[:, 0].min(), Y[:, 0].max() + STEPS, STEPS)
        counts = [0] * len(values)
        indices = [[] for a in range(len(values))]
        for i in range(len(X)):
            score = Y[i, 0]
            idx = np.where(values > score)[0].min()
            if counts[idx] < n_max:
                counts[idx] += 1
                indices[idx].append(i)
                x_new.append(X[i])
                y_new.append(Y[i])

        # boost goes through list again and pertubates random samples
        if boost:
            for i in range(len(indices)):
                # if not enough samples yet, pertubate
                if counts[i] < n_max and counts[i] > 0:
                    missing = n_max - counts[i]
                    samples = random.choices(indices[i], k=missing)
                    samples = X[samples]
                    pert, sc = self.pertubate(samples)
                    for i in range(missing):
                        x_new.append(pert[i])
                        y_new.append(sc[i])

        self.X = torch.Tensor(np.array(x_new))
        self.Y = torch.Tensor(np.array(y_new))

    def pertubate(self, samples, eps_p=0.01):
        # eps_p = 0.001
        new_samples = list()
        new_scores = list()
        for s in samples:
            pert = s + np.random.uniform(-1, 1, len(s)) * eps_p
            # pert_score =
            new_samples.append(pert)
        # new_scores.append(pert_score)
        scores = self.compute_score(new_samples)
        if len(scores[0]) > 1:
            scores -= np.array([self.y_offset_task, self.y_offset_safety])
        else:
            scores -= self.y_offset_task
        return new_samples, scores

    def spec2text(self, spec_to_print):
        if spec_to_print is None:
            return "None"
        funcString = str(inspect.getsourcelines(spec_to_print)[0])
        funcString = funcString.strip("['\\n']").split(" = ")[1]
        return funcString

    def save_to_file(self, path):
        data = dict()
        data['X'] = self.X.cpu().detach().numpy()
        data['Y'] = self.Y.cpu().detach().numpy()
        data['N'] = self.N
        data['O'] = self.objects
        data['spec'] = self.spec2text(self.specification)
        data['safety_spec'] = self.spec2text(self.safety_spec)
        data['yoff'] = self.y_offset_task
        data['yoffsafety'] = self.y_offset_safety
        with open(path, 'wb') as f:
            pkl.dump(data, f)

    @classmethod
    def load_from_file(cls, path, debug=False):
        with open(path, 'rb') as f:
            data = pkl.load(f)
        d = SpaTiaLPoseDataset(0, 0, lambda o: o, lambda o: o, debug=debug)
        d.N = data['N']
        d.X = torch.Tensor(data['X'])
        d.Y = torch.Tensor(data['Y'])
        d.objects = data['O']
        d.specification = data['spec']
        d.y_offset_task = data['yoff']
        d.y_offset_safety = data["yoffsafety"]
        return d

    def create_samples(self, samples):
        data_samples = []

        for (begin, end) in zip(samples[:-1], samples[1:]):
            d = SpaTiaLPoseDataset(0, 0, lambda o: o, lambda o: o)
            print("begin and end: {} - {}".format(begin, end))
            d.X, d.Y = torch.Tensor(self.X[begin:end]), torch.Tensor(self.Y[begin:end])
            d.N = samples[1]
            d.objects = self.objects  # dataset['O']
            d.specification = self.specification  # ['spec']
            d.y_offset_task = self.y_offset_task  # dataset['yoff']
            d.y_offset_safety = self.y_offset_safety
            data_samples.append(d)

        return data_samples

    def create_dataset(self):
        # create example dataset

        objects = None
        ####
        # Shape is:
        #
        # Sample 0: x,y,theta for each object after another
        # Sample 1: x,y,theta for each object after another
        # Sample 2: x,y,theta for each object after another
        # ....
        ####

        for i in range(self.objects):
            poses = np.random.rand(self.N, 3) * [self.X_MAX, self.Y_MAX, 1]
            objects = np.concatenate((objects, poses), axis=1) if objects is not None else poses

        satisfaction = self.compute_score(objects)
        if self.safety_spec is not None:

            self.y_offset_task = satisfaction[:, 0].min()
            self.y_offset_safety = satisfaction[:, 1].min()

            print("y offset task: ", self.y_offset_task)
            print("y offset safety: ", self.y_offset_safety)

            satisfaction -= [self.y_offset_task, self.y_offset_safety]
            print("Satisfaction: ", satisfaction[0:10])
            # satisfaction.min()
        else:
            self.y_offset_task = satisfaction.min()
            self.y_offset_safety = None
            satisfaction -= satisfaction.min()

        return torch.Tensor(objects), torch.Tensor(satisfaction)

    def compute_score(self, samples):
        def create_object_list(poses):
            return poses.reshape(self.objects, 3)[:, :3]

        results = list()
        safety_res = list()
        for s in samples:
            objects = create_object_list(s)
            results.append(self.specification(objects))
            if self.safety_spec is not None:
                safety_res.append(self.safety_spec(objects))

        if self.safety_spec is not None:
            return np.vstack(zip(results, safety_res))
        else:
            return np.vstack(results)

    def histogram(self, STEPS=0.025):
        x, y = self.numpy
        return histogram(y, STEPS)

    def plot_histogram(self, STEPS=0.025, offset=0.0):
        x, y_ = copy.deepcopy(self.numpy)

        #         print("y min: ", y_.min())
        #         print("y max: ", y_.max())
        y_ += offset
        #         print("after min: ", y_.min())
        #         print("After max: ", y_.max())
        plot_histogram(y_, STEPS)

    def plot_2Dhistogram(self, STEPS=0.025, offset=0.0):
        x, y_ = copy.deepcopy(self.numpy)
        y_ += offset
        plot_2Dhistogram(y_, STEPS)

    def plot_dataset(self, step=1, marker='', linestyle='-', color=None, **kwargs):
        for i in range(0, self.N, step):
            self.plot(self.X[i], marker, linestyle, color, **kwargs)

    def plot(self, x, marker='', linestyle='-', color=None, evaluate=False, **kwargs):
        cm = plt.cm.get_cmap('tab10')
        data = x.cpu().detach().numpy() if type(x) == torch.Tensor else x
        for i in range(self.objects):
            self.plot_pose(data[i * 3:(i + 1) * 3], marker, linestyle, color if color is not None else cm.colors[i],
                           **kwargs)
            if i == 0 and evaluate:
                circle1 = plt.Circle(data[i * 3:(i + 1) * 3], 0.25, color='r', edgecolor='r', alpha=0.2,
                                     label="Safety Distance")
                plt.gca().add_patch(circle1)
                self.safety = circle1
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel('X position')
        plt.ylabel('Y position')

    def legend(self, evaluate=False):
        cm = plt.cm.get_cmap('tab10')
        lp = lambda i: plt.plot([], color=cm.colors[i], ms=10, mec="none",
                                label="Object {:g}".format(i), ls="", marker="o")[0]

        handles = [lp(i) for i in range(self.objects)]
        if evaluate:
            handles.append(self.safety)
        plt.legend(handles=handles)

    def plot_pose(self, x, marker, linestyle, color, **kwargs):
        length = 0.1
        angle = x[2] * 2 * np.pi  # self.THETA_MAX - self.THETA_MAX
        dx, dy = np.array([np.cos(angle), np.sin(angle)]) * length
        plt.arrow(x[0], x[1], dx, dy, linestyle=linestyle, color=color, width=0.005, **kwargs)  # ,head_width=0.1)
        plt.plot(x[0], x[1], marker=marker, color=color, **kwargs)

    def get_satisfying_samples(self, task=True, safety=True):
        if task and not safety:
            idx = np.where(self.numpy[1][:, 0] >= 0. - self.y_offset_task)[0]
        if not task and safety:
            idx = np.where(self.numpy[1][:, 1] >= 0. - self.y_offset_safety)[0]
        if task and safety:  # self.safety_spec is not None:
            idx_t = np.where(self.numpy[1][:, 0] >= 0. - self.y_offset_task)[0]
            idx_s = np.where(self.numpy[1][:, 1] >= 0. - self.y_offset_safety)[0]
            idx = set(idx_t.flatten()).intersection(set(idx_s.flatten()))
            idx = list(idx)

        return len(idx), self.numpy[0][idx, :], self.numpy[1][idx, :]


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import pickle as pkl
    from abc import ABC, abstractmethod
    from sklearn.manifold import TSNE
    from gigalib.datasets import SpaTiaLBBoxDataset, SpaTiaLPoseDataset, SpaTiaLDataset
    from gigalib.utils import *

    spec = lambda o: leftOf(o[0], o[1])
    safety = lambda o: closeTo(o[0], o[1])

    spec = lambda o: and_(onPlaceMat(o[0]),
                          and_(centerAbovePlaceMat(o[1]),
                               and_(leftOf(o[2], o[0]),
                                    and_(centerRightPlaceMat(o[3]),
                                         rightOf(o[3], o[0])
                                         ))))
    safety = lambda o: orientation(o[3], NORTH)
    s = SpaTiaLPoseDataset(5000, 4, spec, safety)
    # s.plot_histogram()
    print('Satisfying examples: {}'.format(s.get_satisfying_samples(task=True, safety=False)[0]))
    print('Satisfying examples: {}'.format(s.get_satisfying_samples(task=False, safety=True)[0]))
    print('Satisfying examples: {}'.format(s.get_satisfying_samples(task=True, safety=True)[0]))
    s.plot_2Dhistogram()
    plt.show()
    s.balance(2000)  # , boost=True)
    s.boost_positives(1000)
    print('Satisfying examples: {}'.format(s.get_satisfying_samples()[0]))
    s.plot_histogram()
    s.plot_2Dhistogram()
    plt.show()
    # s.balance(150, boost=True)
    # s.balance(150, boost=True)
    s.plot_histogram()
    plt.show()
