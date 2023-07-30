import numpy as np

import torch
import torch.nn as nn

import time
from collections import defaultdict

import os

from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import statistics
from scipy.stats import norm

from gigalib.datasets import SpaTiaLPoseDataset

try:
    import pybullet
except ImportError as e:
    print('Pybullet is not installed on this system, just using matplotlib instead')
from gigalib.simulation import KitchenRenderer, YCBObject, YCBObjectLoader
from gigalib.utils import *

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact, interactive, fixed
import abc


class Experiment(abc.ABC):

    @abc.abstractmethod
    def init(self):
        pass

    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def interactive(self):
        pass


class Experiment1(Experiment):

    def init(self):
        self.w_spec = widgets.RadioButtons(
            options=['Task (safety satisfied)', 'Safety (task satisfied)', 'Both'],
            value='Task (safety satisfied)',
            #    value='pineapple', # Defaults to 'pineapple'
            #    layout={'width': 'max-content'}, # If the items' names are long
            description='Control:',
            disabled=False
        )
        display(self.w_spec)

        self.w_pyb = widgets.RadioButtons(
            options=['Yes', 'No'],
            value='Yes',
            #    value='pineapple', # Defaults to 'pineapple'
            #    layout={'width': 'max-content'}, # If the items' names are long
            description='Pybullet:',
            disabled=False
        )
        display(self.w_pyb)

        self.slider_task = widgets.SelectionSlider(description='Task', value=0.0, continuous_update=False,
                                                   options=np.around(np.arange(-0.9, 0.91, 0.01), decimals=2),
                                                   readout_format='.01f')

        self.slider_safety = widgets.SelectionSlider(description='Safety', value=0.0, continuous_update=False,
                                                     options=np.around(np.arange(-0.9, 0.91, 0.01), decimals=2),
                                                     readout_format='.01f')
        self.pb_init = False

    def load_model(self):
        self.model = torch.load('files/exp1/EXP1_ae_PoseDataset_nrobjects=2_nrsamples=20000SEED=57245.torch')
        self.dataset = SpaTiaLPoseDataset.load_from_file(
            "files/exp1/EXP1_initial_PoseDataset_nrobjects=2_nrsamples=20000.pkl")
        self.spec = lambda o: and_(leftOf(o[0], o[1]), aboveOf(o[0], o[1]))
        self.spec_s = lambda o: not_(closeTo(o[0], o[1]))
        self.mode = 0 if self.w_spec.value == 'Task (safety satisfied)' else (
            1 if self.w_spec.value == 'Safety (task satisfied)' else 2)

        if self.w_pyb.value == 'Yes':
            self.init_pybullet()

    def sample_plot(self, task=0, safety=0):
        if self.mode == 0:
            print('Desired task satisfaction value is: {}'.format(task))
            print('Safety specification is satisfied')
        if self.mode == 1:
            print('Desired safety satisfaction value is: {}'.format(safety))
            print('Task specification is satisfied')
        if self.mode == 2:
            print('Desired task satisfaction value is: {}'.format(task))
            print('Desired safety satisfaction value is: {}'.format(safety))

        N_desired = 10000
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.mode == 0:
            # just task
            task_score = float(task) - self.dataset.y_offset_task
            c = torch.Tensor(np.vstack(np.ones(N_desired) * task_score))
            c2 = torch.Tensor(np.vstack(
                np.random.uniform(abs(self.dataset.y_offset_safety), self.dataset.numpy[1][:, 1].max(), N_desired)))
            c = torch.hstack((c, c2))
        if self.mode == 1:
            # just safety
            safety_score = float(safety) - self.dataset.y_offset_safety
            c = torch.Tensor(np.vstack(
                np.random.uniform(abs(self.dataset.y_offset_task), self.dataset.numpy[1][:, 0].max(), N_desired)))
            c2 = torch.Tensor(np.vstack(np.ones(N_desired) * safety_score))
            c = torch.hstack((c, c2))
        if self.mode == 2:
            # task and safety
            task_score = float(task) - self.dataset.y_offset_task
            safety_score = float(safety) - self.dataset.y_offset_safety
            c = torch.Tensor(np.vstack(np.ones(N_desired) * task_score))
            c2 = torch.Tensor(np.vstack(np.ones(N_desired) * safety_score))
            c = torch.hstack((c, c2))

        z = torch.randn([c.size(0), 16]).to(device)
        x = self.model.inference(z, c=c)

        samples = x.cpu().detach().numpy()
        samples_c = c.cpu().detach().numpy()

        sample_fix = list()
        sample_c_fix = list()

        if self.mode == 0:
            for i, s in enumerate(samples):
                score = self.spec([s[:3], s[3:6]]) - self.dataset.y_offset_task
                score_s = self.spec_s([s[:3], s[3:6]])
                if np.isclose(score, task_score, atol=0.001) and score_s >= 0.:
                    sample_fix.append(s)
                    sample_c_fix.append(samples_c[i])
                    break

        if self.mode == 1:
            for i, s in enumerate(samples):
                score = self.spec([s[:3], s[3:6]])
                score_s = self.spec_s([s[:3], s[3:6]]) - self.dataset.y_offset_safety
                if np.isclose(score_s, safety_score, atol=0.001) and score >= 0.:
                    sample_fix.append(s)
                    sample_c_fix.append(samples_c[i])
                    break

        if self.mode == 2:
            for i, s in enumerate(samples):
                score = self.spec([s[:3], s[3:6]]) - self.dataset.y_offset_task
                score_s = self.spec_s([s[:3], s[3:6]]) - self.dataset.y_offset_safety
                if np.isclose(score, task_score, atol=0.001) and np.isclose(score_s, safety_score, atol=0.001):
                    sample_fix.append(s)
                    sample_c_fix.append(samples_c[i])
                    break

        samples = np.array(sample_fix)
        samples_c = np.array(sample_c_fix)

        if len(samples) > 0:
            for p in range(len(samples)):
                plt.figure(figsize=(5, 5))
                # plt.title(c[p, 0].item() + y_offset)

                self.dataset.plot(samples[p], marker='*')
                self.dataset.legend()
                plt.show()

            if self.w_pyb.value == 'Yes':
                # set positions
                # self.plate.pose = self.r.scale_to_workspace([s[1], s[0], s[2]])  # s[:3])
                # self.mug.pose = self.r.scale_to_workspace([s[4], s[3], s[5]])
                # set positions
                self.mug.pose = self.r.scale_to_workspace([s[1], s[0], s[2]])
                self.plate.pose = self.r.scale_to_workspace([s[4], s[3], s[5]])

                plt.figure(figsize=(8, 8))
                px = self.r.get_image(top_view=True)
                plt.imshow(px, origin='upper')
                plt.gca().invert_yaxis()
                plt.xticks([])
                plt.yticks([])
                plt.show()
        else:
            print('Cannot find satisfying solution for desired satisfaction value. Try lowering the value.')

    def interactive(self):
        if self.mode == 0:
            interact(self.sample_plot, task=self.slider_task, safety=fixed(0))

        if self.mode == 1:
            interact(self.sample_plot, task=fixed(0), safety=self.slider_safety)

        if self.mode == 2:
            interact(self.sample_plot, task=self.slider_task, safety=self.slider_safety)

    def init_pybullet(self):
        if not self.pb_init:
            # render kitchen
            self.r = KitchenRenderer(gui=False)

            # load objects
            self.mug = YCBObjectLoader.load('mug')
            self.plate = YCBObjectLoader.load('plate')
            self.pb_init = True


class Experiment2(Experiment):

    def init(self):
        self.w_pyb = widgets.RadioButtons(
            options=['Yes', 'No'],
            value='Yes',
            #    value='pineapple', # Defaults to 'pineapple'
            #    layout={'width': 'max-content'}, # If the items' names are long
            description='Pybullet:',
            disabled=False
        )
        display(self.w_pyb)

        self.slider = widgets.SelectionSlider(description='Satisfaction', value=0.0, continuous_update=False,
                                              options=np.around(np.arange(-0.5, 0.11, 0.01), decimals=2),
                                              readout_format='.01f')
        self.pb_init = False
    def load_model(self):
        self.model = torch.load('files/exp2/newModel_BOOSTED_ae_PoseDataset_nrobjects=4_nrsamples=500000.torch')  #
        self.dataset = SpaTiaLPoseDataset.load_from_file(
            "files/exp2/exp2_new.pkl")
        self.spec = lambda o: and_(onPlaceMat(o[0]),  # plate on placemat
                                   and_(centerAbovePlaceMat(o[1]),  # mug above and closeto placemat
                                        and_(centerRightPlaceMat(o[2]),  # knife right of and closeto placemt
                                             and_(orientation(o[2], NORTH),  # knife upright
                                                  and_(aboveOf(o[3], o[1]), leftOf(o[3], o[1])
                                                       # right of and above of mug
                                                       )))))

        if self.w_pyb.value == 'Yes':
            self.init_pybullet()

    def sample_plot(self, task=0):
        print('Desired task satisfaction value is: {}'.format(task))

        N_desired = 10000
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vertices = np.array([PT[:2] + [-PT[2], -PT[3]],
                             PT[:2] + [-PT[2], PT[3]],
                             PT[:2] + [PT[2], PT[3]],
                             PT[:2] + [PT[2], -PT[3]],
                             PT[:2] + [-PT[2], -PT[3]]])

        task_score = float(task) - self.dataset.y_offset_task

        c = torch.Tensor(np.vstack(np.ones(N_desired) * task_score))
        z = torch.randn([c.size(0), 128]).to(device)
        x = self.model.inference(z, c=c)

        samples = x.cpu().detach().numpy()
        samples_c = c.cpu().detach().numpy()

        sample_fix = list()
        sample_c_fix = list()

        for i, s in enumerate(samples):
            score = self.spec([s[:3], s[3:6], s[6:9], s[9:12]]) - self.dataset.y_offset_task
            if np.isclose(score, task_score, atol=0.001):
                sample_fix.append(s)
                sample_c_fix.append(samples_c[i])
                break

        samples = np.array(sample_fix)
        samples_c = np.array(sample_c_fix)

        if len(samples) > 0:
            for p in range(len(samples)):
                plt.figure(figsize=(5, 5))
                self.dataset.plot(samples[p], marker='*')
                plt.plot(vertices[:, 0], vertices[:, 1], '-k')
                self.dataset.legend()
                plt.show()

            if self.w_pyb.value == 'Yes':
                # set positions
                self.plate.pose = self.r.scale_to_workspace([s[1], s[0], s[2]])  # s[:3])
                self.mug.pose = self.r.scale_to_workspace([s[4], s[3], s[5]])
                self.knife.pose = self.r.scale_to_workspace([s[7], s[6], s[8]])
                self.cracker.pose = self.r.scale_to_workspace([s[10], s[9], s[11]])

                plt.figure(figsize=(8, 8))
                px = self.r.get_image(top_view=True)
                plt.imshow(px, origin='upper')
                plt.gca().invert_yaxis()
                plt.xticks([])
                plt.yticks([])
                plt.show()
        else:
            print('Cannot find satisfying solution for desired satisfaction value. Try lowering the value.')

    def interactive(self):
        interact(self.sample_plot, task=self.slider)

    def init_pybullet(self):
        if not self.pb_init:
            # render kitchen
            self.r = KitchenRenderer(gui=False)

            # load objects
            self.mug = YCBObjectLoader.load('mug')
            self.plate = YCBObjectLoader.load('plate')
            self.knife = YCBObjectLoader.load('knife')
            self.cracker = YCBObjectLoader.load('cracker')
            self.pb_init = True


class Experiment3(Experiment):

    def init(self):
        self.w_obj = widgets.RadioButtons(
            options=['3', '5', '7'],
            value='3',
            #    value='pineapple', # Defaults to 'pineapple'
            #    layout={'width': 'max-content'}, # If the items' names are long
            description='Objects:',
            disabled=False
        )
        display(self.w_obj)

        self.w_pyb = widgets.RadioButtons(
            options=['Yes', 'No'],
            value='Yes',
            #    value='pineapple', # Defaults to 'pineapple'
            #    layout={'width': 'max-content'}, # If the items' names are long
            description='Pybullet:',
            disabled=False
        )
        display(self.w_pyb)

        self.slider = widgets.SelectionSlider(description='Satisfaction', value=0.0,continuous_update=False,
                                              options=np.around(np.arange(-0.1, 0.11, 0.01), decimals=2),
                                              readout_format='.01f')

        self.pb_init = False

    def load_model(self):
        self.OBJECTS = int(self.w_obj.value)
        if self.OBJECTS == 3:
            self.model = torch.load(
                'files/exp3/EXP1_gan_PoseDataset_nrobjects=3_nrsamples=20000SEED=67940.torch')
            self.dataset = SpaTiaLPoseDataset.load_from_file(
                "files/exp3/newModel_IROS_PoseDataset_nrobjects=3_nrsamples=20000.pkl")
            self.spec = lambda o: or_(and_(leftOf(o[0], o[1]), leftOf(o[1], o[2])),
                                      or_(and_(rightOf(o[0], o[1]), rightOf(o[1], o[2])),
                                          or_(and_(belowOf(o[0], o[1]), belowOf(o[1], o[2])),
                                              and_(aboveOf(o[0], o[1]), aboveOf(o[1], o[2])))))
            self.spec_left = lambda o: and_(leftOf(o[0], o[1]), leftOf(o[1], o[2]))
            self.spec_right = lambda o: and_(rightOf(o[0], o[1]), rightOf(o[1], o[2]))
            self.spec_below = lambda o: and_(belowOf(o[0], o[1]), belowOf(o[1], o[2]))
            self.spec_above = lambda o: and_(aboveOf(o[0], o[1]), aboveOf(o[1], o[2]))

        if self.OBJECTS == 5:
            self.model = torch.load(
                'files/exp3/EXP1_gan_PoseDataset_nrobjects=5_nrsamples=50000SEED=22175.torch')
            self.dataset = SpaTiaLPoseDataset.load_from_file(
                "files/exp3/newModel_IROSBoosted_EPS=0.05_PoseDataset_nrobjects=5_nrsamples=50000.pkl")
            self.spec = lambda o: or_(and_(leftOf(o[0], o[1]), and_(leftOf(o[1], o[2]), and_(leftOf(o[2], o[3]),
                                                                                             leftOf(o[3], o[4])))),
                                      or_(and_(rightOf(o[0], o[1]), and_(rightOf(o[1], o[2]), and_(rightOf(o[2], o[3]),
                                                                                                   rightOf(o[3],
                                                                                                           o[4])))),
                                          or_(and_(belowOf(o[0], o[1]),
                                                   and_(belowOf(o[1], o[2]), and_(belowOf(o[2], o[3]),
                                                                                  belowOf(o[3],
                                                                                          o[4])))),
                                              and_(aboveOf(o[0], o[1]),
                                                   and_(aboveOf(o[1], o[2]), and_(aboveOf(o[2], o[3]),
                                                                                  aboveOf(o[3], o[4]))))
                                              )
                                          )
                                      )
            self.spec_left = lambda o: and_(leftOf(o[0], o[1]), and_(leftOf(o[1], o[2]), and_(leftOf(o[2], o[3]),
                                                                                             leftOf(o[3], o[4]))))
            self.spec_right = lambda o: and_(rightOf(o[0], o[1]), and_(rightOf(o[1], o[2]), and_(rightOf(o[2], o[3]),
                                                                                             rightOf(o[3], o[4]))))
            self.spec_below = lambda o: and_(belowOf(o[0], o[1]), and_(belowOf(o[1], o[2]), and_(belowOf(o[2], o[3]),
                                                                                             belowOf(o[3], o[4]))))
            self.spec_above = lambda o: and_(aboveOf(o[0], o[1]), and_(aboveOf(o[1], o[2]), and_(aboveOf(o[2], o[3]),
                                                                                             aboveOf(o[3], o[4]))))
        if self.OBJECTS == 7:
            self.model = torch.load(
                'files/exp3/EXP1_gan_PoseDataset_nrobjects=7_nrsamples=50000SEED=56870.torch')
            self.dataset = SpaTiaLPoseDataset.load_from_file(
                "files/exp3/newModel_IROSBoosted_EPS=0.05_PoseDataset_nrobjects=7_nrsamples=50000.pkl")
            self.spec = lambda o: or_(and_(leftOf(o[0], o[1]), and_(leftOf(o[1], o[2]), and_(leftOf(o[2], o[3]),
                                                                                             and_(leftOf(o[3], o[4]),
                                                                                                  and_(leftOf(o[4],
                                                                                                              o[5]),
                                                                                                       leftOf(o[5],
                                                                                                              o[
                                                                                                                  6])))))),
                                      or_(and_(rightOf(o[0], o[1]), and_(rightOf(o[1], o[2]), and_(rightOf(o[2], o[3]),
                                                                                                   and_(rightOf(o[3],
                                                                                                                o[4]),
                                                                                                        and_(rightOf(
                                                                                                            o[4],
                                                                                                            o[5]),
                                                                                                             rightOf(
                                                                                                                 o[5],
                                                                                                                 o[
                                                                                                                     6])))))),
                                          or_(and_(belowOf(o[0], o[1]),
                                                   and_(belowOf(o[1], o[2]), and_(belowOf(o[2], o[3]),
                                                                                  and_(belowOf(o[3],
                                                                                               o[4]),
                                                                                       and_(
                                                                                           belowOf(o[4],
                                                                                                   o[
                                                                                                       5]),
                                                                                           belowOf(o[5],
                                                                                                   o[
                                                                                                       6])))))),
                                              and_(aboveOf(o[0], o[1]),
                                                   and_(aboveOf(o[1], o[2]), and_(aboveOf(o[2], o[3]),
                                                                                  and_(aboveOf(o[3],
                                                                                               o[4]),
                                                                                       and_(
                                                                                           aboveOf(o[4],
                                                                                                   o[
                                                                                                       5]),
                                                                                           aboveOf(o[5],
                                                                                                   o[
                                                                                                       6])))))),
                                              )
                                          )
                                      )
            self.spec_left = lambda o: and_(leftOf(o[0], o[1]), and_(leftOf(o[1], o[2]), and_(leftOf(o[2], o[3]),
                                                                                                 and_(leftOf(o[3], o[4]),
                                                                                                      and_(leftOf(o[4],
                                                                                                                  o[5]),
                                                                                                           leftOf(o[5],
                                                                                                                  o[
                                                                                                                      6]))))))
            self.spec_right = lambda o: and_(rightOf(o[0], o[1]), and_(rightOf(o[1], o[2]), and_(rightOf(o[2], o[3]),
                                                                                                 and_(rightOf(o[3], o[4]),
                                                                                                      and_(rightOf(o[4],
                                                                                                                  o[5]),
                                                                                                           rightOf(o[5],
                                                                                                                  o[
                                                                                                                      6]))))))
            self.spec_below = lambda o: and_(belowOf(o[0], o[1]), and_(belowOf(o[1], o[2]), and_(belowOf(o[2], o[3]),
                                                                                                 and_(belowOf(o[3], o[4]),
                                                                                                      and_(belowOf(o[4],
                                                                                                                  o[5]),
                                                                                                           belowOf(o[5],
                                                                                                                  o[
                                                                                                                      6]))))))
            self.spec_above = lambda o: and_(aboveOf(o[0], o[1]), and_(aboveOf(o[1], o[2]), and_(aboveOf(o[2], o[3]),
                                                                                                 and_(aboveOf(o[3], o[4]),
                                                                                                      and_(aboveOf(o[4],
                                                                                                                  o[5]),
                                                                                                           aboveOf(o[5],
                                                                                                                  o[
                                                                                                                      6]))))))
        if self.w_pyb.value == 'Yes':
            self.init_pybullet()

    def sample_plot(self, value):
        print('Desired satisfaction value is: {}'.format(value))
        #########
        #### FOR CHOOSING TASK SCORE
        #########

        N_desired = 1000
        y_max, y_offset = self.dataset.numpy[1][:,
                          0].max(), self.dataset.y_offset_task
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        task_score = float(value) - self.dataset.y_offset_task

        # arange conditions and sample
        c = torch.Tensor(np.vstack(np.ones(N_desired) * task_score))
        z = torch.randn([c.size(0), 256]).to(device)
        x = self.model.inference(z, c=c)

        samples = x.cpu().detach().numpy()
        samples_c = c.cpu().detach().numpy()

        sample_fix = list()
        sample_c_fix = list()

        for i, s in enumerate(samples):
            score = self.spec(s.reshape(self.OBJECTS, 3)) - self.dataset.y_offset_task
            if np.isclose(score, task_score, atol=0.001):
                sample_fix.append(s)
                sample_c_fix.append(samples_c[i])
                break
        samples = np.array(sample_fix)
        samples_c = np.array(sample_c_fix)
        if len(samples) > 0:
            # test left of
            if self.spec_left(s.reshape(self.OBJECTS, 3)) >=0.:
                print('This example is sorted -leftOf-.')
            if self.spec_right(s.reshape(self.OBJECTS, 3)) >=0.:
                print('This example is sorted -rightOf-.')
            if self.spec_below(s.reshape(self.OBJECTS, 3)) >=0.:
                print('This example is sorted -belowOf-.')
            if self.spec_above(s.reshape(self.OBJECTS, 3)) >=0.:
                print('This example is sorted -aboveOf-.')

            for p in range(len(samples)):
                plt.figure(figsize=(5, 5))
                # plt.title(c[p, 0].item() + y_offset)

                self.dataset.plot(samples[p], marker='*')
                self.dataset.legend()
                plt.show()

            if self.w_pyb.value == 'Yes':
                # set positions
                self.plate.pose = self.r.scale_to_workspace([s[1], s[0], s[2]])  # s[:3])
                self.mug.pose = self.r.scale_to_workspace([s[4], s[3], s[5]])
                self.knife.pose = self.r.scale_to_workspace([s[7], s[6], s[8]])
                if self.OBJECTS > 3:
                    self.cracker.pose = self.r.scale_to_workspace([s[10], s[9], s[11]])
                    self.can.pose = self.r.scale_to_workspace([s[13], s[12], s[14]])
                if self.OBJECTS > 5:
                    self.bottle.pose = self.r.scale_to_workspace([s[16], s[15], s[17]])
                    self.bulle.pose = self.r.scale_to_workspace([s[19], s[18], s[20]])

                plt.figure(figsize=(8, 8))
                px = self.r.get_image(top_view=True)
                plt.imshow(px, origin='upper')
                plt.gca().invert_yaxis()
                plt.xticks([])
                plt.yticks([])
                plt.show()
        else:
            print('Cannot find satisfying solution for desired satisfaction value. Try lowering the value.')

    def init_pybullet(self):

        if not self.pb_init:
            # render kitchen
            self.r = KitchenRenderer(gui=False)

            # load objects
            self.mug = YCBObjectLoader.load('mug')
            self.plate = YCBObjectLoader.load('plate')
            self.knife = YCBObjectLoader.load('knife')
            if self.OBJECTS > 3:
                self.cracker = YCBObjectLoader.load('cracker')
                self.can = YCBObjectLoader.load('can')
            if self.OBJECTS > 5:
                self.bottle = YCBObjectLoader.load('bottle')
                self.bulle = YCBObjectLoader.load('bulle')
            self.pb_init = True
    def interactive(self):
        interact(self.sample_plot, value=self.slider)
