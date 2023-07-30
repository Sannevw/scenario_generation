import numpy as np
import matplotlib.pyplot as plt
import torch


def get_groundtruth(samples, y_offset_task, y_offset_safety=None, spec=None, safety_spec=None):
    '''
            This function creates the groundtruths values for a given set of samples
    '''
    groundtruth = list()
    for sample in samples:
        gt = list()

        gts = list()

        for s in sample:
            gt.append([spec([s[:3], s[3:6], s[6:9], s[9:]]) - y_offset_task])
            if safety_spec is not None:
                gts.append([safety_spec([s[:3], s[3:6], s[6:9], s[9:]]) - y_offset_safety])

        if safety_spec is not None:
            groundtruth.append(np.concatenate((np.array(gt), np.array(gts)), axis=1))
        else:
            groundtruth.append(np.array(gt))
    return groundtruth


def get_samples(model, latent_size, N_desired, desired_scores):
    '''
            Get the samples for specific desired satisfaction values C 
    '''
    samples = list()

    for item in desired_scores:
        if isinstance(item, float):
            score = item
            safety = False
        else:
            score, safety_score = item[0], item[1]
            safety = True
        c = torch.Tensor(np.vstack(np.ones(N_desired)) * score)

        if safety:
            c2 = torch.Tensor(np.vstack(np.ones(N_desired)) * safety_score)
            c = torch.hstack((c, c2))

        z = torch.randn([c.size(0), latent_size]).to('cpu')

        x = model.inference(z, c)

        samples.append(x.detach().cpu().numpy())
    return samples


def descriptive_stats(desired_scores, groundtruth=None):
    '''
         This function gives the mean and std of the differences between groundtruth (the actual score for the 
         sample) and desired score.
    '''

    means = list()
    stds = list()
    diffs = list()

    for gt, desired_scores in zip(groundtruth, desired_scores):
        if not isinstance(desired_scores, float):
            desired_scores = list(desired_scores)
        diff = desired_scores - gt  # -scores

        print("ground truth: ", gt[0:10])
        print("mean diff: ", np.mean(diff, axis=0))
        mean = np.mean(diff)
        std = np.std(diff)

        means.append(mean)
        stds.append(std)
        diffs.append(diff)

    #         print("Descriptives for score {:.2f} ".format(score) )# + dataset.y_offset_task))
    #         print("Mean: {}".format(mean))
    #         print("Standard deviation: {} \n".format(std))

    return means, stds, diffs
