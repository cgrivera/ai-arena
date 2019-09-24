import matplotlib.pyplot as plt
import numpy as np
import random, colorsys

#plot episode results from policies, 
#averaging over windows and displaying with alphas
#saving to filename
def plot_policy_records(records, windows, alphas, filename, colors=None, offsets=None):

	if colors is None:
		colors = []
		for pr in records:
			colors.append(randomRGBPure())

	if offsets is None:
		offsets = []
		for pr in records:
			offsets.append(0)

	fig, ax = plt.subplots(1,1,figsize=(12,6))

	for widx, windowsize in enumerate(windows):

		alpha = alphas[widx]

		for pridx, pol in enumerate(records):

			steps = pol.ep_cumlens
			results = pol.ep_results

			for k in range(len(steps)):
				steps[k] += offsets[pridx]

			rolling_avg_buffer = []
			rolling_avg_results = []
			rolling_avg_steps = []

			for i in range(len(steps)):
				rolling_avg_buffer.append(results[i])

				if len(rolling_avg_buffer) > windowsize:
					rolling_avg_buffer.pop(0)

					rolling_avg_results.append(np.mean(rolling_avg_buffer))
					rolling_avg_steps.append(steps[i])

			#plot
			ax.plot(rolling_avg_steps, rolling_avg_results, color=colors[pridx], alpha=alpha)

	ax.set_xlabel("Training Steps")
	ax.set_ylabel("Episodic Reward")

	plt.savefig(filename)


def randomRGBPure():
    h = random.random()
    s = random.uniform(0.8,0.9)
    v = random.uniform(0.8,0.9)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r,g,b)