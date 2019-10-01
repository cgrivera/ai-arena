import matplotlib.pyplot as plt
import numpy as np
import random, colorsys

#plot episode results from policies, 
#averaging over windows and displaying with alphas
#saving to filename
def plot_policy_records(records, windows, alphas, filename, colors=None, offsets=None):

	default_colors = []
	hues = [1, 4, 7, 10, 3, 6, 9, 12, 2, 5, 8, 11]
	for h in hues:
		default_colors.append(randomRGBPure(float(h)/12.0))

	if colors is None:
		colors = []
		idx = 0
		for pr in records:
			colors.append(default_colors[idx])
			idx += 1
			if idx >= len(default_colors):
				idx = 0

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


def randomRGBPure(hue=None):
    h = random.random() if hue is None else hue
    s = random.uniform(0.8,0.9)
    v = random.uniform(0.8,0.9)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r,g,b)