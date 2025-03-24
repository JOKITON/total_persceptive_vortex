import numpy as np


def prt_scores(train_score, test_score, scores):
	print('Train score: {:.4f}'.format(train_score))
	print('Test score: {:.4f}'.format(test_score))
	print('Cross-Validation Mean: {}'.format(np.mean(scores)))
