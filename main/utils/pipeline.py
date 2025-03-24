from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def crt_pipeline(clf=False, voting='soft', clf_type='svm'):
	""" Creates a pipeline for classification tasks """

	svm_clf = SVC(kernel='rbf', C=250, gamma=0.025, probability=True)
	mlp_clf = MLPClassifier()
	if clf is True:
		ensemble = VotingClassifier(estimators=[('svm', svm_clf), ('mlp', mlp_clf)], voting=voting)
		# ensemble = VotingClassifier(estimators=[('svm', svm_clf)], voting=voting)
		pipeline = Pipeline([
			('voting_cs', ensemble)
		])
	else:
		if clf_type == 'svm':
			pipeline = Pipeline([
				('svm', svm_clf)
			])
		elif clf_type == 'mlp':
			pipeline = Pipeline([
				('mlp', mlp_clf)
			])
		elif clf_type == 'lda':
			pipeline = Pipeline([
				('lda', LDA())
			])
		elif clf_type == 'rf':
			pipeline = Pipeline([
				('rf', RandomForestClassifier(max_depth=25, min_samples_split=15, n_estimators=150))
			])
		else:
			raise(ValueError('Invalid classifier type'))

	return pipeline


