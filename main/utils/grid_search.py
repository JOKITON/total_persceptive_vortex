from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

def wide_prepro_grid(json_grid):
	grid = {}

	grid["csp__n_components"] = json_grid["csp__n_components_00"]
	# grid["pca__n_components"] = json_grid["pca__n_components_00"]

	return grid

def default_prepro_grid(json_grid):
	grid = {}

	grid["csp__n_components"] = json_grid["csp__n_components_01"]
	# grid["pca__n_components"] = json_grid["pca__n_components_01"]

	return grid

def narrow_prepro_grid(json_grid):
	grid = {}

	grid["csp__n_components"] = json_grid["csp__n_components_02"]
	# grid["pca__n_components"] = json_grid["pca__n_components_02"]

def wide_svm_grid(json_grid):
	grid = {}

	grid["svm__C"] = json_grid["svm__C_00"]
	grid["svm__gamma"] = json_grid["svm__gamma_00"]

	return grid

def default_svm_grid(json_grid):
	grid = {}

	grid["svm__C"] = json_grid["svm__C_01"]
	grid["svm__gamma"] = json_grid["svm__gamma_01"]

	return grid

def narrow_svm_grid(json_grid):
	grid = {}

	grid["svm__C"] = json_grid["svm__C_02"]
	grid["svm__gamma"] = json_grid["svm__gamma_02"]

	return grid

def vnarrow_svm_grid(json_grid):
	grid = {}

	grid["svm__C"] = json_grid["svm__C_00"]
	grid["svm__gamma"] = json_grid["svm__gamma_03"]

	return grid

def default_rf_grid(json_grid):
	grid = {}

	grid["rf__n_estimators"] = json_grid["rf__n_estimators_01"]
	grid["rf__max_depth"] = json_grid["rf__max_depth_01"]
	grid["rf__min_samples_split"] = json_grid["rf__min_samples_split_01"]

	return grid

def wide_rf_grid(json_grid):
	grid = {}

	grid["rf__n_estimators"] = json_grid["rf__n_estimators_00"]
	grid["rf__max_depth"] = json_grid["rf__max_depth_00"]
	grid["rf__min_samples_split"] = json_grid["rf__min_samples_split_00"]

	return grid
 
def grid_finder(json_grid, ml_type, grid_type):
	clf = None
	pipeline = Pipeline([])
	pre_grid, svm_grid, rf_grid = None, None, None
	if ml_type == 'preprocess':
		pipeline.steps.append(['clf', SVC()])
		if grid_type == 'default':
			pre_grid = default_prepro_grid(json_grid)
		elif grid_type == 'wide':
			pre_grid = wide_prepro_grid(json_grid)
		elif grid_type == 'narrow':
			pre_grid = narrow_prepro_grid(json_grid)
		else:
			print("Invalid grid type")
		return pre_grid, pipeline
	elif ml_type == 'svm':
		clf = SVC(kernel='rbf', C=100, gamma=2, probability=True)
		pipeline.steps.append(['svm', clf])
		if grid_type == 'default':
			svm_grid = default_svm_grid(json_grid)
		elif grid_type == 'wide':
			svm_grid = wide_svm_grid(json_grid)
		elif grid_type == 'narrow':
			svm_grid = narrow_svm_grid(json_grid)
		elif grid_type == 'vnarrow':
			svm_grid = vnarrow_svm_grid(json_grid)
		else:
			print("Invalid grid type")
		return svm_grid, pipeline
	elif ml_type == 'rf':
		clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
		pipeline.steps.append(['rf', clf])
		if grid_type == 'default':
			rf_grid = default_rf_grid(json_grid)
		elif grid_type == 'wide':
			rf_grid = wide_rf_grid(json_grid)
		else:
			print("Invalid grid type")
		return rf_grid, pipeline
	else:
		print("Invalid ML type")
	return None, None

def grid_search(X, y, pipeline, param_grid):
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	
	grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_weighted', verbose=0, error_score='raise')
	grid_search.fit(X, y)

	print("Best parameters:", grid_search.best_params_)
	print("Best accuracy:", grid_search.best_score_)