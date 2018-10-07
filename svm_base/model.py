from sklearn import svm, metrics, model_selection
import pickle
from load_img import *

def train(X_train, Y_train, model, filename=0):
	print("Training")
	model.fit(X_train, Y_train)
	if filename is 1:
		pickle.dump(model, open(filename, 'wb'))

	return model

def evaluate(model, X_test, Y_test):
	print("Evaluating")
	preds = model.predict(X_test)
	print(preds)
	cr = metrics.classification_report(Y_test, preds)
	cmx = metrics.confusion_matrix(Y_test, preds)
	print(cr)
	print(cmx)

def main():
	TEST_PERCENT = 0.1
	ATTRIBUTE = 'Target'
	ANNO = '../dataset/stage_1_train_labels.csv'
	TRAIN_DIR = '../dataset/sample_data/'

	print('Loading Train Data')
	X, Y = load_data_into_memory(DIR=TRAIN_DIR, ANNO=ANNO, ATTRIBUTE=ATTRIBUTE)

	X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
	print(X.shape)

	print("Splitting data")
	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=TEST_PERCENT, random_state=51, shuffle=True)
	print(X_train.shape, 'training data')
	print(X_test.shape, 'test data')

	model = svm.SVC()

	evaluate(train(X_train, Y_train, model, "svm_model.pkl"), X_test, Y_test)

main()
