import pandas as pd

def perf_measure(y_actual, y_hat):
	TP = 0
	FP = 0
	TN = 0
	FN = 0

	for i in range(len(y_hat)): 
		if y_actual[i]==y_hat[i]==1:
		   TP += 1
		if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
		   FP += 1
		if y_actual[i]==y_hat[i]==0:
		   TN += 1
		if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
		   FN += 1

	fpr = FP/(FP+TN)
	fnr = FN/(FN+TP)
	acc = (TP+TN)/(TP+TN+FP+FN)
	return(TP, FP, TN, FN, fpr, fnr,acc)

# Import Folds
fold0 = pd.read_csv('0.csv')
fold1 = pd.read_csv('1.csv')
fold2 = pd.read_csv('2.csv')
fold3 = pd.read_csv('3.csv')
fold4 = pd.read_csv('4.csv')
folds = [fold0,fold1,fold2,fold3,fold4]

models = fold0.columns[:-1].tolist()

for model in models:
	df = pd.DataFrame(columns=['fold', 'fnr', 'fpr', 'acc'])
	for i,fold in enumerate(folds):
		predlist = fold[model].tolist()
		real = fold['Real'].tolist()
		_,_,_,_,fpr,fnr,acc = perf_measure(real,predlist)
		df = df.append({'fold': float(i), 'fnr': fnr, 'fpr': fpr, 'acc': acc}, ignore_index=True)
	df.to_csv('results/'+model+'.csv',index=True)