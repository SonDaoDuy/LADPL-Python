import numpy as np
from sklearn import preprocessing
from numpy.linalg import inv
import time
import math
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split

def normcol_lessequal(matin):
	return matin/ np.tile(np.sqrt(np.maximum(1,np.sum(matin*matin, axis=0))), (matin.shape[0], 1))

def normcol_equal(matin):
	return matin/ np.tile(np.sqrt(np.sum(matin*matin, axis=0)), (matin.shape[0], 1))

def find_all_index(arrayin,value):
	index_list = []
	for i in range(len(arrayin)):
		if arrayin[i] == value:
			index_list.append(i)
	return index_list

def update_D(Coef, Data, D_Mat):
	Dict = []
	class_num = len(Data)
	dim = len(Data[0])
	i_mat = np.identity(len(Coef[0]))
	#print("%s,%s" % (len(i_mat),len(i_mat[0])))

	for i in range(class_num):
		temp_coef = Coef[i]
		#print("%s,%s" % (len(temp_coef),len(temp_coef[0])))
		temp_data_mat = Data[i]
		#print("%s,%s" % (len(temp_data_mat),len(temp_data_mat[0])))
		rho = 1
		rate_rho = 1.2
		temp_s = D_Mat[i]
		#print("%s,%s" % (len(temp_s),len(temp_s[0])))
		s = (len(temp_s),len(temp_s[0]))
		temp_t = np.zeros(s)
		previous_d = D_Mat[i]
		count = 1
		error = 1
		#print("T: %s" % (i))
		while (error>1e-8 and count<100):
			mat1 = inv(np.multiply(rho,i_mat) + np.dot(temp_coef,temp_coef.transpose()))
			mat2 = np.multiply(rho,(temp_s-temp_t+np.dot(temp_coef,temp_data_mat)))
			temp_d = np.dot(mat1,mat2)
			#print("D :%s,%s" % (len(temp_d),len(temp_d[0])))
			temp_s = normcol_lessequal(temp_d + temp_t)
			#print("S: %s,%s" % (len(temp_s),len(temp_s[0])))
			temp_t = temp_t+temp_d+temp_s
			rho = rate_rho*rho
			hold = np.square(previous_d - temp_d)
			d_hold = []
			sum_ele = 0
			for k in range(len(hold)):
				for j in range(len(hold[k])):
					sum_ele += hold[k][j]
				d_hold.append(sum_ele/len(hold[k]))
				sum_ele = 0
			for k in range(len(d_hold)):
				sum_ele += d_hold[k]
			error = sum_ele/len(d_hold)
			previous_d = temp_d
			count += 1
		Dict.append(temp_d)
	return Dict
         
			

def update_P(Coef, DataInvMat, P_Mat, DataMat, tau):
	final_P = []
	class_num = len(Coef)
	for i in range(class_num):
		temp_p_mat = P_Mat[i]
		temp_coef = Coef[i]
		temp_data_mat = DataMat[i]
		temp_inv_mat = DataInvMat[i]
		temp_p_mat = np.dot(np.multiply(tau, np.dot(temp_coef,temp_data_mat)),temp_inv_mat)
		final_P.append(temp_p_mat)

	return final_P


def update_A(Dict, Data, P_Mat,tau, beta, DictSize, W_Mat, LabelMat):
	Coef = []
	class_num = len(Data)
	i_mat = np.identity(DictSize)
	for i in range(class_num):
		temp_dict_mat = Dict[i]
		temp_data_mat = Data[i]
		temp_w_mat = W_Mat[i]
		temp_label = LabelMat[i]
		temp_p_mat = P_Mat[i]
		temp_coef = np.dot(inv(np.dot(temp_dict_mat,temp_dict_mat.transpose()) + np.multiply(tau,i_mat) + np.multiply(beta,np.dot(temp_w_mat,temp_w_mat.transpose()))),
			(np.dot(temp_dict_mat,temp_data_mat.transpose()) + np.multiply(tau,np.dot(temp_p_mat,temp_data_mat.transpose())) + np.multiply(beta,np.dot(temp_w_mat,temp_label))))
		Coef.append(temp_coef)

	return Coef	

def update_W(Coef, Label, W_Mat, beta):
	final_W = []
	class_num = len(Label)
	dim = len(Label[0])
	i_mat = np.identity(len(Coef[0]))

	for i in range(class_num):
		temp_coef = Coef[i]
		temp_label = Label[i]
		first_term = np.divide(i_mat,np.dot(temp_coef,temp_coef.transpose()))
		second_term = np.dot(temp_coef,temp_label.transpose())
		temp_w_mat = np.dot(first_term,second_term)
		final_W.append(temp_w_mat)

	return final_W

def initialization(Data, Label, DictSize, tau, beta, lamda, gamma):
	result = []
	data_mat = []
	dict_mat = []
	data_inv_mat = []
	p_mat = []
	coef_mat = []
	w_mat = []
	label_mat = []

	class_num = int(max(Label)) + 1 #number of class
	dim = len(Data[0])	#dimension of a sample
	num_of_data = len(Data)	#number of sample
	i_mat = np.identity(dim)
	s = (class_num,num_of_data)
	h_mat = np.zeros(s)
	print len(h_mat)

	#form label matric for the data
	h_mat = h_mat.transpose()
	for i in range(class_num):
		h_mat[int(Label[i])][i] = 1
	h_mat = h_mat.transpose()
	h_mat = h_mat.astype(int)
	for i in range(class_num):
		# form the list of label matric
		index = np.bincount(h_mat[i])
		s = (class_num,index[1])
		temp_label = np.zeros(s)
		for j in range(index[1]):
			temp_label[i][j] = 1
		label_mat.append(temp_label)
		#form the list of data matric responsive to its label 
		#and the data inv mattric contains all datas that dont have the label
		index_list = find_all_index(h_mat[i],1)
		temp_inv_mat = []
		temp_data_mat = []
		for j in range(len(Data)):
			if j in index_list:
				temp_data_mat.append(Data[j])
			else:
				temp_inv_mat.append(Data[j])
		temp_data_mat = np.asarray(temp_data_mat)
		temp_inv_mat = np.asarray(temp_inv_mat)
		data_mat.append(temp_data_mat)
		
		#form dic_mat, p_mat, w_mat randomly
		temp_dict_mat = normcol_equal(np.random.randn(DictSize,dim))
		dict_mat.append(temp_dict_mat)
		temp_p_mat = normcol_equal(np.random.randn(DictSize,dim))
		p_mat.append(temp_p_mat)
		temp_w_mat = normcol_equal(np.random.randn(DictSize,class_num))
		w_mat.append(temp_w_mat)
		#form data_inv_mat
		data_inv_mat.append(inv(np.multiply(tau,np.dot(temp_data_mat.transpose(),temp_data_mat)) + np.multiply(lamda,np.dot(temp_inv_mat.transpose(),temp_inv_mat)) + np.multiply(gamma,i_mat)))

	coef_mat = update_A(dict_mat,data_mat,p_mat,tau,beta,DictSize,w_mat,label_mat)

	result.append(data_mat)
	result.append(dict_mat)
	result.append(p_mat)
	result.append(data_inv_mat)
	result.append(coef_mat)
	result.append(w_mat)
	result.append(label_mat)
	print("done initialize!")
	return result

def train_LADPL(Data, Label, DictSize, tau, beta, lamda, gamma):
	data_mat, dic_mat, p_mat, data_inv_mat, coef_mat, w_mat, label_mat = initialization(Data, Label, DictSize, tau, beta, lamda, gamma)
	for loop in range(30):
		p_mat = update_P(coef_mat,data_inv_mat,p_mat,data_mat,tau)
		dic_mat = update_D(coef_mat, data_mat, dic_mat)
		w_mat = update_W(coef_mat, label_mat, w_mat, beta)
		coef_mat = update_A(dic_mat,data_mat,p_mat,tau,beta,DictSize,w_mat,label_mat)

	encoder_mat = []
	for i in range(len(p_mat)):
		for j in range(len(p_mat[i])):
			encoder_mat.append(p_mat[i][j])

	result = []
	result.append(dic_mat)
	result.append(encoder_mat)
	result.append(w_mat)
	result.append(coef_mat)

	return result

def classification_LADPL(TestData, DictMat, EnconderMat, DictSize, W_mat):
	classify_mat = []
	class_num = len(DictMat)
	for i in range(len(W_mat)):
		for j in range(len(W_mat[i])):
			classify_mat.append(W_mat[i][j])

	classify_mat = np.asarray(classify_mat)
	projectile = np.dot(EnconderMat,TestData.transpose())
	predict = np.dot(classify_mat.transpose(),projectile)
	predict = predict.transpose()
	
	label_ouput = []
	for i in range(len(predict)):
		label = 0
		for j in range(len(predict[i])):
			if predict[i][j] > predict[i][label]:
				label = j
		label_ouput.append(label)

	return label_ouput



#load data
mat_content = sio.loadmat('infore.mat')
temp_data = mat_content['Data']

label = temp_data[0]
label = label - 1
data = temp_data[1:len(temp_data)]
data = data.transpose()

#split data 90% train, 10%test
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=0)

X_train = normcol_equal(X_train)
X_test =  normcol_equal(X_test)

#prameter setting
DictSize = 30
tau = 0.019
beta = 0.0135
lamda = 0.003
gamma = 0.0001

print("training......")
train_time = time.time()
DictMat, EnconderMat, W_mat, CoefMat = train_LADPL(X_train, y_train, DictSize, tau, beta, lamda, gamma)
print("train time: %s" % (time.time()-train_time))

print("testing....")
test_time = time.time()
PredictLabel = classification_LADPL(X_test, DictMat, EnconderMat, DictSize, W_mat)
print("testing time: %s" % (time.time()-test_time))

print y_test
print PredictLabel
print ("%s" % (y_test - PredictLabel))

			
