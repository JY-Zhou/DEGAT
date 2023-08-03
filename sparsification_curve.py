import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse as sp
import pandas as pd
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense,Reshape,GlobalMaxPool1D,MaxPool1D,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import activations, regularizers, constraints, initializers
import tensorflow.keras.backend as K
tf.keras.utils.set_random_seed(123)
tf.random.set_seed(123)


ppi=pd.read_csv('data/BRCA/ppi_network_of_BRCA.csv')
expression= pd.read_csv('data/BRCA/RPPA_data_of_BRCA.csv')
cancertype= pd.read_csv('data/BRCA/clinical_data_of_BRCA.csv')
cancertype=cancertype[['patient','BRCA_Subtype_PAM50']]
expression=expression.transpose()

expression.reset_index(inplace=True)
expression.columns = expression.iloc[0]
expression = expression[1:]
expression["Sample_description"] = expression["Sample_description"].apply(lambda x: x[:12])
expression



def Diffusion(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]

    A_loop = sp.eye(N) + A

    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

    # PPR-based diffusion
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)

    # Sparsify using threshold epsilon
    S_tilde = S.multiply(S >= eps)

    # Column-normalized transition matrix on graph S_tilde
    D_tilde_vec = S_tilde.sum(0).A1
    
    T_S = S_tilde / D_tilde_vec
    
    return T_S

data= cancertype.merge(expression, left_on='patient', right_on='Sample_description')
data.dropna()
columns=data.columns.tolist()[3:]
columns
','.join(columns)

data1=data[columns]
corr = data1.corr()
adjoint=corr.applymap(lambda x:0)

for i in range(0,len(ppi)):
    node1=ppi.iloc[i,0]
    node2=ppi.iloc[i,1]
    
    correlation = np.corrcoef(expression[node1].tolist(), expression[node2].tolist())[0, 1]
    correlation=abs(correlation)
    adjoint.loc[node1,node2]=correlation
    adjoint.loc[node2,node1]=correlation

adjoint

sparse_matrix1 = csr_matrix(adjoint.values)
x=Diffusion(sparse_matrix1,0.4,0.005)
df=pd.DataFrame(x)
x2=df.applymap(lambda x: 1 if x != 0 else 0)
x2

x0=adjoint.applymap(lambda x: 1 if x != 0 else 0)
sns.heatmap(x0, cmap='Blues', cbar_kws={'label': 'Value'})
plt.title('Adjacency Matrix before diffusion')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()

sns.heatmap(x2, cmap='Blues', cbar_kws={'label': 'Value'})
plt.title('Adjacency Matrix after diffusion')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()


plt.rcParams['font.family'] = ['Microsoft YaHei']

def calculate_y(alpha,x):
    x=Diffusion(sparse_matrix1,alpha,x)
    df=pd.DataFrame(x)
    for i in range(0,len(df)):
        df.iloc[i,i]=0
    non_zero_count = df[df != 0].count().sum()
    return non_zero_count

def plot_curve(x_values, y_values, title="", x_label="", y_label=""):
    plt.figure()
    plt.plot(x_values, y_values)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

start_exp = -7  
end_exp = -1    
num_points = 100  

x_values = np.logspace(start_exp, end_exp, num=num_points)


y_values1 = [calculate_y(0.2,x) for x in x_values]
y_values2 = [calculate_y(0.4,x) for x in x_values]
y_values3 = [calculate_y(0.6,x) for x in x_values]
y_values4 = [calculate_y(0.8,x) for x in x_values]

plt.semilogx(x_values, y_values1, label='α=0.2')
plt.semilogx(x_values, y_values2, label='α=0.4')
plt.semilogx(x_values, y_values3, label='α=0.6')
plt.semilogx(x_values, y_values4, label='α=0.8')

plt.grid(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.xlabel("ε(log scale)")
plt.ylabel("number of edges")
plt.legend()
plt.show()
