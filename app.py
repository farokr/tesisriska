import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


@st.cache
def getdata(csv,separator=',') -> pd.DataFrame:
    return pd.read_csv(csv,sep=separator)
    
    
def main():
    st.title("Klasifikasi Kelulusan Mahasiswa")
    st.header("Eksplorasi Data")

    source_df = getdata('data_mahasiswa.csv',separator=';')
    print()
    st.subheader("Sumber Data "+str(source_df.shape))
    #if st.checkbox("Show Source Data"):
    st.write(source_df.sample(10)) 
    
    df = getdata('x1.csv')
    st.subheader("Data Hasil Proses "+str(df.shape))
    st.write(df)
    
    fig1= plt.figure()
    sns.countplot(x ='Label1', data=df)
    plt.title('Tepat waktu ? \n (0: Tidak || 1: Ya)', fontsize=14)
    st.write(fig1)
    st.write("Data Tidak Seimbang (Imbalanced Data)")
    
    df2 = getdata('x2.csv')
    st.subheader("Data Hasil Proses Menggunakan SMOTE "+str(df2.shape))
    st.write(df2)
    
    
    fig2= plt.figure()
    sns.countplot(x ='Label1', data=df2)
    plt.title('Tepat waktu ? \n (0: Tidak || 1: Ya)', fontsize=14)
    st.write(fig2)
    st.write("Data Sudah Seimbang (Balanced Data)")
    
    
    X = df2.drop('Label1',axis=1)
    y = df2['Label1']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=99,stratify=y)
    X2_train, X2_test, y2_train, y2_test = train_test_split(df2,df2['Label1'], test_size=0.1, random_state=99,stratify=df2['Label1'])
    
    
    st.subheader("Data Test "+str(X2_test.shape))
    st.write(X2_test)
    
    
    st.subheader("Klasifikasi Dengan Decision Tree")
    
    model = DecisionTreeClassifier(max_depth = 4).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = round(metrics.accuracy_score(y_test, y_pred),4)*100
    
    st.write("Accuracy: "+str(acc)+"%")
    cm2 = metrics.confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix :")
    st.write(cm2)
    
    
    st.subheader("Klasifikasi Dengan Naive Bayes")
    model = GaussianNB().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = round(metrics.accuracy_score(y_test, y_pred),4)*100
    
    st.write("Accuracy: "+str(acc)+"%")
    cm2 = metrics.confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix :")
    st.write(cm2)
    
    st.subheader("Klasifikasi Dengan KNN n_neighbor=3")
    jumlah_n = 3 #penentuan awal
    model =  KNeighborsClassifier(n_neighbors=jumlah_n).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = round(metrics.accuracy_score(y_test, y_pred),4)*100
    
    st.write("Accuracy: "+str(acc)+"%")
    cm2 = metrics.confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix :")
    st.write(cm2)
    
    st.subheader("Menentukan Nilai n_neighbor Optimal")
    neighbors = np.arange(1, 11)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
  
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)
  
    fig3= plt.figure()
    plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
    
    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    st.write(fig3)
    st.write("Nilai Optimal n_neighbor: 5")
    
    
    st.subheader("Klasifikasi Dengan KNN")
    jumlah_n  = st.slider('Nilai K (2-10)', min_value=2, max_value=10, step=1, value=5)
    st.subheader("Nilai n_neighbor="+str(jumlah_n))
    st.subheader("Klasifikasi Dengan KNN n="+str(jumlah_n))
    model2 =  KNeighborsClassifier(n_neighbors=jumlah_n).fit(X_train, y_train)
    y2_pred = model2.predict(X_test)
    
    acc = round(metrics.accuracy_score(y_test, y2_pred),4)*100
    
    st.write("Accuracy: "+str(acc)+"%")
    cm = metrics.confusion_matrix(y_test, y2_pred)
    st.write("Confusion Matrix :")
    st.write(cm)
    
    
main()

