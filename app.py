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
from imblearn.over_sampling import SMOTE
import base64
import io


arr_map = {1:'TEPAT WAKTU', 0:'TERLAMBAT'}

#pemrosesan data mentah menjadi data siap di proses
def proses_data(df):
    #dropping kolom
    df = df.drop(['no', 'NIM', 'nama','program_studi','ipk','total_sks'],axis =1)
    
    #pengkodean data text menjadi kode angka
    df['jk'] = df['jk'].str.strip().replace({'L': 0, 'P': 1,})
    df['kelulusan'] = df['kelulusan'].str.strip().replace({'TERLAMBAT': 0, 'TEPAT WAKTU': 1})
    df['pekerjaan'] = df['pekerjaan'].str.strip().replace({'KARYAWAN': 0, 'MAHASISWA': 1})
    
    #mengubah notasi koma menjadi notasi titik
    df['ips_1'] = df['ips_1'].apply(lambda x: x.replace(',','.'))
    df['ips_2'] = df['ips_2'].apply(lambda x: x.replace(',','.'))
    df['ips_3'] = df['ips_3'].apply(lambda x: x.replace(',','.'))
    df['ips_4'] = df['ips_4'].apply(lambda x: x.replace(',','.'))
    df['ips_5'] = df['ips_5'].apply(lambda x: x.replace(',','.'))
    df['ips_6'] = df['ips_6'].apply(lambda x: x.replace(',','.'))
    df['ips_7'] = df['ips_7'].apply(lambda x: x.replace(',','.'))
    
    return df

def get_table_download_link(df,txt = 'Download hasil Perhitungan',filename = "datahasil.xlsx"):
    towrite = io.BytesIO()
    df.to_excel(towrite, encoding='utf-8', index=False, header=True, engine='xlsxwriter')
    towrite.seek(0)  # reset pointer
    #csv = df.to_csv(index=False,sep=';')
    b64 = base64.b64encode(towrite.read()).decode()
    href= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{txt}</a>'

    #href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download file hasil clustering</a>'
    return href
#end of get_table_download_link


@st.cache
def getdata(csv,separator=';') -> pd.DataFrame:
    return pd.read_csv(csv,sep=separator)
    
    
def eda():
    
    st.header("Eksplorasi Data")

    source_df = getdata('DATA_MAHASISWA.csv',separator=';')
    #df = proses_data(source_df)
    df = source_df.copy()
    
    st.subheader("Sumber Data "+str(source_df.shape))
    #if st.checkbox("Show Source Data"):
    st.write(source_df) 
    
    
    #dropping kolom
    df = df.drop(['no', 'NIM', 'nama','program_studi','ipk','total_sks'],axis =1)
    st.subheader("Penghilangan Kolom"+str(df.shape))
    st.write(df) 
    
    #pengkodean data text menjadi kode angka
    df['jk'] = df['jk'].str.strip().replace({'L': 0, 'P': 1,})
    df['kelulusan'] = df['kelulusan'].str.strip().replace({'TERLAMBAT': 0, 'TEPAT WAKTU': 1})
    df['pekerjaan'] = df['pekerjaan'].str.strip().replace({'KARYAWAN': 0, 'MAHASISWA': 1})
    
    #mengubah notasi koma menjadi notasi titik
    df['ips_1'] = df['ips_1'].apply(lambda x: x.replace(',','.'))
    df['ips_2'] = df['ips_2'].apply(lambda x: x.replace(',','.'))
    df['ips_3'] = df['ips_3'].apply(lambda x: x.replace(',','.'))
    df['ips_4'] = df['ips_4'].apply(lambda x: x.replace(',','.'))
    df['ips_5'] = df['ips_5'].apply(lambda x: x.replace(',','.'))
    df['ips_6'] = df['ips_6'].apply(lambda x: x.replace(',','.'))
    df['ips_7'] = df['ips_7'].apply(lambda x: x.replace(',','.'))
    st.subheader("Pengkodean Kolom Teks"+str(df.shape))
    st.write(df) 
    
    st.subheader("Pemeriksaan Keseimbangan Kolom Kelulusan")
    fig1= plt.figure()
    sns.countplot(x ='kelulusan', data=df)
    plt.title('0: TERLAMBAT || 1: TEPAT WAKTU', fontsize=14)
    st.write(fig1)
    st.write("Data Tidak Seimbang (Imbalanced Data)")
    
    
    #balancing the data using SMOTE
    df2 = pd.DataFrame()
    sm = SMOTE()
    df2, df2['kelulusan'] = sm.fit_resample(df.drop('kelulusan',axis=1), df['kelulusan'])
    
    st.subheader("Data Hasil Proses Menggunakan SMOTE "+str(df2.shape))
    plt.title('0: TERLAMBAT || 1: TEPAT WAKTU', fontsize=14)
    sns.countplot(x ='kelulusan', data=df2)
    st.write(df2)
    
    
    fig2= plt.figure()
    sns.countplot(x ='kelulusan', data=df2)
    plt.title('0: TERLAMBAT || 1: TEPAT WAKTU', fontsize=14)
    st.write(fig2)
    st.write("Data Sudah Seimbang (Balanced Data)")
    
def dt(df,df1,df2):
    st.header("Decision Tree")
     
    dfm1 = df.copy()
    #parameter  = st.slider('Maximal Depth', min_value=2, max_value=10, step=1, value=4)
    parameter  = st.number_input("Maximal Depth", value=4)
    parameter = int(parameter)
    
    model = DecisionTreeClassifier(max_depth = parameter).fit(df2.drop('kelulusan',axis=1), df2['kelulusan'])
    ypred = model.predict(df1.drop('kelulusan',axis=1))
    
    dfm1['prediksi'] = ypred
    dfm1['prediksi'] = dfm1['prediksi'].map(arr_map)
    
    recall1 = '{:.2f}%'.format(metrics.recall_score(dfm1['kelulusan'], dfm1['prediksi'],pos_label='TEPAT WAKTU' )*100)
    recall2 = '{:.2f}%'.format(metrics.recall_score(dfm1['kelulusan'], dfm1['prediksi'],pos_label='TERLAMBAT' )*100) 
    precision1 = '{:.2f}%'.format(metrics.precision_score(dfm1['kelulusan'], dfm1['prediksi'],pos_label='TEPAT WAKTU')*100)
    precision2 = '{:.2f}%'.format(metrics.precision_score(dfm1['kelulusan'], dfm1['prediksi'],pos_label='TERLAMBAT' )*100)
    
    accuracy = '{:.2f}%'.format(metrics.accuracy_score(dfm1['kelulusan'], dfm1['prediksi'])*100)
    
    st.subheader('Classification Report')
    st.write("Accuracy:",accuracy)
    st.write("Recall TEPAT WAKTU :",recall1)
    st.write("Recall TERLAMBAT :",recall2)
    st.write("Precision TEPAT WAKTU :",precision1)
    st.write("Precision TERLAMBAT :",precision2)
        
    st.subheader('Tree Visualization')
    fig = plt.figure(figsize=(80,40))
    _ = tree.plot_tree(model,
                       feature_names=['jk','ips_1', 'ips_2', 'ips_3',
       'ips_4', 'ips_5', 'ips_6', 'ips_7', 'ipk', 'sks_1', 'sks_2', 'sks_3',
       'sks_4', 'sks_5', 'sks_6', 'sks_7', 'pekerjaan'],  
                       class_names=['TERLAMBAT','TEPAT WAKTU'],
                       filled=True)
    st.pyplot(fig,clear_figure=True,dpi=100)
    
    st.subheader('Confusion Matrix')
    fig= plt.figure()
    confusionmatrix = pd.crosstab(dfm1['kelulusan'], dfm1['prediksi'], rownames=['Aktual'], colnames=['Prediksi'])
    sns.heatmap(confusionmatrix, annot=True, fmt='g',cmap='Blues')
    st.write(fig)
    
    st.subheader('Hasil Perhitungan')
             
    res1 =  dfm1.loc[dfm1['prediksi'] == 'TERLAMBAT']
    res2 =  dfm1.loc[dfm1['prediksi'] == 'TEPAT WAKTU']
    
    
    st.subheader('Semua Prediksi :'+str(dfm1.shape[0]))      
    st.write(dfm1)

    st.subheader('Prediksi TERLAMBAT :'+str(res1.shape[0]))      
    st.write(res1)       

    st.subheader('Prediksi TEPAT WAKTU:'+str(res2.shape[0]))      
    st.write(res2)
             
    st.markdown(get_table_download_link(dfm1,filename = "datahasilDecisionTree.xlsx"), unsafe_allow_html=True)
    
    
    
#end of dt()
     
def nb(df,df1,df2):
    st.header("Naive Bayes")
    dfm1 = df.copy()
    
    model = GaussianNB().fit(df2.drop('kelulusan',axis=1), df2['kelulusan'])
    ypred = model.predict(df1.drop('kelulusan',axis=1))
    dfm1['prediksi'] = ypred
    dfm1['prediksi'] = dfm1['prediksi'].map(arr_map)

    
    recall1 = '{:.2f}%'.format(metrics.recall_score(dfm1['kelulusan'], dfm1['prediksi'],pos_label='TEPAT WAKTU' )*100)
    recall2 = '{:.2f}%'.format(metrics.recall_score(dfm1['kelulusan'], dfm1['prediksi'],pos_label='TERLAMBAT' )*100) 
    precision1 = '{:.2f}'.format(metrics.precision_score(dfm1['kelulusan'], dfm1['prediksi'],pos_label='TEPAT WAKTU' )*100)
    precision1 = '{:.2f}%'.format(metrics.precision_score(dfm1['kelulusan'], dfm1['prediksi'],pos_label='TEPAT WAKTU' )*100)
    precision2 = '{:.2f}%'.format(metrics.precision_score(dfm1['kelulusan'], dfm1['prediksi'],pos_label='TERLAMBAT' )*100)
    accuracy = '{:.2f}%'.format(metrics.accuracy_score(dfm1['kelulusan'], dfm1['prediksi'])*100)
   
    st.subheader('Classification Report')
    st.write("Accuracy:",accuracy)
    st.write("Recall TEPAT WAKTU :",recall1)
    st.write("Recall TERLAMBAT :",recall2)
    st.write("Precision TEPAT WAKTU :",precision1)
    st.write("Precision TERLAMBAT :",precision2)
    
    st.subheader('Confusion Matrix')
    fig= plt.figure()
    confusionmatrix = pd.crosstab(dfm1['kelulusan'], dfm1['prediksi'], rownames=['Aktual'], colnames=['Prediksi'])
    sns.heatmap(confusionmatrix, annot=True, fmt='g',cmap='Blues')
    st.write(fig)
    st.subheader('Hasil Perhitungan')
    
    res1 =  dfm1.loc[dfm1['prediksi'] == 'TERLAMBAT']
    res2 =  dfm1.loc[dfm1['prediksi'] == 'TEPAT WAKTU']
    
    
    st.subheader('Semua Prediksi :'+str(dfm1.shape[0]))     
    st.write(dfm1)

    st.subheader('Prediksi TERLAMBAT :'+str(res1.shape[0]))      
    st.write(res1)       

    st.subheader('Prediksi TEPAT WAKTU:'+str(res2.shape[0]))      
    st.write(res2)
    
    st.markdown(get_table_download_link(dfm1,filename = "datahasilNaiveBayes.xlsx"), unsafe_allow_html=True)
    
#END OF NB

def knn(df,df1,df2):
    st.header("KNN")
    dfm1 = df.copy()
    
    st.subheader("Menentukan Nilai n_neighbor Optimal")
    neighbors = np.arange(2, 11)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
  
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k).fit(df2.drop('kelulusan',axis=1), df2['kelulusan'])
        train_accuracy[i] = knn.score(df2.drop('kelulusan',axis=1), df2['kelulusan'])
        test_accuracy[i] = knn.score(df1.drop('kelulusan',axis=1), df1['kelulusan'])
  
    fig3= plt.figure()
    plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
    
    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    st.write(fig3)
    
    #parameter  = st.slider('Nilai K', min_value=2, max_value=10, step=1, value=3)
    parameter  = st.number_input("Nilai K", value=3)
    parameter = int(parameter)
    
    model =  KNeighborsClassifier(n_neighbors=parameter).fit(df2.drop('kelulusan',axis=1), df2['kelulusan'])
    ypred = model.predict(df1.drop('kelulusan',axis=1))
    
    dfm1['prediksi'] = ypred
    dfm1['prediksi'] = dfm1['prediksi'].map(arr_map)

    recall1 = '{:.2f}%'.format(metrics.recall_score(dfm1['kelulusan'], dfm1['prediksi'],pos_label='TEPAT WAKTU' )*100)
    recall2 = '{:.2f}%'.format(metrics.recall_score(dfm1['kelulusan'], dfm1['prediksi'],pos_label='TERLAMBAT' )*100) 
    precision1 = '{:.2f}%'.format(metrics.precision_score(dfm1['kelulusan'], dfm1['prediksi'],pos_label='TEPAT WAKTU' )*100)
    precision2 = '{:.2f}%'.format(metrics.precision_score(dfm1['kelulusan'], dfm1['prediksi'],pos_label='TERLAMBAT' )*100)
    accuracy = '{:.2f}%'.format(metrics.accuracy_score(dfm1['kelulusan'], dfm1['prediksi'])*100)
    st.subheader('Classification Report')
    st.write("Accuracy:",accuracy)
    st.write("Recall TEPAT WAKTU :",recall1)
    st.write("Recall TERLAMBAT :",recall2)
    st.write("Precision TEPAT WAKTU :",precision1)
    st.write("Precision TERLAMBAT :",precision2)
    st.subheader('Confusion Matrix')
    fig= plt.figure()
    confusionmatrix = pd.crosstab(dfm1['kelulusan'], dfm1['prediksi'], rownames=['Aktual'], colnames=['Prediksi'])
    sns.heatmap(confusionmatrix, annot=True, fmt='g',cmap='Blues')
    st.write(fig)
    st.subheader('Hasil Perhitungan')
    res1 =  dfm1.loc[dfm1['prediksi'] == 'TERLAMBAT']
    res2 =  dfm1.loc[dfm1['prediksi'] == 'TEPAT WAKTU']
    
    
    st.subheader('Semua Prediksi :'+str(dfm1.shape[0]))     
    st.write(dfm1)

    st.subheader('Prediksi TERLAMBAT :'+str(res1.shape[0]))      
    st.write(res1)       

    st.subheader('Prediksi TEPAT WAKTU:'+str(res2.shape[0]))      
    st.write(res2)
    
    st.markdown(get_table_download_link(dfm1,filename = "datahasiKNN.xlsx"), unsafe_allow_html=True)
#END OF KNN     

def apps():
    st.header("Aplikasi Perhitungan")
    data = st.file_uploader("Upload a Dataset", type=["csv"])
    if data is not None:
        
        
        df = pd.read_csv(data,sep=';')
        st.subheader("Sumber Data "+str(df.shape))
        st.dataframe(df)
        
        df1 = proses_data(df)
        st.subheader("Data Hasil Proses "+str(df1.shape))
        st.dataframe(df1)
        
        sm = SMOTE()
        df2 = pd.DataFrame()
        df2, df2['kelulusan'] = sm.fit_resample(df1.drop('kelulusan',axis=1), df1['kelulusan'])
        
        
        choice = st.radio("Pilih Metode",['Decision Tree','Naive Bayes','KNN'])
        if choice == 'Decision Tree':
            dt(df,df1,df2)
        elif choice == 'Naive Bayes': 
            nb(df,df1,df2)
        elif choice == 'KNN':
            knn(df,df1,df2)
            
         
            
#end of apps
    
    
def main():
    st.sidebar.title("Klasifikasi Kelulusan Mahasiswa")
    activities = ['Eksplorasi Data','Aplikasi Perhitungan']
    choice = st.sidebar.selectbox("Select Activities",activities)
    
    if choice == 'Eksplorasi Data':
        eda()
    elif choice == 'Aplikasi Perhitungan':
        apps()
        

if __name__ == '__main__':
    main()

