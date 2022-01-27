import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# memanggil model
loaded_model = pickle.load(
    open('D:\streamlit\model.sav', 'rb'))


def prediksinasabah(baru):
    hasilPrediksi = loaded_model.predict(baru)
    if (hasilPrediksi == 0):
        return'Pengajuan Pinjaman DITERIMA'
    if (hasilPrediksi == 1):
        return'Pengajuan Pinjaman DITOLAK'

st.title('Klasifikasi Pemberian Pinjaman Kepada Nasabah Menggunakan Metode Decision Tree')

credit_dataset = pd.read_csv("D:\streamlit\credit_card_approval.csv")
if st.checkbox('Tampilkan Dataset Yang Digunakan'):
    st.subheader('Dataset')
    st.write(credit_dataset)

# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
credit_dataset['CODE_GENDER'] = labelencoder.fit_transform(credit_dataset['CODE_GENDER'])
credit_dataset['FLAG_OWN_CAR'] = labelencoder.fit_transform(credit_dataset['FLAG_OWN_CAR'])
credit_dataset['FLAG_OWN_REALTY'] = labelencoder.fit_transform(credit_dataset['FLAG_OWN_REALTY'])
credit_dataset['STATUS'] = labelencoder.fit_transform(credit_dataset['STATUS'])
credit_dataset['CNT_CHILDREN'] = labelencoder.fit_transform(credit_dataset['CNT_CHILDREN'])
credit_dataset['NAME_EDUCATION_TYPE'] = labelencoder.fit_transform(credit_dataset['NAME_EDUCATION_TYPE'])
credit_dataset['NAME_FAMILY_STATUS'] = labelencoder.fit_transform(credit_dataset['NAME_FAMILY_STATUS'])
credit_dataset['NAME_HOUSING_TYPE'] = labelencoder.fit_transform(credit_dataset['NAME_HOUSING_TYPE'])
credit_dataset['JOB'] = labelencoder.fit_transform(credit_dataset['JOB'])
if st.button('Ubah Huruf Menjadi Angka'):
    st.write('Pengubahan Berhasil')

if st.checkbox('Menampilkan Hasil Label Encoder'):
    st.subheader('Hasil Label Encoder')
    st.write(credit_dataset)

#feature engineering (mengambil atribut yang ingin digunakan)
X=credit_dataset[['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','NAME_EDUCATION_TYPE',
                  'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','DAYS_BIRTH','DAYS_EMPLOYED','FLAG_WORK_PHONE','FLAG_PHONE',
                  'FLAG_EMAIL','JOB','BEGIN_MONTHS','STATUS']]
# memisahkan Feature dengan class
y = credit_dataset.TARGET
# membagi dari total data menjadi 70% data Latih dan 30% data Uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
if st.button('Membagi Dataset Menjadi Data Latih Dan Data Uji'):
    st.write('Pembagian Berhasil')

if st.checkbox('Menampilkan Data Latih dan Data Uji'):
    st.subheader('Data Latih Fitur dan Target')
    st.write(X_train, y_train)
    st.subheader('Data Uji Fitur dan Target')
    st.write(X_test, y_test)


# mencaritahu proporsi unik di kolom atribut
a = credit_dataset.TARGET.value_counts()
if st.checkbox('Proporsi Unik Kolom Atribut'):
    st.write(a)

#mendefinisikan model Tree
clf = tree.DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None, min_samples_split=2, 
                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                                  random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                  class_weight=None, ccp_alpha=0.0)
if st.button('Mempersiapkan Model Tree'):
    st.write('Persiapan Pembuatan Model Selesai')

#melakukan training model
clf = clf.fit(X_train, y_train)
if st.button("Pelatihan Data latih"):
    st.write('Model Telah Didapatkan')

#melakukan ploting model training
a = tree.plot_tree(clf)
if st.checkbox('Menampilkan Hasil Pemodelan'):
    st.write(a)

# prediksi data uji
hasilPrediksi = clf.predict(X_test)
if st.checkbox('Pengujian Model Menggunakan Data Uji'):
    st.write('Pengujian Berhasil')

#Menampilkan hasil prediksi
if st.checkbox('Hasil Pengujian'):
    st.write('Label Sebenarnya', y_test)
    st.write('Hasil Prediksi', hasilPrediksi)

# membuat confusion matrix
cm = confusion_matrix(y_test, hasilPrediksi)

#mendefinisikan index confusion matrix
cm_df = pd.DataFrame(cm,
                     index = ['DIterima','TidakDIterima'],
                     columns = ['DIterima','TidakDIterima'])

# Plotting the confusion matrix
if st.button("Menampilkan Plot Yang Berkorelasit"):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("### Heatmap")
    fig, ax = plt.subplots(figsize=(5, 4))
    st.write(sns.heatmap(cm_df, annot=True, fmt='d'))
    st.pyplot()
# model report
    target = ['Diterima', 'Ditolak']
    st.write(classification_report(y_test, hasilPrediksi, target_names=target))

def main():
    # membuat form input data baru
    CODE_GENDER = st.text_input('Nilai Bahasa Inggris A')
    FLAG_OWN_CAR = st.text_input('Ni')
    FLAG_OWN_REALTY = st.text_input('Nil')
    CNT_CHILDREN = st.text_input('Nilai k')
    AMT_INCOME_TOTAL = st.text_input('Nilai B')
    NAME_EDUCATION_TYPE = st.text_input('Nilai Bah')
    NAME_FAMILY_STATUS = st.text_input('Nilai Ba')
    NAME_HOUSING_TYPE = st.text_input('Nilai Bahi')
    DAYS_BIRTH = st.text_input('Nilai Bi')
    DAYS_EMPLOYED = st.text_input('Nilai Bahas')
    FLAG_WORK_PHONE = st.text_input('Nilai Bahasa')
    FLAG_PHONE = st.text_input('Nilai Bahasa In')
    FLAG_EMAIL = st.text_input('Nilai Bahasa I')
    JOB = st.text_input('Nilai Bahasa yutj')
    BEGIN_MONTHS = st.text_input('Nilai Bahasa Ingg')
    STATUS = st.text_input('Nilai Bahasa Inggris')
    # prediksi
    prediksi = ''

    # membuat  button untuk prediksi
    if st.button('Prediksi Nasabah'):
        prediksi = prediksinasabah([[CODE_GENDER,FLAG_OWN_CAR,FLAG_OWN_REALTY,CNT_CHILDREN,AMT_INCOME_TOTAL,NAME_EDUCATION_TYPE,
                                    NAME_FAMILY_STATUS,NAME_HOUSING_TYPE,DAYS_BIRTH,DAYS_EMPLOYED,FLAG_WORK_PHONE,FLAG_PHONE,
                                    FLAG_EMAIL,JOB,BEGIN_MONTHS,STATUS]])
        st.success(prediksi)


if __name__ == '__main__':
    main()
