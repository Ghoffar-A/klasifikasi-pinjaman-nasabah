import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from PIL import Image
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#image = Image.open(".\Final Decision Tree.png")

# memanggil model
loaded_model = pickle.load(
    open('.\model.sav', 'rb'))

def prediksinasabah(baru):
    hasilPrediksi = loaded_model.predict(baru)
    if (hasilPrediksi == 0):
        return'Pengajuan Pinjaman DITERIMA'
    if (hasilPrediksi == 1):
        return'Pengajuan Pinjaman DITOLAK'

st.title('Klasifikasi Pemberian Pinjaman Kepada Nasabah Bank Menggunakan Metode Decision Tree')

credit_dataset = pd.read_csv(".\credit_card_approval.csv")
st.subheader('Dataset Yang Digunakan')
st.write(credit_dataset)
# if st.checkbox('Tampilkan Dataset Yang Digunakan'):
#     st.subheader('Dataset')
#     st.write(credit_dataset)

# mencaritahu proporsi unik di kolom atribut
a = credit_dataset.TARGET.value_counts()
# if st.checkbox('Proporsi Unik Kolom Atribut'):
st.write('Perbandingan Class Pada Dataset')
st.write(a)

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
# if st.button('Ubah Huruf Menjadi Angka'):
#     st.write('Pengubahan Berhasil')

# if st.checkbox('Menampilkan Hasil Label Encoder'):
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
#if st.button('Membagi Dataset Menjadi Data Latih Dan Data Uji'):
    #st.write('Pembagian Berhasil')

# if st.checkbox('Menampilkan Data Latih dan Data Uji'):
st.subheader('Membagi Data Latih dan Data Uji')
st.write('Data Latih Fitur dan Target')
st.write(X_train, y_train)
st.write('Data Uji Fitur dan Target')
st.write(X_test, y_test)

#mendefinisikan model Tree
clf = tree.DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None, min_samples_split=2, 
                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                                  random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                  class_weight=None, ccp_alpha=0.0)
# if st.button('Mempersiapkan Model Tree'):
#     st.write('Persiapan Pembuatan Model Selesai')

#melakukan training model
clf = clf.fit(X_train, y_train)
# if st.button("Pelatihan Data latih"):
#     st.write('Model Telah Didapatkan')

#melakukan ploting model training
a = tree.plot_tree(clf)

# if st.checkbox('Menampilkan Hasil Pemodelan'):
#     st.write(a)

#Menampilkan gambar
# st.subheader('Model Hasil Pelatihan')
# st.image(image)

# prediksi data uji
hasilPrediksi = clf.predict(X_test)
# if st.checkbox('Pengujian Model Menggunakan Data Uji'):
#     st.write('Pengujian Berhasil')

#Menampilkan hasil prediksi
# if st.checkbox('Hasil Pengujian'):
st.subheader('Pengujian Model Menggunakan Data Uji')
col1, col2 = st.columns(2)
with col1:
    st.write('Label Sebenarnya', y_test)
with col2:
    st.write('Hasil Prediksi', hasilPrediksi)

# membuat confusion matrix
cm = confusion_matrix(y_test, hasilPrediksi)

#mendefinisikan index confusion matrix
cm_df = pd.DataFrame(cm,
                     index = ['DIterima','TidakDIterima'],
                     columns = ['DIterima','TidakDIterima'])

# Plotting the confusion matrix
#if st.button("Menampilkan Plot Yang Berkorelasit"):
st.subheader('Menampilkan Hasil Ploting Yang Berkorelasi')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("Heatmap")
fig, ax = plt.subplots(figsize=(5, 4))
st.write(sns.heatmap(cm_df, annot=True, fmt='d'))
st.pyplot()
# model report
target = ['Diterima', 'Ditolak']
st.write(classification_report(y_test, hasilPrediksi, target_names=target))

def main():
    st.header(' Pengujian model dengan menggunakan data baru: ')
    st.caption('Jangan lupa membaca caption dalam pengisian :)')
    # membuat form input data baru
    CODE_GENDER = st.text_input(' Jenis kelasmin : ')
    st.caption('0 untuk  perempuan dan 1 untuk laki-laki')
    FLAG_OWN_CAR = st.text_input(' Kepemiliki mobil: ')
    st.caption('0 untuk tidak memiliki dan 1 untuk memiliki mobil')
    FLAG_OWN_REALTY = st.text_input(' Kepemilikan rumah: ')
    st.caption('0 untuk tidak memiliki dan 1 untuk memiliki rumah')
    CNT_CHILDREN = st.text_input(' Jumlah anak yang dimiliki nasabah : ')
    st.caption('0 untuk 1 anak, 1 untuk 2+ anak dan 2 untuk tidak memiliki anak')
    AMT_INCOME_TOTAL = st.text_input(' Berapa penghasilan nasabah pertahun: ')
    st.caption('(Perhitungan dalam bentuk dolar)')
    NAME_EDUCATION_TYPE = st.text_input(' Pendidikan terakhir nasabah: ')
    st.caption('0 = Gelar stata 1 atau dibawahnya')
    st.caption('1 = Gelar Strata 2 atau lebih')
    st.caption('2 = Sedang menjalani pendididkan tingkat universitas')
    st.caption('3 = Lulus SMP ')
    st.caption('4 = Lulus SMA / SMK')
    NAME_FAMILY_STATUS = st.text_input(' Status pernikahan :')
    st.caption('0 = Pernikahan sesuai hukum negara')
    st.caption('1 = Sudah menikah')
    st.caption('2 = Cerai')
    st.caption('3 = Tidak/belum menikah')
    st.caption('4 = Janda')
    NAME_HOUSING_TYPE = st.text_input(' Jenis tempat tinggal : ')
    st.caption('0 = Menyewa bersama orang lain')
    st.caption('1 = Memiliki rumah atau apartement')
    st.caption('2 = Municipal apartment')
    st.caption('3 = Tinggal di kantor')
    st.caption('4 = Menyewa rumah')
    st.caption('5 = Dengan orang tua')
    DAYS_BIRTH = st.text_input(' Umur nasabah: ')
    st.caption('Dihitung dari dia lahir (tahun*(-365))')
    DAYS_EMPLOYED = st.text_input(' Berapa lama telah bekerja:')
    st.caption('Dihitung dalam bentuk hari, -lama bekerja dan 0 tidak bekerja')
    FLAG_WORK_PHONE = st.text_input(' Kepemilikan telfon kantor: ')
    st.caption('0 untuk tidak dan 1 untuk ya')
    FLAG_PHONE = st.text_input(' Kepeemilikan telfon rumah:')
    st.caption('0 untuk tidak dan 1 untuk ya')
    FLAG_EMAIL = st.text_input(' Kepemilikan email:')
    st.caption('0 untuk tidak dan 1 untuk ya')
    JOB = st.text_input(' Apa pekerjaan nasabah:')
    BEGIN_MONTHS = st.text_input(' Lama sudah menjadii nasabah:')
    st.caption('0 untuk nasabah terdaftar bulan ini, -1 untuk bulan sebelumnya dan seterusnya')
    STATUS = st.text_input(' Status pinjaman nasabah bulan ini: ')
    st.caption('0 = memiliki tunggakan antara 1-29 hari')
    st.caption('1 = memiliki tunggakan antara 30-59 hari')
    st.caption('2 = memiliki tunggakan antara 60-89 hari')
    st.caption('3 = memiliki tunggakan antara 90-119 hari')
    st.caption('4 = memiliki tunggakan antara 120-149 hari')
    st.caption('5 = memiliki tunggakan lebih dari 150 hari')
    st.caption('6 = Pinjaman lunas bulan ini')
    st.caption('7 = Tidak memiliki pinjaman bulan ini')
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

