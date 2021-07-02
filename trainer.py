import cv2,os
import numpy as np
from PIL import Image

# membuat recognizer LBPH dengan fungsi yang sudah disediakan
# oleh OpenCV dan memasukkan ke variabel recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# membuat detector dengan "faceclassifier.xml"
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
	# mengambil path dari semua file yang ada di folder dataset
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
	# membuat array sampel wajah
    faceSamples=[]
	# membuat array untuk setiap ID
    Ids=[]
	# looping ke semua imagePaths lalu memuat Id mahasiswa 
	# dan foto dari mahasiswa tersebut
    for imagePath in imagePaths:
		# memuat gambar dan merubah ke dalam bentuk gray scale
        pilImage=Image.open(imagePath).convert('L')
		# merubah PIL image ke dalam bentuk numpy array
        imageNp=np.array(pilImage,'uint8')
		# mengambil Id dari nama gambar
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
		# meng-ekstrak wajah dari sampel training gambar
        faces=detector.detectMultiScale(imageNp)
		# menempelkan hasil ekstraksi data ke dalam array
		# Ids sesuai dengan gambar dari Id yang sedang di-training
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
	#mengembalikan nilai dari variabel
	#faceSamples dan Ids
    return faceSamples,Ids

# memanggil fungsi getImagesAndLabels dengan argumen
# nama folder dataset
faces,Ids = getImagesAndLabels('dataset')
# melakukan training data dengan fungsi train()
recognizer.train(faces, np.array(Ids))
recognizer.write('recognizer/trainer.yml')
