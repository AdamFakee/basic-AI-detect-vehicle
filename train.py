import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import os
from sklearn.preprocessing import LabelEncoder


classes = 4 # có 4 loại phương tiện 
cur_path = os.getcwd() # lấy đường dẫn đang đứng để run file 

# nhãn 
labelFolder = {
    0: 'bus', 1: 'car', 2: 'truck', 3: 'van'
}



#Retrieving the images and their labels 
def retrievDataFromFolde (length = classes, folderName='train'):
    data = []
    labels = []
    for i in range(length):
        path = os.path.join( cur_path, folderName, labelFolder[i] )
        images = os.listdir(path)
        for a in images:
            try:
                image = Image.open(path + '\\'+ a) # mở file ảnh 
                image = image.resize((30,30)) # resize về cùng kích thước 
                image = np.array(image) # chuyển ảnh thành mảng pixel dạng số 
                data.append(image) # thêm vào mảng data
                labels.append(i) # lable 
            except:
                print("Error loading image")
    # chuyển sang dạng numpy 
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

dataTrain, labelTrain = retrievDataFromFolde()

# split to test and train 
X_train, X_test, y_train, y_test = train_test_split(dataTrain, labelTrain, test_size=0.2, random_state=classes) # x = data, y = lable 


# Converting the labels into one hot encoding
y_train = to_categorical(y_train, classes) # chuyển về dạng mảng bit gì gì đấy 
y_test = to_categorical(y_test, classes)


# #Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:])) # convolutional layer: 32 lớp 5*5 
model.add(MaxPool2D(pool_size=(2, 2))) # maxpooling lấy đặc trưng + giảm kích thước input đầu vào cho các lớp sau 
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')) # 64 lớp filter 3*3: lọc đặc trưng chi tiết 
model.add(MaxPool2D(pool_size=(2, 2))) # nt
model.add(Dropout(rate=0.3)) # dropout layer: bỏ ngẫu nhiên 25% input vừa nhận => còn 75% làm input đầu ra 
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')) # nt 
model.add(MaxPool2D(pool_size=(2, 2))) # nt

model.add(Flatten()) # mảng 2 chiều => mảng 1 chiều ( làm phẳng mảng để fullConnected layer sử dụng )
model.add(Dense(256, activation='relu')) # fullyconnected Layer : 256 neural - node 
model.add(Dropout(rate=0.4)) # nt 
model.add(Dense(classes, activation='softmax')) # fullyconnected Layer + softmax để lấy xác suất đầu ra, lớn nhất => chọn 

# run 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # đại khái là hàm mất mát, hàm tối ưu "adam", cơ chế dự đoán tính chính xác 

epochs = 12 # chạy n vòng 
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")





# model.save("my_model.keras")
model.save("txxxxxx.h5")
