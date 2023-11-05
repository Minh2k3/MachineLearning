import customtkinter as cus
from tkinter import filedialog
from PIL import ImageTk, Image
from sklearn import datasets
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import cv2

def calculate_distance(p1,p2):
	dimension = len(p1)
	distance = 0

	for i in range(dimension):
		distance += (p1[i] - p2[i]) * (p1[i] - p2[i])

	return math.sqrt(distance) 

def get_k_neighbors(training_X, label_y, point, k):
	distances = []
	neighbors = []
	# Calculate distance from point to everything in training_X
	for i in range(len(training_X)):
		distance = calculate_distance(training_X[i], point)
		distances.append(distance)
	index = []
	while len(neighbors) < k:
		i = 0
		min_distance = 999999
		min_idx = 0
		while i < len(distances):
			if i in index:
				i+=1
				continue
			if distances[i] <= min_distance:
				min_distance = distances[i]
				min_idx = i
			i+=1
		index.append(min_idx)
		neighbors.append(label_y[min_idx])
	return neighbors

def highest_votes(labels):
	labels_count = [0,0,0,0,0,0,0,0,0,0]
	for label in labels:
		labels_count[label] += 1

	max_count = max(labels_count)
	return labels_count.index(max_count)

def predict(training_X, label_y, point, k):
	neighbors_labels = get_k_neighbors(training_X, label_y, point, k)
	return highest_votes(neighbors_labels)

def accuracy_score(predicts, labels):
	total = len(predicts)
	correct_count = 0
	for i in range(total):
		if predicts[i] == labels[i]:
			correct_count += 1

	accuracy = correct_count/total	

	return accuracy

digits = datasets.load_digits()
digits_X = digits.data # data 
digits_y = digits.target # label

randIndex = np.arange(digits_X.shape[0])
np.random.shuffle(randIndex)

digits_X = digits_X[randIndex]
digits_y = digits_y[randIndex]

X_train = digits_X[:1437,:] # 1437 training points
X_test = digits_X[1437:,:] # 360 testing points
y_train = digits_y[:1437] # 1437 labels of 1437 training points
y_test = digits_y[1437:] # 360 labels of 50 testing points

k=5
y_predict = []
for p in X_test:
	label = predict(X_train, y_train, p, k)
	y_predict.append(label)

def resize_frame(event):
    frame.configure(width=root.winfo_width() - 50, height=root.winfo_height() - 100)  # Cập nhật chiều rộng của frame
root = cus.CTk()
root.title("Giao diện học máy")
root.geometry("800x600")
topicLable = cus.CTkLabel(root, text="Phần mềm nhận dạng chữ viết", font=("Time New Roman", 20, "bold", ), text_color="#34bffa")
topicLable.pack(side="top", fill="both", pady=10)

def process_file():
    file_path = filedialog.askopenfilename()
    def open_file():
        if file_path:
            image = Image.open(file_path).resize((150,150))
            photo = ImageTk.PhotoImage(image)
            show_image.configure(image=photo)
            image_path.configure(text = "Đường dẫn ảnh:" + file_path)

    open_file()

    def predict_file():
        img = plt.imread(file_path)
        new_size = (8, 8)
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        binary_image = np.ceil((img / 255.0) * 16).astype(int)
        # binary_image = (binary_image > 3) * binary_image
        plt.gray()
        accuracy = accuracy_score(y_predict, y_test)
        # print('Predict img: ', predict(X_train, y_train, binary_image, k))
        predict_img.configure(text = "Predict img: " + str(predict(X_train, y_train, binary_image.reshape(1,-1)[0], k)))
        # plt.imshow(binary_image) 
        # plt.show() 

    predict_file()

frame = cus.CTkFrame(root)
frame.pack(padx=10, pady=10)

cus.CTkButton(frame, text="Chọn ảnh", command=process_file).pack(padx=10, pady=10)
show_image = cus.CTkLabel(frame, text = "")
show_image.pack(padx=10, pady=10)
image_path = cus.CTkLabel(frame, text="Đường dẫn ảnh:", font=("Time New Roman", 15, "bold"))
image_path.pack(padx=10, pady=10)

acc = accuracy_score(y_predict, y_test)
precision = precision_score(y_predict, y_test, average='macro')
recall = recall_score(y_predict, y_test, average='macro')

cus.CTkLabel(frame, text="Accuracy: " + str(acc), font=("Time New Roman", 15, "bold")).pack(padx=10, pady=5, anchor="w")
cus.CTkLabel(frame, text="Precision: " + str(precision), font=("Time New Roman", 15, "bold")).pack(padx=10, pady=5, anchor="w")
cus.CTkLabel(frame, text="Recall: " + str(recall), font=("Time New Roman", 15, "bold")).pack(padx=10, pady=5, anchor="w")
predict_img = cus.CTkLabel(frame, text="Predict img: ", font=("Time New Roman", 15, "bold"))
predict_img.pack(padx=10, pady=5, anchor="w")

root.bind("<Configure>", resize_frame)
root.mainloop()
