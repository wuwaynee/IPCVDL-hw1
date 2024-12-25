from fileinput import filename
from this import d
from tkinter import NONE
from PyQt5 import QtCore, QtGui, QtWidgets
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchsummary import summary
import torchvision.models as models

import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import random

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(421, 636)
        self.load_image_btn = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(1))
        self.load_image_btn.setGeometry(QtCore.QRect(60, 60, 301, 51))
        self.load_image_btn.setObjectName("load_image_btn")
        self.show_train_images_btn = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(2))
        self.show_train_images_btn.setGeometry(QtCore.QRect(60, 160, 301, 51))
        self.show_train_images_btn.setObjectName("show_train_images_btn")
        self.show_model_structure_btn = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(3))
        self.show_model_structure_btn.setGeometry(QtCore.QRect(60, 240, 301, 51))
        self.show_model_structure_btn.setObjectName("show_model_structure_btn")
        self.show_data_augmentation_btn = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(4))
        self.show_data_augmentation_btn.setGeometry(QtCore.QRect(60, 330, 301, 51))
        self.show_data_augmentation_btn.setObjectName("show_data_augmentation_btn")
        self.show_accuracy_and_loss_btn = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(5))
        self.show_accuracy_and_loss_btn.setGeometry(QtCore.QRect(60, 420, 301, 51))
        self.show_accuracy_and_loss_btn.setObjectName("show_accuracy_and_loss_btn")
        self.inference_btn = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(6))
        self.inference_btn.setGeometry(QtCore.QRect(60, 510, 301, 51))
        self.inference_btn.setObjectName("inference_btn")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.load_image_btn.setText(_translate("Form", "Load Image"))
        self.show_train_images_btn.setText(_translate("Form", "1. show train images"))
        self.show_model_structure_btn.setText(_translate("Form", "2. show model structure"))
        self.show_data_augmentation_btn.setText(_translate("Form", "3. show data augmentation"))
        self.show_accuracy_and_loss_btn.setText(_translate("Form", "4. show accuracy and loss"))
        self.inference_btn.setText(_translate("Form", "5. inference"))



    batch_size = 256
    learning_rate = 1e-2
    num_epoches = 50

    train_dataset = datasets.CIFAR10('./data', train = True,transform=transforms.ToTensor(), download = False)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataset = datasets.CIFAR10('./data', train = False,transform=transforms.ToTensor(), download = False)
    classes = train_dataset.classes # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    test_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    model = torchvision.models.vgg19()
    model.classifier[6] = nn.Linear(4096, 10)
    device = torch.device("cuda"  if torch.cuda.is_available()  else "cpu")
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    use_gpu = True


    # press the button function
    def press_it(self, data):
        if (data == 1): # load image
            self.filename, self.filetype = QtWidgets.QFileDialog.getOpenFileName(None, "Open file",".\\")      


        elif(data == 2):    # show train images
            # random
            for x in range(0, 9):
                i = random.randint(0, 1000)
                axs = plt.subplot(3, 3, x + 1)
                axs.imshow(self.train_dataset.data[i])
                axs.set_title(self.classes[self.train_dataset.targets[i]])
                axs.axis('off')
            plt.show()


        elif(data == 3):    # show model structure
            # input shape 32*32
            summary(self.model, (3, 32, 32))
            print(self.model)


        elif(data == 4):    # show data augmentation
            from PIL import Image
            img = Image.open(self.filename)
            plt.subplot(2, 3, 2)  
            plt.imshow(img)

            r = torchvision.transforms.Compose([torchvision.transforms.RandomRotation(degrees = 45)])
            a = r(img)
            plt.subplot(2 ,3, 4)
            plt.imshow(a)

            rc = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(80)])
            b = rc(img)
            plt.subplot(2, 3, 5)
            plt.imshow(b)

            hf = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip()])
            c = hf(img)
            plt.subplot(2, 3, 6)
            plt.imshow(c)

            plt.show()


        elif(data == 5):    # show accuracy and loss
            # pic
            acc = cv.imread("python_code/accuracy_loss.jpg")
            cv.imshow("accuracy and loss", acc)

            # train
            test_size = len(self.test_dataset)
            train_size = len(self.train_dataset)
            lost_y = []
            train_acc_y = []
            test_acc_y = []

            x_list = range(self.num_epoches)
            for epoch in range(self.num_epoches):
                running_loss = 0.0
                running_acc = 0.0
                self.model.train()
                for i,data in tqdm(enumerate(self.train_loader, 0)):
                    img,label = data
                    img = img.to(self.device)
                    label = label.to(self.device)
                    out = self.model(img)
                    loss = self.loss_function(out, label)
                    running_loss += loss.item() * label.size(0)

                    _,pred = torch.max(out, 1)
                    num_correct = (pred == label).sum()
                    accuracy = (pred == label).float().mean()
                    running_acc += num_correct.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.model.eval()
                train_acc_y.append(100 * running_acc / train_size)
                lost_y.append(running_loss / train_size)

                eval_loss = 0
                eval_acc = 0.0

                """
                for img,label in test_loader:
                    img = img.to(device)
                    label = label.to(device)
                    out = model(img)
                    _, pred = torch.max(out, 1)
                    num_correct = (pred==label).sum()
                    eval_acc += num_correct.item()
                test_acc_y.append(100 * eval_acc / test_size)
                """

            torch.save(self.model.state_dict(), './model.pth')

            fig, (ax1, ax2) = plt.subplots(2)
            
            ax2.set_xlabel('epoch')
            ax2.set_ylabel('loss')
            ax1.set_ylabel('accuracy(%)')
            ax1.plot(x_list, train_acc_y)
            ax2.plot(x_list, lost_y)


        elif(data == 6):    # inference
            #load any data
            from PIL import Image
            img = Image.open(self.filename)
            img = img.resize((32, 32))
            pixels = np.transpose(np.array(img), (2, 0, 1))
            pixels = np.float32((pixels / 255))
            i = torch.from_numpy(pixels)
            i = i.unsqueeze(0)
            self.model.load_state_dict(torch.load('model.pth', map_location = torch.device('cpu')))
            self.model.eval()
            prediction = nn.functional.softmax(self.model(i).to(self.device).data, dim = -1).numpy()[0]
            ax1 = plt.subplot()
            plt.axis('off')
            ax1.imshow(img)
            ax1.set_title('label: ' + str(self.classes[np.argmax(prediction)]) + '\n' + 'confidence: ' + str(np.max(prediction)))
            plt.show()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
