import imutils
from imutils import paths
import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.models import load_model
import os
import shutil
import pickle








class NoCaptcha():

    def build(self, model_path='NoCaptcha\\captcha.hdf5'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.model = Sequential()

        self.model.add(Conv2D(20, (5,5), padding='same', input_shape=(20,20,1), activation='relu'))
        self.model.add( MaxPooling2D(pool_size=(2,2), strides=(2,2)) )

        self.model.add( Conv2D(50, (5,5), padding='same', activation='relu' ) )
        self.model.add( MaxPooling2D(pool_size=(2,2), strides=(2,2)) )

        self.model.add( Flatten() )
        self.model.add( Dense(500, activation='relu') )

        self.model.add( Dense(26, activation='softmax') )

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')    

        self.trained_model = load_model(model_path)  

    def crack_captcha(self, captcha_path, model_path='NoCaptcha\\captcha.hdf5'):
        """
        Function to decipher a captcha image with a Convolutional Neural Network

        :param captcha_path: directory path to the captcha image
        :type captcha_path: str
        :param model_path: path to the trained Convolutional NN, defaults to 'captcha.hdf5'
        :type model_path: str, optional
        :return: Letters predicted in the captcha image by the Convolutional NN
        :rtype: str
        """        

        if not os.path.exists("NoCaptcha\\Captchas"):
            os.mkdir('NoCaptcha\\Captchas')

        if not os.path.exists("NoCaptcha\\RawLetters"):
            os.mkdir('NoCaptcha\\RawLetters')

        captchas_len = len(os.listdir("NoCaptcha\\Captchas"))+1
        extension = captcha_path[captcha_path.find("."):]
        shutil.copy(captcha_path, f'NoCaptcha\\Captchas\\{captchas_len}{extension}')

        letters = self.split_letters(captcha_path, 'NoCaptcha\\RawLetters')

        answer = ''

        for letter_img in letters:
            
            with open('NoCaptcha\\nocapthca_label_binarizer.pkl','rb') as label_enc:
                label_enc = pickle.load(label_enc)

            model = load_model(model_path)

            letter_img = self.resize_to_fit(letter_img, 20, 20)

            letter_img = np.expand_dims(letter_img, axis=2)
            letter_img = np.expand_dims(letter_img, axis=0)

            captcha = model.predict(letter_img)

            captcha = label_enc.inverse_transform(captcha)[0]

            answer += captcha

        return answer

    def resize_to_fit(self, image, width, height):
        """
        A helper function to resize an image to fit within a given size

        :param image: image to resize
        :param width: desired width in pixels
        :param height: desired height in pixels
        :return: the resized image
        """

        # grab the dimensions of the image, then initialize
        # the padding values
        (h, w) = image.shape[:2]

        # if the width is greater than the height then resize along
        # the width
        if w > h:
            image = imutils.resize(image, width=width)

        # otherwise, the height is greater than the width so resize
        # along the height
        else:
            image = imutils.resize(image, height=height)

        # determine the padding values for the width and height to
        # obtain the target dimensions
        padW = int((width - image.shape[1]) / 2.0)
        padH = int((height - image.shape[0]) / 2.0)

        # pad the image then apply one more resizing to handle any
        # rounding issues
        image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
            cv2.BORDER_REPLICATE)
        image = cv2.resize(image, (width, height))

        # return the pre-processed image
        return image

    def train(self, X_train, X_test, Y_train, Y_test, path_for_model=None,batch_size=26, epochs=7, verbose=1): 
        """
        Train the convolutional neural network to read captchas, data should be preprocessed to function with Keras

        :param X_train: training data
        :type X_train: np.array
        :param Y_train: training labels
        :type Y_train: np.array
        :param X_test: test data
        :type X_test: np.array
        :param Y_test: test labels
        :type Y_test: np.array
        :param path_for_model: path to where the user wishes to save the trained model, defaults to None
        :type path_for_model: str, optional
        """        
        self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=epochs, verbose=verbose)

        
        self.model.save(path_for_model)

    def split_letters(self, captcha_path, save_data=False):
        """
        Tries to find the letters in the captcha image and return them as image matrices, inside in iterable list of letters.
        List of letter matrices is ordered by the position of the letters in the captcha image from left to right.

        :param captcha_path: Path to the captcha image
        :type captcha_path: str
        :param save_data: Option to save the letters extracted from the captcha image
        :type save_data: bool, optional
        :raises RuntimeError: Error for when the recursive function that searches for letters reaches recursive limit, meaning it didn't find possible letters
        :return: List of image matrices of the letters, ordered as they appear on the captcha image, and a list of paths to the respective letter images if save_data=True
        :rtype: list, list
        """        

        if save_data:
            if not os.path.exists("NoCaptcha\\RawLetters"):
                os.mkdir('NoCaptcha\\RawLetters')
        
        method = cv2.THRESH_TRUNC

        img = cv2.imread(f'{captcha_path}')
        
        img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        (h, w) = img_grey.shape[:2]
        img_grey = cv2.resize(img_grey, (w*3, h*3))
        del img

        _, treated_img = cv2.threshold(img_grey, 127, 255, method + cv2.THRESH_OTSU)

        _, treated_img = cv2.threshold(treated_img, 115, 255, cv2.THRESH_BINARY_INV)
        binary_img = np.array(treated_img)

       

        
        contours , _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        def find_contours(thresh):
            regions = []
            

            for contour in contours:
                (x, y, width, height) = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                if area > thresh:
                    regions.append((x, y, width, height))


            if len(regions) == 5:
                return regions
                
            else:
                if len(regions)>5: return find_contours(thresh+5)
                    
                else: return find_contours(thresh-5)
                

            

        
            
            

        try: regions = find_contours(115)
        except: raise RuntimeError('Failed to find contours in captcha image, check if image has letters in it')

        regions = sorted(regions, key= lambda list : list[0])
        
         
        letters=[]
        paths = []

        for rect in regions:
            x, y, width, height = rect
            letter_img = binary_img[y-5:y+height+5, x-5:x+width+5]
            
            (h, w) = letter_img.shape[:2]
            letter_img = cv2.resize(letter_img, (w*2, h*2))
            
            


            letter_img = cv2.GaussianBlur(letter_img, (5,5), 0)

            letter_img = 255 - letter_img
            letters.append(letter_img)

            if save_data:
                raw_letters = len(os.listdir('NoCaptcha\\RawLetters'))


                img_path= 'NoCaptcha\\RawLetters'+f'\\XYZ{raw_letters+1}.png'
                cv2.imwrite(img_path, letter_img)
                paths.append(img_path)

        
        
        if save_data: return letters, paths
        else: return letters

    def create_dataset(self, letters_location, dataset_width, dataset_height):
        """Creates a dataset for the Captcha-Reading model, recieves a path with letters labeled in folders, also recieves the desired height and width to fit
        the images to.

        :param letters_location: path to letter_labeled folders with letter images inside
        :type letters_location: str
        :param dataset_width: image width desired for the whole dataset that will be made
        :type dataset_width: int
        :param dataset_height: image height desired for the whole dataset that will be made
        :type dataset_height: int
        :return: list of training data (X_train, X_test, Y_train, Y_test)
        :rtype: list
        """        

        #List of training images 
        data = []

        #List of labels
        labels = []

        #List of full paths to all our labeled data. This way we have the path to all our labeled images
        # and, since the images are in labeled folders, we get our labels for them automatically.
        images = paths.list_images(letters_location)

        for file in images:
            #Splitting the path to get our labels
            label = file.split(os.path.sep)[2]

            #Reading image
            image = cv2.imread(file)

            #Converting image to GrayScale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            #Using our resize function to resize the image
            image = self.resize_to_fit(image, dataset_width, dataset_height)

            #Creating 3rd dimension because Keras needs it, remembering that GrayScale images have 2 dimensions
            image = np.expand_dims(image, axis=2)

            #Appending labels and images to their lists
            labels.append(label)
            data.append(image)

        #Normalizing our data to range from 0 to 1 for better model performance, also making it into an array
        data = np.array(data, dtype='float') / 255

        #Making labels into array
        labels = np.array(labels)
        
        #Splitting our data into Train and Test set
        (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

        #Creating a label binarizer for One Hot Encoding our labels
        label_binarizer = LabelBinarizer().fit(Y_train)

        #Transforming our labels to one hot encode
        Y_train = label_binarizer.transform(Y_train)
        Y_test = label_binarizer.transform(Y_test)

        #Saving our binarizer
        with open('NoCaptcha\\nocapthca_label_binarizer.pkl', 'wb') as handle:
            pickle.dump(label_binarizer, handle)

        #Show length of training data
        print('TRAIN SIZE: '+str(len(X_train)))

        #Returning dataset
        return X_train, X_test, Y_train, Y_test




class NoCaptchaGPU(NoCaptcha):

    def build(self, model_path='NoCaptcha\\captchaGPU.hdf5'):
        print('_________BUILDING GPU MODEL___________')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.model = Sequential()

        self.model.add(Conv2D(100, (5,5), padding='same', input_shape=(100,100,1), activation='relu'))
        self.model.add( MaxPooling2D(pool_size=(2,2), strides=(2,2)) )

        self.model.add( Conv2D(200, (5,5), padding='same', activation='relu' ) )
        self.model.add( MaxPooling2D(pool_size=(2,2), strides=(2,2)) )

        self.model.add( Flatten() )
        self.model.add( Dense(500, activation='relu') )

        self.model.add( Dense(26, activation='softmax') )

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        with open('NoCaptcha\\nocapthca_label_binarizer.pkl','rb') as label_enc:
                self.label_encoder = pickle.load(label_enc)

        try:self.trained_model = load_model(model_path)
        except: pass
        
        
    def crack_captcha(self, captcha_path, model_path='NoCaptcha\\captchaGPU.hdf5', save_data=False):
        """
        Function to decipher a captcha image with a Convolutional Neural Network

        :param captcha_path: directory path to the captcha image
        :type captcha_path: str
        :param model_path: path to the trained Convolutional NN, defaults to 'captcha.hdf5'
        :type model_path: str, optional
        :param save_path: option to save or not all the output images made in the process of cracking captchas
        :type save_path: bool, optional
        :return: Letters predicted in the captcha image by the Convolutional NN
        :rtype: str
        """        

        if save_data:
            if not os.path.exists("NoCaptcha\\Captchas"):
                os.mkdir('NoCaptcha\\Captchas')

        if save_data:
            if not os.path.exists("NoCaptcha\\RawLetters"):
                os.mkdir('NoCaptcha\\RawLetters')

        captchas_len = len(os.listdir("NoCaptcha\\Captchas"))+1
        extension = captcha_path[captcha_path.find("."):]
        shutil.copy(captcha_path, f'NoCaptcha\\Captchas\\{captchas_len}{extension}')

        if save_data: letters, paths = self.split_letters(captcha_path, save_data=save_data)
        else: letters = self.split_letters(captcha_path)

        answer = ''

        

        for letter_img in letters:
            
            

            letter_img = self.resize_to_fit(letter_img, 100, 100)

            letter_img = np.expand_dims(letter_img, axis=2)
            letter_img = np.expand_dims(letter_img, axis=0)

            captcha = self.trained_model.predict(letter_img)

            captcha = self.label_encoder.inverse_transform(captcha)[0]

            answer += captcha

        

       
        if save_data: return answer, paths
        else: return answer

    
if __name__ == '__main__':
    
    '''NC = NoCaptchaGPU()
    NC.build(model_path=None)
    X_train, X_test, Y_train, Y_test = NC.create_dataset('NoCaptcha\\Letters',100,100)
    NC.train(X_train, X_test, Y_train, Y_test, path_for_model='NoCaptcha\\captchaGPU.hdf5', epochs=6)'''