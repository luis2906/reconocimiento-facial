#OpenCV module
import cv2
#Modulo para leer directorios y rutas de archivos
import os
#OpenCV trabaja con arreglos de numpy
import numpy
#Se importa la lista de personas con acceso al laboratorio
from listaPermitidos import flabianos
flabs=flabianos()

# Parte 1: Creando el entrenamiento del modelo
#Directorio donde se encuentran las carpetas con las caras de entrenamiento
dir_faces = 'att_faces/orl_faces'
#Tamaño para reducir a miniaturas las fotografias
size = 4

# Crear una lista de imagenes y una lista de nombres correspondientes
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(dir_faces):
	for subdir in dirs:
		names[id] = subdir
		subjectpath = os.path.join(dir_faces, subdir)
		for filename in os.listdir(subjectpath):
			path = subjectpath + '/' + filename
			lable = id
			images.append(cv2.imread(path, 0))
			lables.append(int(lable))
		id += 1
(im_width, im_height) = (112, 92)

# Crear una matriz Numpy de las dos listas anteriores
(images, lables) = [numpy.array(lis) for lis in [images, lables]]
# OpenCV entrena un modelo a partir de las imagenes
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)


# Parte 2: Utilizar el modelo entrenado en funcionamiento con la camara
face_cascade = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

age_net = cv2.dnn.readNetFromCaffe(
		'data/deploy_age.prototxt', 
		'data/age_net.caffemodel')

gender_net = cv2.dnn.readNetFromCaffe(
		'data/deploy_gender.prototxt', 
		'data/gender_net.caffemodel')


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

while True:
	#leemos un frame y lo guardamos
	rval, frame = cap.read()
	frame=cv2.flip(frame,1,0)

	#convertimos la imagen a blanco y negro    
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#redimensionar la imagen
	mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

	"""buscamos las coordenadas de los rostros (si los hay) y
		guardamos su posicion"""
	faces = face_cascade.detectMultiScale(mini)

	for i in range(len(faces)):
		face_i = faces[i]
		(x, y, w, h) = [v * size for v in face_i]

		# Obteniendo rostro
		face_img = frame[y:y+h, h:h+w].copy()
		blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

		face = gray[y:y + h, x:x + w]
		face_resize = cv2.resize(face, (im_width, im_height))

		# Intentado reconocer la cara
		prediction = model.predict(face_resize)

		#Dibujamos un rectangulo en las coordenadas del rostro
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
		
		roi_gray = gray[y:y + h, x:x + w]
		roi_color = frame[y:y + h, x:x + w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

		#Predcir Genero
		gender_net.setInput(blob)
		gender_preds = gender_net.forward()
		gender = gender_list[gender_preds[0].argmax()]

		#Predecir Edad
		age_net.setInput(blob)
		age_preds = age_net.forward()
		age = age_list[age_preds[0].argmax()]

		# Escribiendo el nombre de la cara reconocida
		# La variable cara tendra el nombre de la persona reconocida
		cara = '%s' % (names[prediction[0]])

		#Si la prediccion tiene una exactitud menor a 100 se toma como prediccion valida
		if prediction[1]<100 :
			#Ponemos el nombre de la persona que se reconoció
			cv2.putText(frame,'%s : %.0f , %s , %s' % (cara,prediction[1], " Edad : "+str(age), "Gender:"+str(gender) ),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

			#En caso de que la cara sea de algun conocido se realizara determinadas accione          
			#Busca si los nombres de las personas reconocidas estan dentro de los que tienen acceso          
			flabs.valida_invitado(cara)

		#Si la prediccion es mayor a 100 no es un reconomiento con la exactitud suficiente
		elif prediction[1]>101 and prediction[1]<500:           
			#Si la cara es desconocida, poner desconocido
			cv2.putText(frame, 'Desconocido',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))  

		#Mostramos la imagen
		cv2.imshow('OpenCV Reconocimiento facial', frame)

	#Si se presiona la tecla ESC se cierra el programa
	key = cv2.waitKey(10)
	if key == 27:
		cv2.destroyAllWindows()
		break
