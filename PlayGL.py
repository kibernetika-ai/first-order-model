from PyQt5.QtCore import *  # QTimer
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtOpenGL import *  # QGLWidget
import PyQt5.QtOpenGL as QtOpenGL
from OpenGL.GL import *  # OpenGL functionality
from OpenGL.GL import shaders  # Utilities to compile shaders, we may not actually use this
import cv2
from scipy.spatial import Delaunay
import numpy as np
import threading
import dlib
# this is the basic window

gVShader = """
             attribute vec4 position;
             attribute vec2 texture_coordinates;   
             varying vec4 dstColor;
             varying vec2 v_texture_coordinates;

            void main() {    
                v_texture_coordinates = texture_coordinates;
                gl_Position = position;    
            }"""

gFShader = """   
            uniform sampler2D texture1;
            varying vec2 v_texture_coordinates;

            void main() {

                gl_FragColor = texture2D(texture1, v_texture_coordinates);
            }"""

def shape_to_array(sh,sx,sy):
    points = sh.parts()
    return np.array([[p.x/sx, p.y/sy] for p in points], dtype=float)

frame_size = (420, 240)
class OpenGLView(QGLWidget):

    def init_triangulate(self):
        points = []
        points.append([0, 0])
        points.append([0.5, 0.0])
        points.append([1, 0.0])
        points.append([1, 0.5])
        points.append([1, 1])
        points.append([0.5, 1])
        points.append([0, 1])
        for i in range(self.target.shape[0]):
            points.append([self.target[i, 0], self.target[i, 1]])
        self.points = points
        self.delaunay = Delaunay(np.array(points))
        self.simplices = self.delaunay.simplices

    def get_triangulation(self,move=None):
        triangles = []
        index1 = 0
        index2 = 69
        for i in range(self.simplices.shape[0]):
            for k in range(self.simplices.shape[1]):
                index = self.simplices[i, k]
                p = self.points[index]
                if move is None:
                    triangles.extend([p[0]*2-1,-1*(p[1]*2-1),p[0],p[1]])
                else:
                    index = index-7
                    if index>=index1 and index<index2:
                        x = p[0]+move[index,0]
                        x = max(x,0)
                        x = min(x,1)
                        y = p[1] + move[index, 1]
                        y = max(y, 0)
                        y = min(y, 1)
                        triangles.extend([x * 2 - 1, -1*(y * 2 - 1), p[0], p[1]])
                    else:
                        triangles.extend([p[0] * 2 - 1, -1*(p[1] * 2 - 1), p[0], p[1]])
        data = np.array(triangles,np.float32)
        return data


    def initializeGL(self):
        self.move = None
        self._lock = threading.Lock()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

        self.__timer = QTimer()
        self.__timer.timeout.connect(self.repaint)  # make it repaint when triggered
        self.__timer.start(1000/30)  #

        # set the RGBA values of the background
        glClearColor(0.1, 0.2, 0.3, 1.0)
        # set a timer to redraw every 1/60th of a second


        ##target = [(253, 405), (233, 405), (204, 398), (182, 381), (164, 362), (145, 334), (133, 307), (122, 271), (106, 220), (123, 191), (139, 167), (161, 161), (201, 166), (232, 181), (248, 201), (250, 277), (251, 255), (250, 229), (250, 298), (260, 298), (272, 291), (231, 295), (239, 298), (169, 189), (205, 190), (155, 212), (218, 213), (175, 238), (206, 237), (280, 210), (294, 188), (332, 190), (300, 238), (335, 232), (344, 214), (370, 183), (264, 180), (290, 165), (324, 160), (351, 166), (272, 402), (395, 214), (299, 394), (317, 381), (341, 353), (359, 325), (371, 294), (381, 255), (243, 336), (249, 335), (257, 336), (244, 341), (251, 341), (258, 341), (209, 335), (216, 336), (289, 333), (282, 335), (250, 346), (264, 344), (273, 341), (224, 341), (236, 342), (248, 332), (261, 331), (273, 330), (233, 332), (222, 333)]
        target = [(125, 224), (134, 245), (139, 266), (149, 298), (161, 325), (178, 350), (198, 371), (220, 385), (247, 389), (278, 385), (303, 373), (323, 353), (340, 325), (353, 299), (361, 269), (367, 246), (378, 220), (145, 206), (169, 190), (189, 186), (215, 192), (230, 204), (269, 201), (285, 193), (313, 186), (338, 194), (352, 208), (250, 224), (250, 244), (250, 264), (250, 279), (232, 296), (243, 298), (249, 300), (257, 298), (266, 297), (167, 217), (184, 210), (205, 211), (224, 225), (205, 238), (180, 238), (274, 225), (294, 210), (316, 209), (332, 217), (323, 235), (295, 239), (210, 324), (222, 320), (237, 320), (248, 322), (259, 319), (275, 318), (289, 322), (277, 338), (262, 342), (251, 346), (238, 344), (222, 338), (218, 326), (234, 327), (248, 328), (270, 327), (283, 325), (264, 335), (250, 337), (235, 335)]
        img = cv2.imread('iAvatar.png')[:, :, ::-1]
        avatar = cv2.resize(img,frame_size)
        self.target = np.array([[p[0] / img.shape[1], p[1] / img.shape[0]] for p in target],np.float32)
        self.init_triangulate()
        vshader = shaders.compileShader(gVShader, GL_VERTEX_SHADER)
        fshader = shaders.compileShader(gFShader, GL_FRAGMENT_SHADER)
        self._program = shaders.compileProgram(vshader, fshader)
        glUseProgram(self._program)
        # data array (2 [position], 2 [texture coordinates])

        #data = np.array([-1.0, -1.0, 0.0, 1.0,
        #                 1.0, -1.0, 1.0, 1.0,
        #                 -1.0, 1.0, 0.0, 0.0,
        #                1.0, 1.0, 1.0, 0.0],
        #                dtype=np.float32)

        #data = np.array([-1.0, -1.0, 0.0, 1.0,
        #                 -1.0, 1.0, 0.0, 0.0,
        #                 1.0, -1.0, 1.0, 1.0,
        #                 -1.0, 1.0, 0.0, 0.0,
        #                 1.0, 1.0, 1.0, 0.0,
        #                 1.0, -1.0, 1.0, 1.0
        #                 ],
        #                dtype=np.float32)
        # create a buffer and bind it to the 'data' array
        data = self.get_triangulation()

        self.bufferID = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.bufferID)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        # tell OpenGL how to handle the buffer of data that is already on the GPU

        loc = glGetAttribLocation(self._program,"position")
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, 2, GL_FLOAT, False, 16, ctypes.c_void_p(0))

        loc = glGetAttribLocation(self._program, "texture_coordinates")
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, 2, GL_FLOAT, False, 16, ctypes.c_void_p(8))

        self._imageTextureID = glGenTextures(1)

        #image_data = cv2.imread('myAvatar.png')[:,:,::-1]
        image_data = avatar

        glBindTexture(GL_TEXTURE_2D, self._imageTextureID)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_data.shape[1], image_data.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._imageTextureID)

        loc = glGetAttribLocation(self._program, "texture1")
        glUniform1f(loc,0)
        #glUniform1ui('texture1', 0)
        #glUniform1i('texture1', 0)
        #self._program.setUniformValue('texture1', 0)
        self.i = 0

        self.video = threading.Thread(target=self._video_loop, daemon=True)
        self.video.start()

    def _face_landmarks(self,image):
        faces = self.detector(image)
        for face in faces:
            shape = self.predictor(image, face)
            if shape.num_parts>0:
                return shape_to_array(shape,image.shape[1],image.shape[0])
            else:
                break

        return None

    def _video_loop(self):
        cam = cv2.VideoCapture(0)
        frame_size = (420, 240)
        prev_landmarks = None
        avg_move = None
        avg_index = 0
        count_avg = 0
        mc = 2

        while True:
            ret, frame_in = cam.read()
            frame_in = cv2.resize(frame_in, dsize=frame_size)
            landmarks = self._face_landmarks(frame_in)
            if landmarks is not None:
                if prev_landmarks is None:
                    prev_landmarks = landmarks
                    avg_move = np.zeros((mc,landmarks.shape[0],landmarks.shape[1]))
                    continue
                else:
                    avg_move[avg_index,:,:] = landmarks
                    avg_index += 1
                    count_avg += 1
                    if avg_index >= mc:
                        avg_index = 0
                    if count_avg>mc:
                        with self._lock:
                            avg = np.average(avg_move,axis=0)
                            move = avg - prev_landmarks
                            #move = np.clip(move,)
                            self.move = move
                            #prev_landmarks= avg
                            #print(self.move)
                        #self.move = None




    def resizeGL(self, width, height):
        # this tells openGL how many pixels it should be drawing into
        glViewport(0, 0, width, height)

    def read_texture(self, filename):
        print('trying to open', filename)
        img = cv2.imread(filename)[:,:,::-1]

        textureID = glGenTextures(1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glBindTexture(GL_TEXTURE_2D, textureID)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0],
                     0, GL_RGB, GL_UNSIGNED_BYTE, img)
        return textureID

    def paintGL(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.bufferID)
        with self._lock:
            move = self.move
        data = self.get_triangulation(move)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        glClearColor(0, 0.2, 0.3, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, len(data)//4)
        #glBindVertexArray(self.__vao)
        #glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)


def resizeGL(self, width, height):
    pass


def paintGL(self):
    pass

# this initializes Qt
app = QApplication([])
window = OpenGLView()
window.show()
app.exec_()
