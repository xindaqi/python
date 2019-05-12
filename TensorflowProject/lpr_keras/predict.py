from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, Input
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from keras.utils.vis_utils import model_to_dot
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


from IPython.display import SVG

from genplate import *

# %matplotlib inline

np.random.seed(5)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"
             ]

M_strIdx = dict(zip(chars, range(len(chars))))
print("string dict: {}".format(M_strIdx))
def show_image():
    n_generate = 100
    rows = 20
    cols = int(n_generate/rows)

    G = GenPlate("./font/platech.ttf",'./font/platechar.ttf',"./NoPlates")
    l_plateStr,l_plateImg = G.genBatch(100,2,range(31,65),"./plate_test_one",(272,72))
    # print("plate string: {}".format(l_plateStr))
    l_out = []
    fig = plt.figure(figsize=(10, 10))
    for i in range(rows):
        l_tmp = []
        for j in range(cols):
            l_tmp.append(l_plateImg[i*cols+j])
        l_out.append(np.hstack(l_tmp))
        # fig = plt.figure(figsize=(10, 10))
        ax  = fig.add_subplot(111)
        ax.imshow( np.vstack(l_out), aspect="auto" )
    plt.show()

def generate_test_img():
    G = GenPlate("./font/platech.ttf",'./font/platechar.ttf',"./NoPlates")
    l_plateStr,l_plateImg = G.genBatch(100,2,range(31,65),"./plate_test_one",(272,72))

def generate_image(batch_size=32):
    while True:
        l_plateStr,l_plateImg = G.genBatch(batch_size,2,range(31,65),"./plate",(272,72))
        X = np.array(l_plateImg, dtype=np.uint8)
        ytmp = np.array(list(map(lambda x: [M_strIdx[a] for a in list(x)], l_plateStr)), dtype=np.uint8)
        y = np.zeros([ytmp.shape[1], batch_size, len(chars)])
        for batch in range(batch_size):
            for idx, row_i in enumerate(ytmp[batch]):
                y[idx, batch, row_i] = 1
        yield X, [yy for yy in y]
def train():
    adam = Adam(lr=0.001)

    input_tensor = Input((72, 272, 3))
    x = input_tensor
    for i in range(3):
        x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
        x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)

    n_class = len(chars)
    x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(7)]
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    SVG(model_to_dot(model=model, show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))

    best_model = ModelCheckpoint('chepai_best.h5', monitor='val_loss', verbose=0, save_best_only=True)
    model.fit_generator(generate_image(32), steps_per_epoch=2000, epochs=5,
                            validation_data=generate_image(32), validation_steps=1280,
                            callbacks=[best_model])
def prediction():
    M_idxStr = dict(zip(range(len(chars)), chars))
    # input_tensor = Input((72, 272, 3))
    # x = input_tensor
    # for i in range(3):
    #     x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
    #     x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
    #     x = MaxPool2D(pool_size=(2, 2))(x)

    # x = Flatten()(x)
    # x = Dropout(0.25)(x)

    # n_class = len(chars)
    # x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(7)]
    # model = Model(inputs=input_tensor, outputs=x)
    model = load_model("chepai_best.h5")
    G = GenPlate("./font/platech.ttf",'./font/platechar.ttf',"./NoPlates")
    l_plateStr,l_plateImg = G.genBatch(40,2,range(31,65),"./plate_test_predict",(272,72))
    myfont = FontProperties(fname="./font/Lantinghei.ttc")
    fig = plt.figure(figsize=(12, 12))
    result = [np.argmax(result) for result in model.predict(np.array(l_plateImg))]
    # print("predict result: {}".format(result))
    l_titles = list(map(lambda x: "".join([M_idxStr[xx] for xx in x]), np.argmax(np.array(model.predict( np.array(l_plateImg) )), 2).T))
    # l_titles = list(map(lambda x: "".join(M_idxStr[x]), result))
    print("plate info: {}".format(l_titles))
    for idx, img in enumerate(l_plateImg[0:40]):
        ax = fig.add_subplot(10, 4, idx+1)
        ax.imshow(img)
        ax.set_title(l_titles[idx], fontproperties=myfont)
        ax.set_axis_off()
    plt.show()

if __name__ == "__main__":
    # train()
    prediction()