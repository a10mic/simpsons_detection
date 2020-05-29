# import the necessary packages
from config import model_config as config
from helper.utils.tfannotations import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os
import tensorflow.compat.v1.app as app


def main():
    # open classes file
    f = open(config.CLASSES_FILE, "w")

    for (name, i) in config.CLASSES.items():
        # construct the class information and write to file
        # id starts from 1 , 0 is reserved for background
        item = (
                "item {\n"
                "\tid: " + str(i) + "\n"
                                    "\tname: " + name + "\n"
                                                        "}\n")
        f.write(item)

    # close the output classes file
    f.close()

    # initializing a data dictionary used to map each image filename
    # to all bounding boxes associated with the image, then load
    # the contents of annotations file
    D = {}
    rows = open(config.ANNOT_PATH).read().strip().split("\n")

    for row in rows[1:]:
        row = row.split(',')
        (imagePath, label, startX, startY, endX, endY) = row
        (startX, startY, endX, endY) = (float(startX), float(startY), float(endX), float(endY))

        if label not in config.CLASSES:
            continue

        path = os.path.sep.join([config.BASE_PATH, imagePath])
        b = D.get(path, [])

        b.append((label, (startX, startY, endX, endY)))
        D[path] = b

    # create train and test splits
    (trainKeys, testKeys) = train_test_split(list(D.keys()), test_size=config.TEST_SIZE, random_state=42)

    datasets = [
        ("train", trainKeys, config.TRAIN_RECORD),
        ("test", testKeys, config.TEST_RECORD)
    ]

    for (dType, keys, outputPath) in datasets:
        print(f'processing {dType}')
        writer = tf.io.TFRecordWriter(outputPath)

        total = 0  # variable to store no of images traversed

        for path in keys:
            encoded = tf.io.gfile.GFile(path, "rb").read()
            encoded = bytes(encoded)

            pilImage = Image.open(path)
            w, h = pilImage.size
            filename = path.split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".") + 1:]

            tfAnnot = TFAnnotation()
            tfAnnot.image = encoded
            tfAnnot.encoding = encoding
            tfAnnot.filename = filename
            tfAnnot.width = w
            tfAnnot.height = h

            for (label, (startX, startY, endX, endY)) in D[path]:
                # scaling coordinates to lie in range(0,1) as tensorflow accept in this format
                xMin = startX / w
                xMax = endX / w
                yMin = startY / h
                yMax = endY / h

                # update the bounding boxes + labels lists
                tfAnnot.xMins.append(xMin)
                tfAnnot.xMaxs.append(xMax)
                tfAnnot.yMins.append(yMin)
                tfAnnot.yMaxs.append(yMax)
                tfAnnot.textLabels.append(label.encode("utf8"))
                tfAnnot.classes.append(config.CLASSES[label])
                tfAnnot.difficult.append(0)
                # increment the total number of examples
                total += 1
            # encode the data point attributes using the TensorFlow
            # helper functions
            features = tf.train.Features(feature=tfAnnot.build())
            example = tf.train.Example(features=features)
            # add the example to the writer
            writer.write(example.SerializeToString())
        # close the writer and print info
        writer.close()
        print(f"[INFO] {total} examples saved for ’{dType}’")


if __name__ == "__main__":
    main()
