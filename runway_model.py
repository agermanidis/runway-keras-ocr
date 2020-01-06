import runway
import keras_ocr
import numpy as np


@runway.setup
def setup():
    return keras_ocr.pipeline.Pipeline()


@runway.command('recognize', inputs={'image': runway.image}, outputs={'bboxes': runway.array(runway.image_bounding_box), 'labels': runway.array(runway.text)})
def recognize(model, inputs):
    width, height = inputs['image'].size
    predictions = model.recognize(np.array(inputs['image']))
    labels = []
    bboxes = []
    for label, bbox in predictions:
        labels.append(label)
        p1, p2, p3, _ = bbox
        bboxes.append([
            p1[0] / width,
            p1[1] / height,
            p2[0] / width,
            p3[1] / height
        ])
    return {'labels': labels, 'bboxes': bboxes}


if __name__ == "__main__":
    runway.run()